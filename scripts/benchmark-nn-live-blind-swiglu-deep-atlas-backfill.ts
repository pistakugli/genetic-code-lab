#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type ModelFamily = 'Mistral-family' | 'LLaMA-family';

type BackfillCase = {
  family: ModelFamily;
  model: string;
  dimension: number;
  layerCount: number;
  intermediateSize: number;
  layer: number;
  unit: number;
  variant: string;
};

type CaseResult = BackfillCase & {
  reportPath: string;
  pass: boolean;
  status: number | null;
  r2?: number;
  mae?: number;
  worstError?: number;
  subspaceMinPrincipalCosine?: number;
  rawOracleCalls?: number;
  skippedExisting: boolean;
};

type BackfillArtifactAudit = {
  variant: string;
  reportPath: string;
  pass: boolean;
  modelType?: string;
  layer?: number;
  units?: number[];
  r2?: number;
  subspaceMinPrincipalCosine?: number;
  rawOracleCalls?: number;
};

type BackfillMode = 'missing' | 'all';

type FamilySpec = {
  family: ModelFamily;
  model: string;
  dimension: number;
  layerCount: number;
  intermediateSize: number;
};

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-deep-atlas-backfill';
const defaultVariant = 'mistral-llama-full-layer-single-unit-backfill';
const atlasPath = path.join(
  repoRoot,
  'benchmarks',
  'release',
  'nn-live-blind-swiglu-deep-atlas',
  'mistral-llama-multilayer-swiglu-atlas',
  'report.json'
);

const familySpecs: FamilySpec[] = [
  {
    family: 'Mistral-family',
    model: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
    dimension: 4096,
    layerCount: 32,
    intermediateSize: 14336
  },
  {
    family: 'LLaMA-family',
    model: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    dimension: 3072,
    layerCount: 28,
    intermediateSize: 8192
  }
];

function envValue(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const gpuPython = path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
  if (fs.existsSync(gpuPython)) return gpuPython;
  return 'python3';
}

function existingLayersByFamily(): Map<ModelFamily, Set<number>> {
  if (!fs.existsSync(atlasPath)) {
    spawnChecked('ts-node', ['scripts/benchmark-nn-live-blind-swiglu-deep-atlas.ts'], process.env);
  }
  const report = JSON.parse(fs.readFileSync(atlasPath, 'utf8'));
  const map = new Map<ModelFamily, Set<number>>();
  for (const family of report.summary.familySummaries ?? []) {
    map.set(family.modelFamily, new Set(family.uniqueLayers));
  }
  return map;
}

function selectedFamilies(): Set<ModelFamily> {
  const raw = envValue('GCL_ATLAS_BACKFILL_FAMILIES', 'Mistral-family,LLaMA-family');
  return new Set(raw.split(',').map(item => item.trim()).filter(Boolean) as ModelFamily[]);
}

function selectedLayerFilter(): Set<number> | null {
  const raw = process.env.GCL_ATLAS_BACKFILL_LAYERS;
  if (!raw || !raw.trim()) return null;
  const layers = raw.split(',')
    .map(item => Number(item.trim()))
    .filter(layer => Number.isInteger(layer) && layer >= 0);
  if (layers.length === 0) throw new Error(`GCL_ATLAS_BACKFILL_LAYERS=${raw} did not contain valid non-negative integer layers`);
  return new Set(layers);
}

function deterministicUnit(family: ModelFamily, layer: number): number {
  if (family === 'Mistral-family') return (3125 + layer * 7919) % 14336;
  return (1786 + layer * 3571) % 8192;
}

function backfillMode(): BackfillMode {
  const mode = envValue('GCL_ATLAS_BACKFILL_MODE', 'missing');
  if (mode === 'missing' || mode === 'all') return mode;
  throw new Error(`Unsupported GCL_ATLAS_BACKFILL_MODE=${mode}; expected missing or all`);
}

function buildCases(): BackfillCase[] {
  const present = existingLayersByFamily();
  const families = selectedFamilies();
  const layers = selectedLayerFilter();
  const mode = backfillMode();
  const cases: BackfillCase[] = [];
  for (const spec of familySpecs) {
    if (!families.has(spec.family)) continue;
    const have = present.get(spec.family) ?? new Set<number>();
    for (let layer = 0; layer < spec.layerCount; layer++) {
      if (layers && !layers.has(layer)) continue;
      if (mode === 'missing' && have.has(layer)) continue;
      const unit = deterministicUnit(spec.family, layer);
      const prefix = spec.family === 'Mistral-family' ? 'mistral-4096d' : 'llama-3072d';
      cases.push({
        ...spec,
        layer,
        unit,
        variant: `${prefix}-layer${layer}-unit${unit}-atlas-backfill-oracle-only`
      });
    }
  }
  const limit = Number(envValue('GCL_ATLAS_BACKFILL_LIMIT', '0'));
  return Number.isFinite(limit) && limit > 0 ? cases.slice(0, limit) : cases;
}

function hardCaseEnv(testCase: BackfillCase): Record<string, string> {
  if (testCase.family === 'Mistral-family' && testCase.layer === 23 && testCase.unit === 13230) {
    return {
      GCL_BLIND_SWIGLU_PROBE_POINTS: envValue('GCL_ATLAS_BACKFILL_HARD_PROBE_POINTS', '7'),
      GCL_BLIND_SWIGLU_TRAIN_COUNT: envValue('GCL_ATLAS_BACKFILL_HARD_TRAIN_COUNT', '2400'),
      GCL_BLIND_SWIGLU_VALIDATION_COUNT: envValue('GCL_ATLAS_BACKFILL_HARD_VALIDATION_COUNT', '768'),
      GCL_BLIND_SWIGLU_FIT_STEPS: envValue('GCL_ATLAS_BACKFILL_HARD_FIT_STEPS', '1600'),
      GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS: envValue('GCL_ATLAS_BACKFILL_HARD_MAX_ORACLE_CALLS', '70000'),
      GCL_BLIND_SWIGLU_MIN_R2: envValue('GCL_ATLAS_BACKFILL_HARD_MIN_R2', '0.985'),
      GCL_BLIND_SWIGLU_MIN_SUBSPACE_COSINE: envValue('GCL_ATLAS_BACKFILL_HARD_MIN_SUBSPACE_COSINE', '0.999')
    };
  }
  return {};
}

function sourceReportPath(variant: string): string {
  return path.join(repoRoot, 'benchmarks', 'release', 'nn-live-blind-swiglu-oracle-decompiler', variant, 'report.json');
}

function spawnChecked(command: string, args: string[], env: NodeJS.ProcessEnv): void {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: 'inherit',
    env,
    timeout: Number(envValue('GCL_ATLAS_BACKFILL_TIMEOUT_MS', '1800000')),
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} ${args.join(' ')} failed with status ${result.status}`);
}

function runCase(testCase: BackfillCase): CaseResult {
  const reportPath = sourceReportPath(testCase.variant);
  if (fs.existsSync(reportPath) && envValue('GCL_ATLAS_BACKFILL_FORCE', '0') !== '1') {
    return readResult(testCase, reportPath, true, 0);
  }
  const env = {
    ...process.env,
    GCL_PYTHON: resolveGclPython(),
    GCL_BENCHMARK_VARIANT: testCase.variant,
    GCL_BLIND_SWIGLU_MODEL: testCase.model,
    GCL_BLIND_SWIGLU_LAYER: String(testCase.layer),
    GCL_BLIND_SWIGLU_UNIT: String(testCase.unit),
    GCL_BLIND_SWIGLU_ACTIVE_RANK: '2',
    GCL_BLIND_SWIGLU_GRADIENT_MODE: 'coordinate',
    GCL_BLIND_SWIGLU_PROBE_POINTS: envValue('GCL_ATLAS_BACKFILL_PROBE_POINTS', '2'),
    GCL_BLIND_SWIGLU_TRAIN_COUNT: envValue('GCL_ATLAS_BACKFILL_TRAIN_COUNT', '1600'),
    GCL_BLIND_SWIGLU_VALIDATION_COUNT: envValue('GCL_ATLAS_BACKFILL_VALIDATION_COUNT', '512'),
    GCL_BLIND_SWIGLU_FIT_STEPS: envValue('GCL_ATLAS_BACKFILL_FIT_STEPS', '1200'),
    GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS: envValue('GCL_ATLAS_BACKFILL_MAX_ORACLE_CALLS', '25000'),
    GCL_BLIND_SWIGLU_MIN_R2: envValue('GCL_ATLAS_BACKFILL_MIN_R2', '0.985'),
    GCL_BLIND_SWIGLU_MIN_SUBSPACE_COSINE: envValue('GCL_ATLAS_BACKFILL_MIN_SUBSPACE_COSINE', '0.93'),
    ...hardCaseEnv(testCase)
  };
  const result = spawnSync('ts-node', ['scripts/benchmark-nn-live-blind-swiglu-oracle-decompiler.ts'], {
    cwd: repoRoot,
    stdio: 'inherit',
    env,
    timeout: Number(envValue('GCL_ATLAS_BACKFILL_TIMEOUT_MS', '1800000')),
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;
  return readResult(testCase, reportPath, false, result.status);
}

function readResult(testCase: BackfillCase, reportPath: string, skippedExisting: boolean, status: number | null): CaseResult {
  if (!fs.existsSync(reportPath)) {
    return {
      ...testCase,
      reportPath: path.relative(repoRoot, reportPath),
      pass: false,
      status,
      skippedExisting
    };
  }
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  // `status` was recorded and then never consulted: pass came from the report on disk
  // alone. So a child that OOMed or hit the timeout after an earlier successful run left
  // report.json in place scored as a pass, and the lane claimed a backfill it never
  // performed. A run that actually executed must agree with its own exit code; a skipped
  // case has no exit code to agree with, which is why skippedExisting is excluded here and
  // counted separately in the summary.
  const exitCodeAgrees = skippedExisting || status === 0;
  return {
    ...testCase,
    reportPath: path.relative(repoRoot, reportPath),
    pass: Boolean(report.summary?.pass) && exitCodeAgrees,
    reportPass: Boolean(report.summary?.pass),
    exitCodeAgrees,
    status,
    r2: report.summary?.metrics?.r2,
    mae: report.summary?.metrics?.mae,
    worstError: report.summary?.metrics?.worstError,
    subspaceMinPrincipalCosine: report.summary?.sourceAlignment?.subspaceMinPrincipalCosine,
    rawOracleCalls: report.summary?.rawOracleCalls,
    skippedExisting
  };
}

function coverageAudit(refreshedAtlas: any): Array<{
  family: ModelFamily;
  expectedLayerCount: number;
  observedLayerCount: number;
  coverageRatio: number;
  missingLayers: number[];
  complete: boolean;
}> {
  const summaries = new Map<ModelFamily, any>();
  for (const family of refreshedAtlas?.familySummaries ?? []) {
    summaries.set(family.modelFamily, family);
  }
  return familySpecs
    .filter(spec => selectedFamilies().has(spec.family))
    .map(spec => {
      const summary = summaries.get(spec.family);
      const observed = new Set<number>(summary?.uniqueLayers ?? []);
      const missingLayers = Array.from({ length: spec.layerCount }, (_, layer) => layer)
        .filter(layer => !observed.has(layer));
      return {
        family: spec.family,
        expectedLayerCount: spec.layerCount,
        observedLayerCount: observed.size,
        coverageRatio: spec.layerCount > 0 ? observed.size / spec.layerCount : 0,
        missingLayers,
        complete: missingLayers.length === 0
      };
    });
}

function discoverBackfillArtifactAudits(): BackfillArtifactAudit[] {
  const root = path.join(repoRoot, 'benchmarks', 'release', 'nn-live-blind-swiglu-oracle-decompiler');
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root)
    .filter(variant => variant.endsWith('-atlas-backfill-oracle-only'))
    .map(variant => {
      const reportPath = path.join(root, variant, 'report.json');
      if (!fs.existsSync(reportPath)) {
        return {
          variant,
          reportPath: path.relative(repoRoot, reportPath),
          pass: false
        };
      }
      const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
      return {
        variant,
        reportPath: path.relative(repoRoot, reportPath),
        pass: Boolean(report.summary?.pass),
        modelType: report.summary?.modelType,
        layer: report.summary?.layer,
        units: report.summary?.units,
        r2: report.summary?.metrics?.r2,
        subspaceMinPrincipalCosine: report.summary?.sourceAlignment?.subspaceMinPrincipalCosine,
        rawOracleCalls: report.summary?.rawOracleCalls
      };
    })
    .sort((a, b) => a.variant.localeCompare(b.variant));
}

const output = resolveBenchmarkOutput(repoRoot, lane, defaultVariant);
fs.mkdirSync(output.outputDir, { recursive: true });

const cases = buildCases();
const results: CaseResult[] = [];
const mode = backfillMode();
for (const testCase of cases) {
  console.log(`[BACKFILL] ${testCase.family} layer=${testCase.layer} unit=${testCase.unit} variant=${testCase.variant}`);
  results.push(runCase(testCase));
}

const failed = results.filter(result => !result.pass);
if (failed.length === 0) {
  spawnChecked('ts-node', ['scripts/benchmark-nn-live-blind-swiglu-deep-atlas.ts'], process.env);
}

const refreshedAtlas = fs.existsSync(atlasPath)
  ? JSON.parse(fs.readFileSync(atlasPath, 'utf8')).summary
  : null;
const coverage = coverageAudit(refreshedAtlas);
const coverageComplete = coverage.every(item => item.complete);
const backfillArtifacts = discoverBackfillArtifactAudits();
const failedBackfillArtifacts = backfillArtifacts.filter(item => !item.pass);
const summary = {
  pass: failed.length === 0 && failedBackfillArtifacts.length === 0 && coverageComplete && Boolean(refreshedAtlas?.pass),
  mode,
  requestedCases: cases.length,
  passedCases: results.filter(result => result.pass).length,
  skippedExistingCases: results.filter(result => result.skippedExisting).length,
  existingBackfillArtifactCount: backfillArtifacts.length,
  passedBackfillArtifactCount: backfillArtifacts.filter(item => item.pass).length,
  failedBackfillArtifacts: failedBackfillArtifacts.map(item => ({
    variant: item.variant,
    reportPath: item.reportPath
  })),
  failedCases: failed.map(result => ({
    family: result.family,
    layer: result.layer,
    unit: result.unit,
    variant: result.variant,
    status: result.status
  })),
  coverageComplete,
  coverage,
  refreshedAtlas
};
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  lane,
  mode,
  status: summary.pass ? 'pass' : 'needs-fix',
  cases,
  results,
  backfillArtifacts,
  summary,
  boundaries: [
    'This runner audits, backfills, or regenerates configured deep-atlas layers with real-model oracle-only single-unit SwiGLU recoveries.',
    'Default mode is missing-layer gap fill. Set GCL_ATLAS_BACKFILL_MODE=all with GCL_ATLAS_BACKFILL_FORCE=1 for a full deterministic configured-layer rerun.',
    'It expands layer coverage; it does not claim all-channel FFN extraction.',
    'If a case fails, the required action is to repair that case or add a stronger oracle-only recovery configuration, not to hide the layer gap.'
  ]
};
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# NN Live Blind SwiGLU Deep Atlas Backfill',
    '',
    `Status: **${summary.pass ? 'PASS' : 'NEEDS FIX'}**`,
    '',
    `Mode: ${summary.mode}`,
    `Requested cases: ${summary.requestedCases}`,
    `Passed cases: ${summary.passedCases}`,
    `Skipped existing cases: ${summary.skippedExistingCases}`,
    `Existing backfill artifacts: ${summary.existingBackfillArtifactCount}`,
    `Passed backfill artifacts: ${summary.passedBackfillArtifactCount}`,
    `Coverage complete: ${summary.coverageComplete}`,
    '',
    '## Coverage Audit',
    '',
    '| family | observed layers | expected layers | coverage | missing layers |',
    '| --- | ---: | ---: | ---: | --- |',
    ...coverage.map(item => `| ${item.family} | ${item.observedLayerCount} | ${item.expectedLayerCount} | ${item.coverageRatio} | ${item.missingLayers.length ? item.missingLayers.join(',') : '-'} |`),
    '',
    '## Results',
    '',
    '| family | layer | unit | pass | R2 | subspace | report |',
    '| --- | ---: | ---: | --- | ---: | ---: | --- |',
    ...results.map(result => `| ${result.family} | ${result.layer} | ${result.unit} | ${result.pass} | ${result.r2 ?? '-'} | ${result.subspaceMinPrincipalCosine ?? '-'} | ${result.reportPath} |`),
    '',
    '## Existing Backfill Artifacts',
    '',
    '| model | layer | units | pass | R2 | subspace | raw calls | report |',
    '| --- | ---: | --- | --- | ---: | ---: | ---: | --- |',
    ...backfillArtifacts.map(item => `| ${item.modelType ?? '-'} | ${item.layer ?? '-'} | ${item.units?.join(',') ?? '-'} | ${item.pass} | ${item.r2 ?? '-'} | ${item.subspaceMinPrincipalCosine ?? '-'} | ${item.rawOracleCalls ?? '-'} | ${item.reportPath} |`),
    '',
    '## Boundaries',
    '',
    ...report.boundaries.map(item => `- ${item}`)
  ].join('\n')
);
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: report.boundaries
});

console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
