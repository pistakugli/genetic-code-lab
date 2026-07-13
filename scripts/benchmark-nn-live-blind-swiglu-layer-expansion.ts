#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Family = 'mistral' | 'llama';

type ExpansionCase = {
  family: Family;
  layer: number;
  unitCount: number;
  units: number[];
  activeRank: number;
  minR2: number;
  maxMae: number;
  maxWorstError: number;
  minSubspaceCosine: number;
};

type FamilySpec = {
  family: Family;
  model: string;
  modelLabel: string;
  familyLabel: string;
  dimension: number;
  layerCount: number;
  intermediateSize: number;
};

type SweepSummary = {
  pass: boolean;
  caseCount: number;
  passedCases: number;
  failedCases: number;
  minR2: number;
  maxMae: number;
  maxWorstError: number;
  minSubspacePrincipalCosine: number;
  minPairedAtomJointCosine: number | null;
  minPairedAtomCountAboveThreshold: number | null;
  pairedAtomThreshold: number | null;
  maxRawOracleCallsPerCase: number;
  totalRawOracleCalls: number;
  maxUnitCount: number;
  maxAtomCount: number;
  accessInvariant: boolean;
};

type SweepReport = {
  summary: SweepSummary;
  cases: Array<{
    case: {
      id: string;
      layer: number;
      unit: number;
      units: number[];
      activeRank: number;
      atomCount: number;
    };
    variant: string;
    reportPath: string;
    pass: boolean;
    failures: string[];
    summary?: unknown;
  }>;
  accessAudit?: Record<string, unknown>;
  selection?: Record<string, unknown>;
};

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-layer-expansion';
const defaultCaseSpec = 'mistral:20:16:-';

const familySpecs: Record<Family, FamilySpec> = {
  mistral: {
    family: 'mistral',
    model: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
    modelLabel: 'mistral-4096d',
    familyLabel: 'Mistral-family',
    dimension: 4096,
    layerCount: 32,
    intermediateSize: 14336
  },
  llama: {
    family: 'llama',
    model: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    modelLabel: 'llama-3072d',
    familyLabel: 'LLaMA-family',
    dimension: 3072,
    layerCount: 28,
    intermediateSize: 8192
  }
};

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

function deterministicSpread(count: number, intermediateSize: number, salt: number): number[] {
  if (!Number.isInteger(count) || count <= 0) throw new Error(`invalid unit count ${count}`);
  const anchors = [0, 17, 123, 456, 789, 1024, 2048, 4096, 6144, 8191, 10000, 12288, 14000]
    .filter(unit => unit >= 0 && unit < intermediateSize);
  const units = new Set<number>();
  for (const unit of anchors) {
    units.add(unit);
    if (units.size >= count) return [...units];
  }
  let state = (salt % 2147483647) || 260627;
  while (units.size < count) {
    state = (state * 48271) % 2147483647;
    units.add(state % intermediateSize);
  }
  return [...units].sort((a, b) => a - b).slice(0, count);
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function parseExpansionCase(raw: string): ExpansionCase {
  const [familyRaw, layerRaw, unitCountRaw, unitsRaw, minR2Raw, maxMaeRaw, minSubspaceRaw, maxWorstRaw] = raw.split(':');
  const family = familyRaw as Family;
  const spec = familySpecs[family];
  if (!spec) throw new Error(`unknown family ${familyRaw}; expected mistral or llama`);
  const layer = Number(layerRaw);
  const unitCount = Number(unitCountRaw);
  if (!Number.isInteger(layer) || layer < 0 || layer >= spec.layerCount) {
    throw new Error(`invalid ${family} layer ${layerRaw}; expected 0..${spec.layerCount - 1}`);
  }
  if (!Number.isInteger(unitCount) || unitCount <= 0 || unitCount > spec.intermediateSize) {
    throw new Error(`invalid ${family} unitCount ${unitCountRaw}`);
  }
  const explicitUnits = unitsRaw && unitsRaw.trim() && unitsRaw !== '-'
    ? unitsRaw.split(/[+|,]/).map(value => Number(value.trim())).filter(Number.isInteger)
    : [];
  const units = explicitUnits.length
    ? [...new Set(explicitUnits)]
    : deterministicSpread(unitCount, spec.intermediateSize, layer + unitCount * 131);
  if (units.length !== unitCount) throw new Error(`${raw} resolved ${units.length} unique units, expected ${unitCount}`);
  for (const unit of units) {
    if (unit < 0 || unit >= spec.intermediateSize) {
      throw new Error(`unit ${unit} outside ${family} intermediate size ${spec.intermediateSize}`);
    }
  }
  return {
    family,
    layer,
    unitCount,
    units,
    activeRank: Number(envValue('GCL_LAYER_EXPANSION_ACTIVE_RANK', String(unitCount * 2))),
    minR2: minR2Raw ? Number(minR2Raw) : Number(envValue('GCL_LAYER_EXPANSION_MIN_R2', '0.995')),
    maxMae: maxMaeRaw ? Number(maxMaeRaw) : Number(envValue('GCL_LAYER_EXPANSION_MAX_MAE', '0.015')),
    maxWorstError: maxWorstRaw ? Number(maxWorstRaw) : Number(envValue('GCL_LAYER_EXPANSION_MAX_WORST_ERROR', '0.12')),
    minSubspaceCosine: minSubspaceRaw ? Number(minSubspaceRaw) : Number(envValue('GCL_LAYER_EXPANSION_MIN_SUBSPACE_COSINE', '0.98'))
  };
}

function generateRandomExpansionCases(count: number): ExpansionCase[] {
  const family = envValue('GCL_LAYER_EXPANSION_RANDOM_FAMILY', 'mistral') as Family;
  const spec = familySpecs[family];
  if (!spec) throw new Error(`unknown random expansion family ${family}; expected mistral or llama`);
  const unitCount = Number(envValue('GCL_LAYER_EXPANSION_RANDOM_UNIT_COUNT', '16'));
  if (!Number.isInteger(count) || count <= 0) throw new Error(`invalid random expansion count ${count}`);
  if (!Number.isInteger(unitCount) || unitCount <= 0 || unitCount > spec.intermediateSize) {
    throw new Error(`invalid random expansion unit count ${unitCount}`);
  }
  const random = makeRandom(Number(envValue('GCL_LAYER_EXPANSION_RANDOM_SEED', '260627')));
  const cases: ExpansionCase[] = [];
  const seen = new Set<string>();
  while (cases.length < count) {
    const layer = Math.floor(random() * spec.layerCount);
    const unitSet = new Set<number>();
    while (unitSet.size < unitCount) unitSet.add(Math.floor(random() * spec.intermediateSize));
    const units = [...unitSet].sort((a, b) => a - b);
    const key = `${family}:${layer}:${units.join('+')}`;
    if (seen.has(key)) continue;
    seen.add(key);
    cases.push({
      family,
      layer,
      unitCount,
      units,
      activeRank: Number(envValue('GCL_LAYER_EXPANSION_ACTIVE_RANK', String(unitCount * 2))),
      minR2: Number(envValue('GCL_LAYER_EXPANSION_MIN_R2', '0.995')),
      maxMae: Number(envValue('GCL_LAYER_EXPANSION_MAX_MAE', '0.015')),
      maxWorstError: Number(envValue('GCL_LAYER_EXPANSION_MAX_WORST_ERROR', '0.12')),
      minSubspaceCosine: Number(envValue('GCL_LAYER_EXPANSION_MIN_SUBSPACE_COSINE', '0.98'))
    });
  }
  return cases;
}

function resolveExpansionCases(): ExpansionCase[] {
  const randomCount = Number(process.env.GCL_LAYER_EXPANSION_RANDOM_COUNT ?? '0');
  if (randomCount > 0) return generateRandomExpansionCases(randomCount);
  const raw = envValue('GCL_LAYER_EXPANSION_CASES', defaultCaseSpec);
  return raw.split(';').map(item => item.trim()).filter(Boolean).map(parseExpansionCase);
}

function caseToSweepCase(testCase: ExpansionCase): string {
  return [
    testCase.layer,
    testCase.units.join('+'),
    testCase.minR2,
    testCase.maxMae,
    testCase.minSubspaceCosine,
    testCase.maxWorstError
  ].join(':');
}

function expansionVariant(testCase: ExpansionCase): string {
  const spec = familySpecs[testCase.family];
  const quantizationStep = Number(envValue('GCL_LAYER_EXPANSION_ORACLE_QUANTIZATION_STEP', '0'));
  const quantizationTag = quantizationStep > 0
    ? `-q${String(quantizationStep).replace(/^0\./, '').replace(/\./g, '')}`
    : '';
  return `${spec.modelLabel}-layer${testCase.layer}-${testCase.unitCount}unit${quantizationTag}-expansion-oracle-only`;
}

function sweepAggregatePath(variant: string): string {
  return path.join(repoRoot, 'benchmarks', 'release', 'nn-live-blind-swiglu-oracle-decompiler', variant, 'report.json');
}

const cases = resolveExpansionCases();
if (!cases.length) throw new Error('layer expansion requires at least one case');

const defaultVariant = cases.length === 1
  ? `${familySpecs[cases[0].family].family}-layer${cases[0].layer}-${cases[0].unitCount}unit-expansion-oracle-only`
  : 'multi-case-layer-expansion-oracle-only';
const output = resolveBenchmarkOutput(repoRoot, lane, envValue('GCL_BENCHMARK_VARIANT', defaultVariant));
fs.mkdirSync(output.outputDir, { recursive: true });
for (const staleOutput of [output.jsonOutputPath, output.markdownOutputPath, output.manifestOutputPath]) {
  fs.rmSync(staleOutput, { force: true });
}

const caseResults: Array<{
  case: ExpansionCase;
  sweepVariant: string;
  sweepReportPath: string;
  pass: boolean;
  failures: string[];
  sweepSummary?: SweepSummary;
}> = [];

for (const testCase of cases) {
  const spec = familySpecs[testCase.family];
  const sweepVariant = expansionVariant(testCase);
  const sweepCases = caseToSweepCase(testCase);
  const highAtomProfile = testCase.unitCount >= 16;
  const llamaHighAtomProfile = testCase.family === 'llama' && highAtomProfile;
  const defaultTimeoutMs = highAtomProfile ? '3600000' : '1800000';
  const defaultTrainCount = highAtomProfile ? String(Math.max(16000, testCase.unitCount * 1000)) : String(Math.max(14400, testCase.unitCount * 1400));
  const defaultValidationCount = highAtomProfile ? '4096' : '3072';
  const defaultFitSteps = highAtomProfile ? '2600' : '3200';
  const defaultFitRestarts = highAtomProfile ? '512' : '768';
  const defaultProbePoints = llamaHighAtomProfile ? String(Math.max(33, testCase.activeRank + 1)) : String(Math.max(17, testCase.activeRank + 1));
  const defaultInputScale = llamaHighAtomProfile ? '0.35' : (testCase.family === 'llama' ? '0.65' : '0.45');
  const defaultValidationScale = llamaHighAtomProfile ? '0.35' : '0.65';
  const defaultFitGradientCount = '256';
  const defaultFitGradientWeight = highAtomProfile ? '1.0' : '0.2';
  const defaultMaxOracleCalls = highAtomProfile ? '300000' : '180000';
  const env = {
    ...process.env,
    GCL_PYTHON: resolveGclPython(),
    GCL_BLIND_SWIGLU_TIMEOUT_MS: envValue('GCL_LAYER_EXPANSION_TIMEOUT_MS', defaultTimeoutMs),
    GCL_BLIND_SWIGLU_MODEL: spec.model,
    GCL_BLIND_SWIGLU_MODEL_LABEL: spec.modelLabel,
    GCL_BLIND_SWIGLU_FAMILY_LABEL: spec.familyLabel,
    GCL_BLIND_SWIGLU_EXPECTED_DIMENSION: String(spec.dimension),
    GCL_BLIND_SWIGLU_LAYER_COUNT: String(spec.layerCount),
    GCL_BLIND_SWIGLU_INTERMEDIATE_SIZE: String(spec.intermediateSize),
    GCL_BLIND_SWIGLU_ORACLE_QUANTIZATION_STEP: envValue('GCL_LAYER_EXPANSION_ORACLE_QUANTIZATION_STEP', '0'),
    GCL_BLIND_SWIGLU_SWEEP_VARIANT: sweepVariant,
    GCL_BLIND_SWIGLU_SWEEP_CASES: sweepCases,
    GCL_BLIND_SWIGLU_ACTIVE_RANK: String(testCase.activeRank),
    GCL_BLIND_SWIGLU_GRADIENT_MODE: envValue('GCL_LAYER_EXPANSION_GRADIENT_MODE', 'forward-coordinate'),
    GCL_BLIND_SWIGLU_BASIS_MOMENT: envValue('GCL_LAYER_EXPANSION_BASIS_MOMENT', 'centered'),
    GCL_BLIND_SWIGLU_PROBE_POINTS: envValue('GCL_LAYER_EXPANSION_PROBE_POINTS', defaultProbePoints),
    GCL_BLIND_SWIGLU_CUBIC_DIRECTION_COUNT: envValue('GCL_LAYER_EXPANSION_CUBIC_DIRECTION_COUNT', '0'),
    GCL_BLIND_SWIGLU_CUBIC_STEP: envValue('GCL_LAYER_EXPANSION_CUBIC_STEP', '0.35'),
    GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEP: envValue('GCL_LAYER_EXPANSION_FINITE_DIFFERENCE_STEP', '0.04'),
    GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEPS: envValue('GCL_LAYER_EXPANSION_FINITE_DIFFERENCE_STEPS', ''),
    GCL_BLIND_SWIGLU_INPUT_SCALE: envValue('GCL_LAYER_EXPANSION_INPUT_SCALE', defaultInputScale),
    GCL_BLIND_SWIGLU_TRAIN_SCALE_MIXTURE: envValue('GCL_LAYER_EXPANSION_TRAIN_SCALE_MIXTURE', ''),
    GCL_BLIND_SWIGLU_TRAIN_SAMPLING_MODE: envValue('GCL_LAYER_EXPANSION_TRAIN_SAMPLING_MODE', 'gaussian'),
    GCL_BLIND_SWIGLU_VALIDATION_SCALE: envValue('GCL_LAYER_EXPANSION_VALIDATION_SCALE', defaultValidationScale),
    GCL_BLIND_SWIGLU_TRAIN_COUNT: envValue('GCL_LAYER_EXPANSION_TRAIN_COUNT', defaultTrainCount),
    GCL_BLIND_SWIGLU_VALIDATION_COUNT: envValue('GCL_LAYER_EXPANSION_VALIDATION_COUNT', defaultValidationCount),
    GCL_BLIND_SWIGLU_FIT_STEPS: envValue('GCL_LAYER_EXPANSION_FIT_STEPS', defaultFitSteps),
    GCL_BLIND_SWIGLU_FIT_RESTARTS: envValue('GCL_LAYER_EXPANSION_FIT_RESTARTS', defaultFitRestarts),
    GCL_BLIND_SWIGLU_FIT_RESTART_CHUNK_SIZE: envValue('GCL_LAYER_EXPANSION_FIT_RESTART_CHUNK_SIZE', '128'),
    GCL_BLIND_SWIGLU_FIT_MODE: envValue('GCL_LAYER_EXPANSION_FIT_MODE', 'batched'),
    GCL_BLIND_SWIGLU_FIT_GRADIENT_COUNT: envValue('GCL_LAYER_EXPANSION_FIT_GRADIENT_COUNT', defaultFitGradientCount),
    GCL_BLIND_SWIGLU_FIT_GRADIENT_STEP: envValue('GCL_LAYER_EXPANSION_FIT_GRADIENT_STEP', '0.04'),
    GCL_BLIND_SWIGLU_FIT_GRADIENT_STEPS: envValue('GCL_LAYER_EXPANSION_FIT_GRADIENT_STEPS', ''),
    GCL_BLIND_SWIGLU_FIT_GRADIENT_WEIGHT: envValue('GCL_LAYER_EXPANSION_FIT_GRADIENT_WEIGHT', defaultFitGradientWeight),
    GCL_BLIND_SWIGLU_FIT_SELECTION_COUNT: envValue('GCL_LAYER_EXPANSION_FIT_SELECTION_COUNT', '0'),
    GCL_BLIND_SWIGLU_FIT_SELECTION_SCALE: envValue('GCL_LAYER_EXPANSION_FIT_SELECTION_SCALE', '0'),
    GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT: envValue('GCL_LAYER_EXPANSION_SELECTION_TAIL_WEIGHT', '0'),
    GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE: envValue('GCL_LAYER_EXPANSION_SELECTION_TAIL_QUANTILE', '0.995'),
    GCL_BLIND_SWIGLU_SELECTION_WORST_WEIGHT: envValue('GCL_LAYER_EXPANSION_SELECTION_WORST_WEIGHT', '0'),
    GCL_BLIND_SWIGLU_COUNTEREXAMPLE_COUNT: envValue('GCL_LAYER_EXPANSION_COUNTEREXAMPLE_COUNT', '0'),
    GCL_BLIND_SWIGLU_COUNTEREXAMPLE_REPEAT: envValue('GCL_LAYER_EXPANSION_COUNTEREXAMPLE_REPEAT', '4'),
    GCL_BLIND_SWIGLU_COUNTEREXAMPLE_JITTER_COUNT: envValue('GCL_LAYER_EXPANSION_COUNTEREXAMPLE_JITTER_COUNT', '0'),
    GCL_BLIND_SWIGLU_COUNTEREXAMPLE_JITTER_SCALE: envValue('GCL_LAYER_EXPANSION_COUNTEREXAMPLE_JITTER_SCALE', '0.025'),
    GCL_BLIND_SWIGLU_TAIL_PROBE_COUNT: envValue('GCL_LAYER_EXPANSION_TAIL_PROBE_COUNT', '0'),
    GCL_BLIND_SWIGLU_TAIL_PROBE_REPEAT: envValue('GCL_LAYER_EXPANSION_TAIL_PROBE_REPEAT', '4'),
    GCL_BLIND_SWIGLU_TAIL_PROBE_SCALE: envValue('GCL_LAYER_EXPANSION_TAIL_PROBE_SCALE', '0'),
    GCL_BLIND_SWIGLU_RESCUE_TRAIN_TARGET: envValue('GCL_LAYER_EXPANSION_RESCUE_TRAIN_TARGET', '0'),
    GCL_BLIND_SWIGLU_RESCUE_GRADIENT_TARGET: envValue('GCL_LAYER_EXPANSION_RESCUE_GRADIENT_TARGET', '0'),
    GCL_BLIND_SWIGLU_RESCUE_STEPS: envValue('GCL_LAYER_EXPANSION_RESCUE_STEPS', '0'),
    GCL_BLIND_SWIGLU_RESCUE_RESTARTS: envValue('GCL_LAYER_EXPANSION_RESCUE_RESTARTS', '0'),
    GCL_BLIND_SWIGLU_DISABLE_ADAPTIVE_RESCUE: envValue('GCL_LAYER_EXPANSION_DISABLE_ADAPTIVE_RESCUE', llamaHighAtomProfile ? '1' : '0'),
    GCL_BLIND_SWIGLU_PROGRESS: envValue('GCL_LAYER_EXPANSION_PROGRESS', '0'),
    GCL_BLIND_SWIGLU_POLISH_TOP_K: envValue('GCL_LAYER_EXPANSION_POLISH_TOP_K', '1'),
    GCL_BLIND_SWIGLU_BACKFIT_CYCLES: envValue('GCL_LAYER_EXPANSION_BACKFIT_CYCLES', '0'),
    GCL_BLIND_SWIGLU_TAIL_POLISH_STEPS: envValue('GCL_LAYER_EXPANSION_TAIL_POLISH_STEPS', llamaHighAtomProfile ? '1100' : '420'),
    GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT: envValue('GCL_LAYER_EXPANSION_TAIL_POLISH_WEIGHT', llamaHighAtomProfile ? '2.2' : '0.35'),
    GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION: envValue('GCL_LAYER_EXPANSION_TAIL_POLISH_FRACTION', llamaHighAtomProfile ? '0.30' : '0.125'),
    GCL_BLIND_SWIGLU_TAIL_POLISH_LR: envValue('GCL_LAYER_EXPANSION_TAIL_POLISH_LR', llamaHighAtomProfile ? '0.003' : '0.006'),
    GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS: envValue('GCL_LAYER_EXPANSION_TAIL_POLISH_LOSS', 'topk'),
    GCL_BLIND_SWIGLU_ROBUST_SCALAR_CALIBRATION: envValue('GCL_LAYER_EXPANSION_ROBUST_SCALAR_CALIBRATION', llamaHighAtomProfile ? '1' : '0'),
    GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_STEPS: envValue('GCL_LAYER_EXPANSION_SCALAR_CALIBRATION_STEPS', '900'),
    GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_LR: envValue('GCL_LAYER_EXPANSION_SCALAR_CALIBRATION_LR', '0.01'),
    GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_WEIGHT: envValue('GCL_LAYER_EXPANSION_SCALAR_CALIBRATION_TAIL_WEIGHT', llamaHighAtomProfile ? '2.0' : '1.0'),
    GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_FRACTION: envValue('GCL_LAYER_EXPANSION_SCALAR_CALIBRATION_TAIL_FRACTION', llamaHighAtomProfile ? '0.25' : '0.20'),
    GCL_BLIND_SWIGLU_SWEEP_MIN_R2: String(testCase.minR2),
    GCL_BLIND_SWIGLU_SWEEP_MAX_MAE: String(testCase.maxMae),
    GCL_BLIND_SWIGLU_SWEEP_MAX_WORST_ERROR: String(testCase.maxWorstError),
    GCL_BLIND_SWIGLU_SWEEP_MIN_SUBSPACE_COSINE: String(testCase.minSubspaceCosine),
    GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS: envValue('GCL_LAYER_EXPANSION_MAX_ORACLE_CALLS', defaultMaxOracleCalls)
  };

  console.log(`[EXPAND] ${spec.familyLabel} layer=${testCase.layer} unitCount=${testCase.unitCount} units=${testCase.units.join(',')}`);
  const result = spawnSync('ts-node', ['scripts/benchmark-nn-live-blind-swiglu-oracle-decompiler-sweep.ts'], {
    cwd: repoRoot,
    stdio: 'inherit',
    env,
    timeout: Number(env.GCL_BLIND_SWIGLU_TIMEOUT_MS),
    maxBuffer: 128 * 1024 * 1024
  });
  const sweepReportPath = sweepAggregatePath(sweepVariant);
  const failures: string[] = [];
  let sweepSummary: SweepSummary | undefined;
  if (!fs.existsSync(sweepReportPath)) failures.push('missing sweep report.json');
  else {
    const sweepReport = JSON.parse(fs.readFileSync(sweepReportPath, 'utf8')) as SweepReport;
    sweepSummary = sweepReport.summary;
    if (!sweepSummary.pass) failures.push('sweep summary pass=false');
    if (!sweepSummary.accessInvariant) failures.push('accessInvariant=false');
    if (sweepSummary.maxUnitCount < testCase.unitCount) failures.push(`maxUnitCount=${sweepSummary.maxUnitCount}<${testCase.unitCount}`);
  }
  if (result.error) failures.push(String(result.error));
  if (result.status !== 0 && failures.length === 0) failures.push(`sweep exit ${result.status ?? 'unknown'}`);
  caseResults.push({
    case: testCase,
    sweepVariant,
    sweepReportPath: path.relative(repoRoot, sweepReportPath),
    pass: failures.length === 0,
    failures,
    sweepSummary
  });
}

const summaries = caseResults.map(result => result.sweepSummary).filter((summary): summary is SweepSummary => Boolean(summary));
const summary = {
  pass: caseResults.every(result => result.pass),
  caseCount: caseResults.length,
  passedCases: caseResults.filter(result => result.pass).length,
  failedCases: caseResults.filter(result => !result.pass).length,
  maxExpandedUnitCount: summaries.length ? Math.max(...summaries.map(item => item.maxUnitCount)) : 0,
  minR2: summaries.length ? Math.min(...summaries.map(item => item.minR2)) : null,
  maxMae: summaries.length ? Math.max(...summaries.map(item => item.maxMae)) : null,
  maxWorstError: summaries.length ? Math.max(...summaries.map(item => item.maxWorstError)) : null,
  minSubspacePrincipalCosine: summaries.length ? Math.min(...summaries.map(item => item.minSubspacePrincipalCosine)) : null,
  minPairedAtomJointCosine: summaries.length
    ? Math.min(...summaries.map(item => item.minPairedAtomJointCosine ?? Number.POSITIVE_INFINITY))
    : null,
  minPairedAtomCountAboveThreshold: summaries.length
    ? Math.min(...summaries.map(item => item.minPairedAtomCountAboveThreshold ?? 0))
    : null,
  pairedAtomThreshold: summaries.find(item => item.pairedAtomThreshold !== null)?.pairedAtomThreshold ?? null,
  totalRawOracleCalls: summaries.reduce((sum, item) => sum + item.totalRawOracleCalls, 0),
  accessInvariant: caseResults.every(result => result.sweepSummary?.accessInvariant === true),
  solverAccess: 'oracle-only',
  weightsUsedBySolver: false,
  hooksUsedBySolver: false,
  postHocWitnessOnly: true
};

const report = {
  summary,
  cases: caseResults,
  config: {
    caseSpec: process.env.GCL_LAYER_EXPANSION_RANDOM_COUNT
      ? null
      : envValue('GCL_LAYER_EXPANSION_CASES', defaultCaseSpec),
    randomCount: Number(process.env.GCL_LAYER_EXPANSION_RANDOM_COUNT ?? '0'),
    randomSeed: process.env.GCL_LAYER_EXPANSION_RANDOM_COUNT
      ? Number(envValue('GCL_LAYER_EXPANSION_RANDOM_SEED', '260627'))
      : null,
    randomFamily: process.env.GCL_LAYER_EXPANSION_RANDOM_COUNT
      ? envValue('GCL_LAYER_EXPANSION_RANDOM_FAMILY', 'mistral')
      : null,
    randomUnitCount: process.env.GCL_LAYER_EXPANSION_RANDOM_COUNT
      ? Number(envValue('GCL_LAYER_EXPANSION_RANDOM_UNIT_COUNT', '16'))
      : null,
    defaultVariant,
    gradientMode: envValue('GCL_LAYER_EXPANSION_GRADIENT_MODE', 'forward-coordinate')
  },
  accessAudit: {
    solverAccess: 'oracle-only',
    weightsUsedBySolver: false,
    hooksUsedBySolver: false,
    sourceIdsUsedBySolver: false,
    postHocWitnessOnly: true,
    delegatedSweepReportsEnforceInvariant: true
  }
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# Live Blind SwiGLU Layer Expansion',
    '',
    `- pass: \`${String(summary.pass)}\``,
    `- cases: \`${summary.passedCases}/${summary.caseCount}\``,
    `- max expanded unit count: \`${summary.maxExpandedUnitCount}\``,
    `- min R2: \`${summary.minR2}\``,
    `- max MAE: \`${summary.maxMae}\``,
    `- max worst error: \`${summary.maxWorstError}\``,
    `- min subspace principal cosine: \`${summary.minSubspacePrincipalCosine}\``,
    `- min post-hoc paired atom joint cosine: \`${summary.minPairedAtomJointCosine}\``,
    `- min paired atoms above threshold: \`${summary.minPairedAtomCountAboveThreshold}\``,
    `- total raw oracle calls: \`${summary.totalRawOracleCalls}\``,
    `- access invariant: \`${summary.accessInvariant}\``,
    '',
    ...caseResults.map(result => `- ${result.case.family} layer ${result.case.layer} ${result.case.unitCount}-unit: \`${result.pass ? 'pass' : `fail ${result.failures.join('; ')}`}\` (${result.sweepReportPath})`),
    ''
  ].join('\n'),
  'utf8'
);
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Runs configured real-model multi-unit SwiGLU layer expansion through the oracle-only sweep runner.',
    'This gate is intended to push beyond selected 8-unit blocks while preserving solver access invariants.'
  ]
});

console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
