#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import { createHash } from 'crypto';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

type CourtCase = {
  id: string;
  family: 'mistral' | 'llama';
  dimension: number;
  layer: number;
  units: number[];
  outputChannel: number;
  seed: number;
  caseSpec: string;
  childVariant: string;
  baseSnapshot: string;
};

type CourtRow = {
  id: string;
  family: CourtCase['family'];
  dimension: number;
  layer: number;
  unitCount: number;
  outputChannel: number | null;
  quantizationStep: number;
  baseSnapshot: string;
  baseSnapshotSha256: string;
  reportPath: string;
  pass: boolean;
  r2: number;
  mae: number;
  worstError: number;
  subspaceMinPrincipalCosine: number;
  pairedAtomMinJointCosine: number;
  pairedAtomCountAbove099: number;
  rawOracleCalls: number;
  accessInvariant: boolean;
  failures: string[];
};

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-asymptotic-structure';
const sourceLane = 'nn-live-blind-swiglu-layer-expansion';
const defaultVariant = 'mistral4096d-llama3072d-16unit-q000001-oracle-only';
const quantizationStep = 1e-6;
const unitCount = 16;
const maxOracleCalls = 120000;

const cases: CourtCase[] = [
  {
    id: 'mistral-layer9-output2048',
    family: 'mistral',
    dimension: 4096,
    layer: 9,
    units: [6396, 6290, 3681, 5380, 6149, 558, 5485, 9989, 9998, 11864, 2097, 8926, 8910, 2266, 8108, 2280],
    outputChannel: 2048,
    seed: 260726,
    caseSpec: 'mistral:9:16:6396+6290+3681+5380+6149+558+5485+9989+9998+11864+2097+8926+8910+2266+8108+2280:0.999:0.001:0.999:0.005',
    childVariant: 'mistral-layer9-output2048-top16-q000001-asymptotic-structure-court-oracle-only',
    baseSnapshot: 'artifacts/mistral-layer9-output2048-top16-q1e-6-richardson-xlarge-cubic-final.npz'
  },
  {
    id: 'mistral-layer15-output2048',
    family: 'mistral',
    dimension: 4096,
    layer: 15,
    units: [7885, 2037, 13787, 2761, 5013, 3496, 5314, 9185, 2434, 2487, 223, 5617, 12804, 5572, 12373, 10618],
    outputChannel: 2048,
    seed: 260727,
    caseSpec: 'mistral:15:16:7885+2037+13787+2761+5013+3496+5314+9185+2434+2487+223+5617+12804+5572+12373+10618:0.999:0.001:0.999:0.005',
    childVariant: 'mistral-layer15-output2048-top16-q000001-asymptotic-structure-court-oracle-only',
    baseSnapshot: 'artifacts/mistral-layer15-output2048-top16-q1e-6-pair-replay.npz'
  },
  {
    id: 'llama-layer9-selected16',
    family: 'llama',
    dimension: 3072,
    layer: 9,
    units: [958, 1532, 1647, 1765, 2755, 3182, 3856, 4191, 4371, 4373, 5442, 7098, 7145, 7305, 7679, 7685],
    outputChannel: -1,
    seed: 260728,
    caseSpec: 'llama:9:16:958+1532+1647+1765+2755+3182+3856+4191+4371+4373+5442+7098+7145+7305+7679+7685:0.999:0.001:0.999:0.005',
    childVariant: 'llama-layer9-16unit-q000001-asymptotic-structure-court-oracle-only',
    baseSnapshot: 'artifacts/llm-reader/snapshots/llama-layer9-16unit-base.npz'
  }
];

function asRecord(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function asNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be a finite number`);
  return value;
}

function asBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`);
  return value;
}

function sha256(filePath: string): string {
  return createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function sourceReportPath(testCase: CourtCase): string {
  return path.join(repoRoot, 'benchmarks', 'release', sourceLane, testCase.childVariant, 'report.json');
}

function runCase(testCase: CourtCase): void {
  const snapshotPath = path.join(repoRoot, testCase.baseSnapshot);
  if (!fs.existsSync(snapshotPath)) throw new Error(`Missing base snapshot for ${testCase.id}: ${testCase.baseSnapshot}`);
  const artifactRoot = path.join(repoRoot, 'artifacts', 'llm-reader', 'asymptotic-structure-court');
  fs.mkdirSync(artifactRoot, { recursive: true });
  const env = {
    ...process.env,
    GCL_PYTHON: process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python',
    GCL_BENCHMARK_VARIANT: testCase.childVariant,
    GCL_LAYER_EXPANSION_CASES: testCase.caseSpec,
    GCL_LAYER_EXPANSION_ORACLE_QUANTIZATION_STEP: String(quantizationStep),
    GCL_LAYER_EXPANSION_OUTPUT_CHANNEL: String(testCase.outputChannel),
    GCL_LAYER_EXPANSION_DISABLE_ADAPTIVE_RESCUE: '1',
    GCL_LAYER_EXPANSION_MAX_ORACLE_CALLS: String(maxOracleCalls),
    GCL_LAYER_EXPANSION_PROGRESS: process.env.GCL_LAYER_EXPANSION_PROGRESS || '1',
    GCL_BLIND_SWIGLU_SEED: String(testCase.seed),
    GCL_BLIND_SWIGLU_FIT_SNAPSHOT_INPUT: testCase.baseSnapshot,
    GCL_BLIND_SWIGLU_FINAL_FIT_SNAPSHOT_OUTPUT: path.relative(
      repoRoot,
      path.join(artifactRoot, `${testCase.id}-fit.npz`)
    ),
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_SNAPSHOT_OUTPUT: path.relative(
      repoRoot,
      path.join(artifactRoot, `${testCase.id}-structure.npz`)
    ),
    GCL_BLIND_SWIGLU_SWEEP_MIN_PAIRED_ATOM_COSINE: '0.99',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH: '1',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_METHOD: 'asymptotic-gate-stitch',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_ON_CUBIC: '1',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_TRAIN_LIMIT: '16000',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_TAIL_COUNT: '256',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_TAIL_REPEAT: '1',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_POOLS: '4',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_SELECTION_HOLDOUT_POOLS: '2',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_SELECTION_INCLUDE_FULL: '1',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_COUNT: '2048',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_TAIL_COUNT: '256',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH_HOLDOUT_TAIL_REPEAT: '1',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_COUNT: '24',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SAMPLES_PER_LINE: '1537',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE: '6',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_OFFSET_SCALE: '0.75',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SCALE: '512',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_NORMAL_DELTA: '0.10',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES: '5',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_CALIBRATION: '1',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_REFINE: '1',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_VALUE_POLISH: '0',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_STEPS: '160',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_RESTARTS: '4',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_LR: '0.8',
    GCL_BLIND_SWIGLU_STRUCTURE_REFINE_STEPS: '180',
    GCL_BLIND_SWIGLU_STRUCTURE_REFINE_RESTARTS: '2',
    GCL_BLIND_SWIGLU_STRUCTURE_REFINE_LR: '0.6'
  };
  const result = spawnSync('ts-node', ['scripts/benchmark-nn-live-blind-swiglu-layer-expansion.ts'], {
    cwd: repoRoot,
    env,
    stdio: 'inherit',
    timeout: 20 * 60 * 1000,
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${testCase.id} exited with status ${String(result.status)}`);
}

function loadRow(testCase: CourtCase): CourtRow {
  const reportPath = sourceReportPath(testCase);
  if (!fs.existsSync(reportPath)) throw new Error(`Missing court report for ${testCase.id}: ${path.relative(repoRoot, reportPath)}`);
  const report = asRecord(JSON.parse(fs.readFileSync(reportPath, 'utf8')), `${testCase.id}.report`);
  const summary = asRecord(report.summary, `${testCase.id}.summary`);
  const access = asRecord(report.accessAudit, `${testCase.id}.accessAudit`);
  const r2 = asNumber(summary.minR2, `${testCase.id}.minR2`);
  const mae = asNumber(summary.maxMae, `${testCase.id}.maxMae`);
  const worstError = asNumber(summary.maxWorstError, `${testCase.id}.maxWorstError`);
  const subspace = asNumber(summary.minSubspacePrincipalCosine, `${testCase.id}.minSubspacePrincipalCosine`);
  const pairedCosine = asNumber(summary.minPairedAtomJointCosine, `${testCase.id}.minPairedAtomJointCosine`);
  const pairedCount = asNumber(summary.minPairedAtomCountAboveThreshold, `${testCase.id}.minPairedAtomCountAboveThreshold`);
  const rawOracleCalls = asNumber(summary.totalRawOracleCalls, `${testCase.id}.totalRawOracleCalls`);
  const accessInvariant = asBoolean(summary.accessInvariant, `${testCase.id}.accessInvariant`)
    && access.solverAccess === 'oracle-only'
    && access.weightsUsedBySolver === false
    && access.hooksUsedBySolver === false
    && access.sourceIdsUsedBySolver === false
    && access.postHocWitnessOnly === true;
  const failures: string[] = [];
  if (!asBoolean(summary.pass, `${testCase.id}.pass`)) failures.push('source report pass=false');
  if (asNumber(summary.caseCount, `${testCase.id}.caseCount`) !== 1) failures.push('caseCount must equal 1');
  if (asNumber(summary.maxExpandedUnitCount, `${testCase.id}.maxExpandedUnitCount`) !== unitCount) {
    failures.push(`unitCount must equal ${unitCount}`);
  }
  if (r2 < 0.999) failures.push(`R2=${r2}<0.999`);
  if (mae > 0.001) failures.push(`MAE=${mae}>0.001`);
  if (worstError > 0.005) failures.push(`worstError=${worstError}>0.005`);
  if (subspace < 0.999) failures.push(`subspace=${subspace}<0.999`);
  if (pairedCosine < 0.99) failures.push(`pairedAtomMinJointCosine=${pairedCosine}<0.99`);
  if (pairedCount !== unitCount) failures.push(`pairedAtomCountAbove099=${pairedCount}<${unitCount}`);
  if (rawOracleCalls > maxOracleCalls) failures.push(`rawOracleCalls=${rawOracleCalls}>${maxOracleCalls}`);
  if (!accessInvariant) failures.push('oracle-only access invariant failed');
  const snapshotPath = path.join(repoRoot, testCase.baseSnapshot);
  return {
    id: testCase.id,
    family: testCase.family,
    dimension: testCase.dimension,
    layer: testCase.layer,
    unitCount: testCase.units.length,
    outputChannel: testCase.outputChannel >= 0 ? testCase.outputChannel : null,
    quantizationStep,
    baseSnapshot: testCase.baseSnapshot,
    baseSnapshotSha256: sha256(snapshotPath),
    reportPath: path.relative(repoRoot, reportPath),
    pass: failures.length === 0,
    r2,
    mae,
    worstError,
    subspaceMinPrincipalCosine: subspace,
    pairedAtomMinJointCosine: pairedCosine,
    pairedAtomCountAbove099: pairedCount,
    rawOracleCalls,
    accessInvariant,
    failures
  };
}

function fmt(value: number): string {
  if (value === 0) return '0';
  if (Math.abs(value) < 1e-4 || Math.abs(value) >= 1000) return value.toExponential(6);
  return value.toFixed(12).replace(/0+$/, '').replace(/\.$/, '');
}

const regenerate = process.env.GCL_ASYMPTOTIC_STRUCTURE_REGENERATE !== '0';
if (regenerate) {
  for (const testCase of cases) {
    console.log(`[ASYMPTOTIC-STRUCTURE] regenerate ${testCase.id}`);
    runCase(testCase);
  }
}

const rows = cases.map(loadRow);
const failures = rows.flatMap(row => row.failures.map(failure => `${row.id}: ${failure}`));
const summary = {
  pass: failures.length === 0,
  regenerated: regenerate,
  caseCount: rows.length,
  passedCases: rows.filter(row => row.pass).length,
  modelFamilies: [...new Set(rows.map(row => row.family))].sort(),
  dimensions: [...new Set(rows.map(row => row.dimension))].sort((a, b) => a - b),
  unitCount,
  quantizationStep,
  minR2: Math.min(...rows.map(row => row.r2)),
  maxMae: Math.max(...rows.map(row => row.mae)),
  maxWorstError: Math.max(...rows.map(row => row.worstError)),
  minSubspacePrincipalCosine: Math.min(...rows.map(row => row.subspaceMinPrincipalCosine)),
  minPairedAtomJointCosine: Math.min(...rows.map(row => row.pairedAtomMinJointCosine)),
  minPairedAtomCountAbove099: Math.min(...rows.map(row => row.pairedAtomCountAbove099)),
  maxRawOracleCallsPerCase: Math.max(...rows.map(row => row.rawOracleCalls)),
  totalRawOracleCalls: rows.reduce((sum, row) => sum + row.rawOracleCalls, 0),
  accessInvariant: rows.every(row => row.accessInvariant),
  failures
};

const output = resolveBenchmarkOutput(repoRoot, lane, defaultVariant);
fs.mkdirSync(output.outputDir, { recursive: true });
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  lane,
  sourceLane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  rows,
  accessAudit: {
    solverAccess: 'oracle-only',
    weightsUsedBySolver: false,
    hooksUsedBySolver: false,
    sourceIdsUsedBySolver: false,
    postHocWitnessOnly: true,
    sourceReportsChecked: true
  }
};
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# Live Blind SwiGLU Asymptotic Structure Court',
    '',
    `Status: **${summary.pass ? 'PASS' : 'FAIL'}**`,
    '',
    `- regenerated: \`${summary.regenerated}\``,
    `- cases: \`${summary.passedCases}/${summary.caseCount}\``,
    `- model families: \`${summary.modelFamilies.join(', ')}\``,
    `- dimensions: \`${summary.dimensions.join(', ')}\``,
    `- units per scalar block: \`${summary.unitCount}\``,
    `- output quantization step: \`${summary.quantizationStep}\``,
    `- min R2: \`${fmt(summary.minR2)}\``,
    `- max MAE: \`${fmt(summary.maxMae)}\``,
    `- max worst error: \`${fmt(summary.maxWorstError)}\``,
    `- min subspace cosine: \`${fmt(summary.minSubspacePrincipalCosine)}\``,
    `- min paired gate/up joint cosine: \`${fmt(summary.minPairedAtomJointCosine)}\``,
    `- min paired atoms above 0.99: \`${summary.minPairedAtomCountAbove099}/${summary.unitCount}\``,
    `- max raw oracle calls per case: \`${summary.maxRawOracleCallsPerCase}\``,
    `- total raw oracle calls: \`${summary.totalRawOracleCalls}\``,
    `- access invariant: \`${summary.accessInvariant}\``,
    '',
    '| case | family | dimension | layer | R2 | MAE | worst | paired min | pairs >=0.99 | calls | pass |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ...rows.map(row => `| ${row.id} | ${row.family} | ${row.dimension} | ${row.layer} | ${fmt(row.r2)} | ${fmt(row.mae)} | ${fmt(row.worstError)} | ${fmt(row.pairedAtomMinJointCosine)} | ${row.pairedAtomCountAbove099}/${row.unitCount} | ${row.rawOracleCalls} | ${row.pass} |`),
    '',
    failures.length ? `## Failures\n\n${failures.map(failure => `- ${failure}`).join('\n')}\n` : ''
  ].join('\n'),
  'utf8'
);
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Runs snapshot-backed oracle-only asymptotic gate stitching and trust-region structure refinement on real Mistral/LLaMA-family scalar SwiGLU blocks.',
    'The gate requires 16/16 post-hoc paired gate/up atoms above joint cosine 0.99, R2 >= 0.999, explicit finite precision, a 120000-call per-case budget, and the oracle-only access invariant.'
  ]
});

console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
