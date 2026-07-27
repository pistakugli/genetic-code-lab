#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-asymptotic-cold-start';
const regenerate = process.env.GCL_COLD_START_REGENERATE !== '0';
const variant = regenerate
  ? 'mistral4096d-layer15-16unit-q000001-fresh-oracle-only'
  : 'mistral4096d-layer15-16unit-q000001-artifact-audit';
const childVariant = 'mistral-layer15-output2048-top16-q000001-asymptotic-cold-start-fresh-oracle-only';
const childReportPath = path.join(
  repoRoot,
  'benchmarks',
  'release',
  'nn-live-blind-swiglu-layer-expansion',
  childVariant,
  'report.json'
);
const units = [7885, 2037, 13787, 2761, 5013, 3496, 5314, 9185, 2434, 2487, 223, 5617, 12804, 5572, 12373, 10618];
const maxOracleCalls = 1_500_000;

function record(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function numberValue(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be boolean`);
  return value;
}

function makeChildEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of ['PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']) {
    if (process.env[key] !== undefined) env[key] = process.env[key];
  }
  Object.assign(env, {
    GCL_PYTHON: process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python',
    GCL_BENCHMARK_VARIANT: childVariant,
    GCL_LAYER_EXPANSION_CASES: `mistral:15:16:${units.join('+')}:0.999:0.001:0.999:0.005`,
    GCL_LAYER_EXPANSION_ORACLE_QUANTIZATION_STEP: '0.000001',
    GCL_LAYER_EXPANSION_OUTPUT_CHANNEL: '2048',
    GCL_LAYER_EXPANSION_DISABLE_ADAPTIVE_RESCUE: '1',
    GCL_LAYER_EXPANSION_MAX_ORACLE_CALLS: String(maxOracleCalls),
    GCL_LAYER_EXPANSION_PROGRESS: process.env.GCL_LAYER_EXPANSION_PROGRESS || '1',
    GCL_LAYER_EXPANSION_PROBE_POINTS: '65',
    GCL_LAYER_EXPANSION_GRADIENT_MODE: 'symmetric-multiscale-coordinate',
    GCL_LAYER_EXPANSION_BASIS_MOMENT: 'centered',
    GCL_LAYER_EXPANSION_FINITE_DIFFERENCE_STEPS: '1.0,2.0',
    GCL_LAYER_EXPANSION_CUBIC_DIRECTION_COUNT: '2048',
    GCL_LAYER_EXPANSION_CUBIC_HOLDOUT_COUNT: '512',
    GCL_LAYER_EXPANSION_CUBIC_STEPS: '0.35,0.25',
    GCL_LAYER_EXPANSION_TRAIN_COUNT: '16000',
    GCL_LAYER_EXPANSION_TRAIN_SAMPLING_MODE: 'gaussian',
    GCL_LAYER_EXPANSION_INPUT_SCALE: '0.45',
    GCL_LAYER_EXPANSION_VALIDATION_COUNT: '4096',
    GCL_LAYER_EXPANSION_VALIDATION_SCALE: '0.65',
    GCL_LAYER_EXPANSION_FIT_GRADIENT_COUNT: '256',
    GCL_LAYER_EXPANSION_FIT_GRADIENT_MODE: 'forward',
    GCL_LAYER_EXPANSION_FIT_GRADIENT_STEP: '0.04',
    GCL_LAYER_EXPANSION_FIT_GRADIENT_STEPS: '0.04',
    GCL_LAYER_EXPANSION_FIT_SELECTION_COUNT: '4096',
    GCL_LAYER_EXPANSION_FIT_SELECTION_SCALE: '0.65',
    GCL_BLIND_SWIGLU_SEED: '260723',
    GCL_BLIND_SWIGLU_RECOVERY_PATH: 'asymptotic-cold-start',
    GCL_BLIND_SWIGLU_INDEPENDENT_RNG: '1',
    GCL_BLIND_SWIGLU_SWEEP_MIN_PAIRED_ATOM_COSINE: '0.99',
    GCL_BLIND_SWIGLU_VALIDATION_FULL_TAIL_POLISH: '0',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_COUNT: '48',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SAMPLES_PER_LINE: '1537',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_LINE_RANGE: '6',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_OFFSET_SCALE: '0.75',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_SCALE: '512',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_NORMAL_DELTA: '0.10',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_HESSIAN_REPRESENTATIVES: '8',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_MIN_GATE_SUPPORT: '4',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_MIN_UP_SUPPORT: '3',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_GATE_NORM: '0.25',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_COLD_START_ORIENTATION_SAMPLES: '4096',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_CALIBRATION: '1',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_STRUCTURE_REFINE: '1',
    GCL_BLIND_SWIGLU_ASYMPTOTIC_STITCH_VALUE_POLISH: '0',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_STEPS: '200',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_RESTARTS: '5',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_LR: '0.8',
    GCL_BLIND_SWIGLU_GATE_NORM_CALIBRATION_MAX_NORM_RATIO: '8',
    GCL_BLIND_SWIGLU_STRUCTURE_REFINE_STEPS: '220',
    GCL_BLIND_SWIGLU_STRUCTURE_REFINE_RESTARTS: '3',
    GCL_BLIND_SWIGLU_STRUCTURE_REFINE_LR: '0.6',
    GCL_BLIND_SWIGLU_TIMEOUT_MS: '3600000'
  });
  return env;
}

const regenerationStartedAt = Date.now();
if (regenerate) {
  fs.rmSync(childReportPath, { force: true });
  const result = spawnSync('ts-node', ['scripts/benchmark-nn-live-blind-swiglu-layer-expansion.ts'], {
    cwd: repoRoot,
    env: makeChildEnv(),
    stdio: 'inherit',
    timeout: 60 * 60 * 1000,
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`cold-start child exited with status ${String(result.status)}`);
}

if (!fs.existsSync(childReportPath)) throw new Error(`missing fresh cold-start report: ${childReportPath}`);
if (regenerate && fs.statSync(childReportPath).mtimeMs < regenerationStartedAt - 2_000) {
  throw new Error(`cold-start child report was not regenerated by this run: ${childReportPath}`);
}
const child = record(JSON.parse(fs.readFileSync(childReportPath, 'utf8')), 'child report');
const childSummary = record(child.summary, 'child summary');
const childAccess = record(child.accessAudit, 'child access');
const cases = child.cases;
if (!Array.isArray(cases) || cases.length !== 1) throw new Error('cold-start child must contain exactly one case');
const childCase = record(cases[0], 'child case');
const caseSweepSummary = record(childCase.sweepSummary, 'case sweep summary');
const sweepReportPath = String(childCase.sweepReportPath || '');
const sweepPath = path.join(repoRoot, sweepReportPath);
if (!sweepReportPath || !fs.existsSync(sweepPath)) throw new Error(`missing cold-start sweep report: ${sweepReportPath}`);
if (regenerate && fs.statSync(sweepPath).mtimeMs < regenerationStartedAt - 2_000) {
  throw new Error(`cold-start sweep report was not regenerated by this run: ${sweepReportPath}`);
}
const sweep = record(JSON.parse(fs.readFileSync(sweepPath, 'utf8')), 'sweep report');
const sweepCases = sweep.cases;
if (!Array.isArray(sweepCases) || sweepCases.length !== 1) throw new Error('cold-start sweep must contain exactly one case');
const sweepCase = record(sweepCases[0], 'sweep case');
const detailedSummary = record(sweepCase.summary, 'detailed summary');
const metrics = record(detailedSummary.metrics, 'metrics');
const sourceAlignment = record(detailedSummary.sourceAlignment, 'source alignment');
const reportPath = String(sweepCase.reportPath || '');
const detailedPath = path.join(repoRoot, reportPath);
if (!reportPath || !fs.existsSync(detailedPath)) throw new Error(`missing detailed cold-start report: ${reportPath}`);
if (regenerate && fs.statSync(detailedPath).mtimeMs < regenerationStartedAt - 2_000) {
  throw new Error(`detailed cold-start report was not regenerated by this run: ${reportPath}`);
}
const detailed = record(JSON.parse(fs.readFileSync(detailedPath, 'utf8')), 'detailed report');
const solver = record(detailed.solver, 'solver');
const detailedAccess = record(detailed.accessAudit, 'detailed access');
const structure = record(solver.asymptoticStructure, 'asymptotic structure');

const failures: string[] = [];
const r2 = numberValue(metrics.r2, 'R2');
const mae = numberValue(metrics.mae, 'MAE');
const worstError = numberValue(metrics.worstError, 'worst error');
const subspace = numberValue(sourceAlignment.subspaceMinPrincipalCosine, 'subspace cosine');
const pairedMin = numberValue(sourceAlignment.pairedAtomMinJointCosine, 'paired minimum');
const pairedCount = numberValue(sourceAlignment.pairedAtomCountAbove099, 'paired count');
const rawOracleCalls = numberValue(detailedSummary.rawOracleCalls, 'raw oracle calls');
if (!booleanValue(childSummary.pass, 'child pass')) failures.push('child report pass=false');
if (!booleanValue(caseSweepSummary.pass, 'case sweep pass')) failures.push('case sweep report pass=false');
if (solver.recoveryPath !== 'asymptotic-cold-start') failures.push('recoveryPath is not asymptotic-cold-start');
if (booleanValue(solver.nonlinearInitialFitUsed, 'nonlinear initial fit')) failures.push('nonlinear initial fit was used');
if (solver.fitMode !== 'asymptotic-structure-cold-start') failures.push('fitMode is not asymptotic-structure-cold-start');
if (numberValue(solver.fitRestarts, 'fit restarts') !== 0) failures.push('fitRestarts is not zero');
if (numberValue(solver.fitRandomRestarts, 'random fit restarts') !== 0) failures.push('fitRandomRestarts is not zero');
if (!booleanValue(structure.coldStart, 'cold start flag')) failures.push('cold-start structure flag is false');
if (!booleanValue(structure.stitched, 'stitched flag')) failures.push('structure was not stitched');
if (numberValue(structure.clusterCount, 'cluster count') !== 16) failures.push('clusterCount is not 16');
if (!Array.isArray(structure.sourceFallbackAtoms)) failures.push('sourceFallbackAtoms is not an array');
else if (structure.sourceFallbackAtoms.length !== 0) failures.push('source fallback atoms are present');
if (booleanValue(detailedAccess.fitSnapshotUsedBySolver, 'snapshot use')) failures.push('fit snapshot was used');
if (booleanValue(detailedAccess.nonlinearInitialFitUsed, 'access nonlinear initial fit')) failures.push('access audit says nonlinear initial fit was used');
if (r2 < 0.999) failures.push(`R2=${r2}<0.999`);
if (mae > 0.001) failures.push(`MAE=${mae}>0.001`);
if (worstError > 0.005) failures.push(`worstError=${worstError}>0.005`);
if (subspace < 0.999) failures.push(`subspace=${subspace}<0.999`);
if (pairedMin < 0.99) failures.push(`paired minimum=${pairedMin}<0.99`);
if (pairedCount !== 16) failures.push(`paired count=${pairedCount}/16`);
if (rawOracleCalls > maxOracleCalls) failures.push(`oracle calls=${rawOracleCalls}>${maxOracleCalls}`);
if (childAccess.solverAccess !== 'oracle-only' || detailedAccess.weightsUsedBySolver !== false || detailedAccess.hooksUsedBySolver !== false) {
  failures.push('oracle-only weight/hook access invariant failed');
}

const summary = {
  pass: failures.length === 0,
  regenerated: regenerate,
  modelFamily: 'mistral',
  dimension: 4096,
  layer: 15,
  unitCount: 16,
  quantizationStep: 1e-6,
  r2,
  mae,
  worstError,
  subspaceMinPrincipalCosine: subspace,
  pairedAtomMinJointCosine: pairedMin,
  pairedAtomCountAbove099: pairedCount,
  rawOracleCalls,
  maxOracleCalls,
  recoveryPath: solver.recoveryPath,
  nonlinearInitialFitUsed: solver.nonlinearInitialFitUsed,
  fitSnapshotUsedBySolver: detailedAccess.fitSnapshotUsedBySolver,
  failures
};
const output = resolveBenchmarkOutput(repoRoot, lane, variant);
fs.mkdirSync(output.outputDir, { recursive: true });
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  sourceReport: path.relative(repoRoot, childReportPath),
  detailedReport: reportPath,
  accessAudit: {
    solverAccess: 'oracle-only',
    weightsUsedBySolver: false,
    hooksUsedBySolver: false,
    fitSnapshotUsedBySolver: false,
    nonlinearInitialFitUsed: false,
    teacherCaseManifestVisibleToOrchestrator: true,
    postHocTruthOnly: true
  }
};
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Live Blind SwiGLU Asymptotic Cold Start',
  '',
  `Status: **${summary.pass ? 'PASS' : 'FAIL'}**`,
  '',
  `- fresh regeneration: \`${regenerate}\``,
  `- recovery path: \`${summary.recoveryPath}\``,
  `- nonlinear initial fit: \`${summary.nonlinearInitialFitUsed}\``,
  `- fit snapshot used: \`${summary.fitSnapshotUsedBySolver}\``,
  `- R2: \`${r2}\``,
  `- MAE: \`${mae}\``,
  `- worst error: \`${worstError}\``,
  `- subspace cosine: \`${subspace}\``,
  `- paired atom minimum: \`${pairedMin}\``,
  `- paired atoms >= 0.99: \`${pairedCount}/16\``,
  `- raw oracle calls: \`${rawOracleCalls}/${maxOracleCalls}\``,
  '',
  failures.length ? `## Failures\n\n${failures.map(failure => `- ${failure}`).join('\n')}\n` : ''
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Fresh real-model oracle-only recovery. Input fit snapshots and nonlinear initial atom fitting are forbidden.',
    'Teacher unit ids remain visible to the orchestration process; strict opaque-endpoint process isolation is a separate gate.'
  ]
});

console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
