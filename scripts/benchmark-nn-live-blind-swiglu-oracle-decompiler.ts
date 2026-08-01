#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');

function envValue(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

function optionalEnvArg(flag: string, name: string): string[] {
  const value = process.env[name];
  return value && value.trim() ? [flag, value.trim()] : [];
}

const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-blind-swiglu-oracle-decompiler',
  envValue('GCL_BENCHMARK_VARIANT', 'mistral-4096d-single-unit-oracle-only')
);

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const gpuPython = path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
  if (fs.existsSync(gpuPython)) return gpuPython;
  return 'python3';
}

fs.mkdirSync(output.outputDir, { recursive: true });
for (const staleOutput of [output.jsonOutputPath, output.markdownOutputPath, output.manifestOutputPath]) {
  fs.rmSync(staleOutput, { force: true });
}

const toolPath = path.join(repoRoot, 'tools', 'llm_reader', 'blind_swiglu_oracle_decompiler.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing blind SwiGLU oracle decompiler: ${toolPath}`);

const args = [
  toolPath,
  '--model', envValue('GCL_BLIND_SWIGLU_MODEL', 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'),
  '--layer', envValue('GCL_BLIND_SWIGLU_LAYER', '20'),
  '--unit', envValue('GCL_BLIND_SWIGLU_UNIT', '0'),
  ...(process.env.GCL_BLIND_SWIGLU_UNITS ? ['--units', envValue('GCL_BLIND_SWIGLU_UNITS', '')] : []),
  ...optionalEnvArg('--output-channel', 'GCL_BLIND_SWIGLU_OUTPUT_CHANNEL'),
  '--device', envValue('GCL_BLIND_SWIGLU_DEVICE', 'cuda'),
  '--oracle-quantization-step', envValue('GCL_BLIND_SWIGLU_ORACLE_QUANTIZATION_STEP', '0'),
  '--active-rank', envValue('GCL_BLIND_SWIGLU_ACTIVE_RANK', '2'),
  ...(process.env.GCL_BLIND_SWIGLU_ATOM_COUNT ? ['--atom-count', envValue('GCL_BLIND_SWIGLU_ATOM_COUNT', '')] : []),
  '--gradient-mode', envValue('GCL_BLIND_SWIGLU_GRADIENT_MODE', 'coordinate'),
  '--basis-moment', envValue('GCL_BLIND_SWIGLU_BASIS_MOMENT', 'auto'),
  '--probe-points', envValue('GCL_BLIND_SWIGLU_PROBE_POINTS', '2'),
  '--direction-count', envValue('GCL_BLIND_SWIGLU_DIRECTION_COUNT', '384'),
  '--subspace-count', envValue('GCL_BLIND_SWIGLU_SUBSPACE_COUNT', '0'),
  '--cubic-direction-count', envValue('GCL_BLIND_SWIGLU_CUBIC_DIRECTION_COUNT', '0'),
  '--cubic-step', envValue('GCL_BLIND_SWIGLU_CUBIC_STEP', '0.35'),
  '--cubic-steps', envValue('GCL_BLIND_SWIGLU_CUBIC_STEPS', ''),
  '--cubic-holdout-count', envValue('GCL_BLIND_SWIGLU_CUBIC_HOLDOUT_COUNT', '0'),
  '--finite-difference-step', envValue('GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEP', '0.08'),
  '--finite-difference-steps', envValue('GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEPS', ''),
  '--input-scale', envValue('GCL_BLIND_SWIGLU_INPUT_SCALE', '0.45'),
  '--validation-scale', envValue('GCL_BLIND_SWIGLU_VALIDATION_SCALE', '0.65'),
  '--train-sampling-mode', envValue('GCL_BLIND_SWIGLU_TRAIN_SAMPLING_MODE', 'gaussian'),
  '--train-count', envValue('GCL_BLIND_SWIGLU_TRAIN_COUNT', '1600'),
  '--validation-count', envValue('GCL_BLIND_SWIGLU_VALIDATION_COUNT', '512'),
  '--fit-steps', envValue('GCL_BLIND_SWIGLU_FIT_STEPS', '1200'),
  '--fit-learning-rate', envValue('GCL_BLIND_SWIGLU_FIT_LEARNING_RATE', '0.025'),
  '--fit-restarts', envValue('GCL_BLIND_SWIGLU_FIT_RESTARTS', '0'),
  '--fit-restart-chunk-size', envValue('GCL_BLIND_SWIGLU_FIT_RESTART_CHUNK_SIZE', '0'),
  '--fit-mode', envValue('GCL_BLIND_SWIGLU_FIT_MODE', 'auto'),
  '--recovery-path', envValue('GCL_BLIND_SWIGLU_RECOVERY_PATH', 'nonlinear-fit'),
  '--fit-gradient-count', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_COUNT', '0'),
  '--fit-gradient-mode', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_MODE', 'forward'),
  '--fit-gradient-step', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_STEP', '0'),
  '--fit-gradient-steps', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_STEPS', ''),
  '--fit-gradient-weight', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_WEIGHT', '0.2'),
  '--fit-selection-count', envValue('GCL_BLIND_SWIGLU_FIT_SELECTION_COUNT', '0'),
  '--fit-selection-scale', envValue('GCL_BLIND_SWIGLU_FIT_SELECTION_SCALE', '0'),
  '--selection-tail-weight', envValue('GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT', '0'),
  '--selection-tail-quantile', envValue('GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE', '0.995'),
  '--selection-worst-weight', envValue('GCL_BLIND_SWIGLU_SELECTION_WORST_WEIGHT', '0'),
  '--seed', envValue('GCL_BLIND_SWIGLU_SEED', '260620'),
  ...optionalEnvArg('--polish-top-k', 'GCL_BLIND_SWIGLU_POLISH_TOP_K'),
  ...optionalEnvArg('--backfit-cycles', 'GCL_BLIND_SWIGLU_BACKFIT_CYCLES'),
  ...optionalEnvArg('--tail-polish-steps', 'GCL_BLIND_SWIGLU_TAIL_POLISH_STEPS'),
  ...optionalEnvArg('--tail-polish-weight', 'GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT'),
  ...optionalEnvArg('--tail-polish-fraction', 'GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION'),
  ...optionalEnvArg('--tail-polish-lr', 'GCL_BLIND_SWIGLU_TAIL_POLISH_LR'),
  ...optionalEnvArg('--tail-polish-loss', 'GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS'),
  ...optionalEnvArg('--tail-polish-smoothness', 'GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS'),
  ...optionalEnvArg('--tail-polish-calibration', 'GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION'),
  ...optionalEnvArg('--final-lbfgs-steps', 'GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS'),
  ...optionalEnvArg('--final-lbfgs-tail-weight', 'GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT'),
  ...optionalEnvArg('--final-lbfgs-tail-fraction', 'GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION'),
  ...(envValue('GCL_BLIND_SWIGLU_ROBUST_SCALAR_CALIBRATION', '0') === '1' ? ['--robust-scalar-calibration'] : []),
  ...optionalEnvArg('--scalar-calibration-steps', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_STEPS'),
  ...optionalEnvArg('--scalar-calibration-lr', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_LR'),
  ...optionalEnvArg('--scalar-calibration-tail-weight', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_WEIGHT'),
  ...optionalEnvArg('--scalar-calibration-tail-fraction', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_FRACTION'),
  '--min-r2', envValue('GCL_BLIND_SWIGLU_MIN_R2', '0.985'),
  '--max-mae', envValue('GCL_BLIND_SWIGLU_MAX_MAE', '0.015'),
  '--max-worst-error', envValue('GCL_BLIND_SWIGLU_MAX_WORST_ERROR', '1e30'),
  '--min-subspace-cosine', envValue('GCL_BLIND_SWIGLU_MIN_SUBSPACE_COSINE', '0.93'),
  '--max-cubic-witness-relative-rmse', envValue('GCL_BLIND_SWIGLU_MAX_CUBIC_WITNESS_RELATIVE_RMSE', 'Infinity'),
  '--max-oracle-calls', envValue('GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS', '20000'),
  ...(envValue('GCL_BLIND_SWIGLU_BASIS_DIAGNOSTIC_ONLY', '0') === '1' ? ['--basis-diagnostic-only'] : []),
  ...optionalEnvArg('--fit-snapshot-input', 'GCL_BLIND_SWIGLU_FIT_SNAPSHOT_INPUT'),
  ...optionalEnvArg('--fit-snapshot-output', 'GCL_BLIND_SWIGLU_FIT_SNAPSHOT_OUTPUT'),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_BLIND_SWIGLU_TIMEOUT_MS', '1800000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-blind-swiglu-oracle-decompiler',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Real local Mistral/LLaMA-family SwiGLU unit used only as an oracle provider.',
      'The solver process receives only f(x) batch outputs. Source weights are requested only after recovery for post-hoc witness alignment.',
      'This is a single-unit oracle-only K5 gate, not full FFN extraction or arbitrary LLM decompilation.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-blind-swiglu-oracle-decompiler',
    status: 'fail',
    summary: { missingReport: true },
    notes: ['Blind SwiGLU oracle decompiler did not produce report.json.']
  });
}

if (result.status !== 0) {
  console.error(JSON.stringify({ pass: false, summary }, null, 2));
  process.exit(result.status ?? 1);
}

// The wrapper's verdict used to be the child's exit code alone, so a child that exited 0
// while writing summary.pass false, or that exited 0 without writing report.json at all,
// produced a manifest reading "status": "fail" from a process that exited 0. Four of these
// wrappers are release-gated, where runStep only sees the exit status and would mark the
// step green against a failing manifest on disk. The Python tools currently raise
// SystemExit(1) themselves, so this is defence in depth rather than a live false pass, but
// the TypeScript side must not depend on that to stay honest.
if (!fs.existsSync(output.jsonOutputPath)) {
  console.error(JSON.stringify({ pass: false, reason: 'missing report.json', summary }, null, 2));
  process.exit(1);
}
if ((summary as { pass?: unknown } | undefined)?.pass !== true) {
  console.error(JSON.stringify({ pass: false, reason: 'report does not report pass', summary }, null, 2));
  process.exit(1);
}
