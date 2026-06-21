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
  '--device', envValue('GCL_BLIND_SWIGLU_DEVICE', 'cuda'),
  '--active-rank', envValue('GCL_BLIND_SWIGLU_ACTIVE_RANK', '2'),
  ...(process.env.GCL_BLIND_SWIGLU_ATOM_COUNT ? ['--atom-count', envValue('GCL_BLIND_SWIGLU_ATOM_COUNT', '')] : []),
  '--gradient-mode', envValue('GCL_BLIND_SWIGLU_GRADIENT_MODE', 'coordinate'),
  '--probe-points', envValue('GCL_BLIND_SWIGLU_PROBE_POINTS', '2'),
  '--direction-count', envValue('GCL_BLIND_SWIGLU_DIRECTION_COUNT', '384'),
  '--subspace-count', envValue('GCL_BLIND_SWIGLU_SUBSPACE_COUNT', '0'),
  '--finite-difference-step', envValue('GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEP', '0.08'),
  '--train-count', envValue('GCL_BLIND_SWIGLU_TRAIN_COUNT', '1600'),
  '--validation-count', envValue('GCL_BLIND_SWIGLU_VALIDATION_COUNT', '512'),
  '--fit-steps', envValue('GCL_BLIND_SWIGLU_FIT_STEPS', '1200'),
  '--fit-learning-rate', envValue('GCL_BLIND_SWIGLU_FIT_LEARNING_RATE', '0.025'),
  '--fit-restarts', envValue('GCL_BLIND_SWIGLU_FIT_RESTARTS', '0'),
  '--min-r2', envValue('GCL_BLIND_SWIGLU_MIN_R2', '0.985'),
  '--max-mae', envValue('GCL_BLIND_SWIGLU_MAX_MAE', '0.015'),
  '--max-worst-error', envValue('GCL_BLIND_SWIGLU_MAX_WORST_ERROR', '1e30'),
  '--min-subspace-cosine', envValue('GCL_BLIND_SWIGLU_MIN_SUBSPACE_COSINE', '0.93'),
  '--max-oracle-calls', envValue('GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS', '20000'),
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
