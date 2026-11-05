#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-gpt2-internal-trace',
  'five-facts'
);
const python = process.env.GCL_PYTHON
  || path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
const device = process.env.GCL_GPT2_INTERNAL_TRACE_DEVICE || 'auto';
if (!['auto', 'cuda', 'cpu'].includes(device)) {
  throw new Error('GCL_GPT2_INTERNAL_TRACE_DEVICE must be auto, cuda, or cpu');
}
const sourceReport = path.join(
  repoRoot,
  'benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder',
  'gpt2-50-facts-ordered-residual-path',
  'report.json'
);
const runner = path.join(repoRoot, 'tools', 'llm_reader', 'gpt2_internal_trace.py');
if (!fs.existsSync(python)) throw new Error(`Missing configured GCL Python interpreter: ${python}`);
if (!fs.existsSync(sourceReport)) throw new Error(`Missing immutable source artifact: ${sourceReport}`);
if (!fs.existsSync(runner)) throw new Error(`Missing GPT-2 trace runner: ${runner}`);
fs.mkdirSync(output.outputDir, { recursive: true });

const result = spawnSync(python, [
  runner,
  '--source-report', sourceReport,
  '--output-dir', output.outputDir,
  '--model', process.env.GCL_GPT2_INTERNAL_TRACE_MODEL || 'gpt2',
  '--device', device
], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(process.env.GCL_GPT2_INTERNAL_TRACE_TIMEOUT_MS || '1200000'),
  maxBuffer: 128 * 1024 * 1024
});
if (result.error) throw result.error;
if (result.status !== 0) {
  process.exit(result.status ?? 1);
}

const reportPath = path.join(output.outputDir, 'report.json');
if (!fs.existsSync(reportPath)) throw new Error('Trace runner exited successfully but did not write report.json');
const report = JSON.parse(fs.readFileSync(reportPath, 'utf8')) as {
  kind?: string;
  model?: { id?: string; device?: string };
  summary?: { caseCount?: number };
};
if (report.kind !== 'gpt2-instrumented-internal-computation-trace') {
  throw new Error('Trace runner emitted an unexpected report kind');
}
const expectedMarker = 'GPT2_INTERNAL_TRACE_EXPERIMENT_WRITTEN';
console.log(expectedMarker);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-live-gpt2-internal-trace',
  status: 'generated',
  summary: {
    model: report.model,
    caseCount: report.summary?.caseCount,
    reportJson: path.relative(repoRoot, reportPath)
  },
  notes: [
    'Instrumented local GPT-2 witness; not a black-box solver result.',
    'Raw full-sequence tensor captures and causal intervention measurements are stored beside report.json.',
    'Run tools/llm_reader/verify_gpt2_internal_trace.py independently to verify the report.'
  ]
});
