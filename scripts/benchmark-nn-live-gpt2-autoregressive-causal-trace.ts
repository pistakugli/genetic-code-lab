#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-gpt2-autoregressive-causal-trace',
  'all-50'
);
const protocolByVariant: Record<string, { protocol: string; caseSet: string }> = {
  'regression-five': { protocol: 'legacy', caseSet: 'regression-five' },
  'pilot-five': { protocol: 'autoregressive', caseSet: 'pilot-five' },
  'all-50': { protocol: 'autoregressive', caseSet: 'all-50' }
};
const selection = protocolByVariant[output.variant];
if (!selection) {
  throw new Error('GCL_BENCHMARK_VARIANT must be regression-five, pilot-five, or all-50');
}
const python = process.env.GCL_PYTHON
  || path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
const device = process.env.GCL_GPT2_AUTOREGRESSIVE_DEVICE || 'auto';
if (!['auto', 'cuda', 'cpu'].includes(device)) {
  throw new Error('GCL_GPT2_AUTOREGRESSIVE_DEVICE must be auto, cuda, or cpu');
}
const sourceReport = path.join(
  repoRoot,
  'benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder',
  'gpt2-50-facts-ordered-residual-path',
  'report.json'
);
const runner = path.join(repoRoot, 'tools', 'llm_reader', 'gpt2_internal_trace.py');
if (!fs.existsSync(python)) throw new Error(`Missing configured GCL Python interpreter: ${python}`);
if (!fs.existsSync(sourceReport)) throw new Error(`Missing frozen source artifact: ${sourceReport}`);
if (!fs.existsSync(runner)) throw new Error(`Missing GPT-2 trace runner: ${runner}`);
fs.mkdirSync(output.outputDir, { recursive: true });

const result = spawnSync(python, [
  runner,
  '--source-report', sourceReport,
  '--output-dir', output.outputDir,
  '--model', process.env.GCL_GPT2_INTERNAL_TRACE_MODEL || 'gpt2',
  '--device', device,
  '--protocol', selection.protocol,
  '--case-set', selection.caseSet,
  '--max-new-tokens', '12',
  '--control-count', '100',
  '--control-magnitude-tolerance', '0.2'
], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(process.env.GCL_GPT2_AUTOREGRESSIVE_TIMEOUT_MS || '3600000'),
  maxBuffer: 128 * 1024 * 1024
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);

const reportPath = path.join(output.outputDir, 'report.json');
if (!fs.existsSync(reportPath)) {
  throw new Error('Trace runner exited successfully but did not write report.json');
}
const report = JSON.parse(fs.readFileSync(reportPath, 'utf8')) as {
  kind?: string;
  model?: { id?: string; device?: string };
  summary?: { caseCount?: number; generatedPredictionCount?: number };
};
const expectedKind = selection.protocol === 'legacy'
  ? 'gpt2-instrumented-internal-computation-trace'
  : 'gpt2-autoregressive-causal-computation-trace';
if (report.kind !== expectedKind) throw new Error('Trace runner emitted an unexpected report kind');

writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-live-gpt2-autoregressive-causal-trace',
  status: 'generated',
  summary: {
    protocol: selection.protocol,
    caseSet: selection.caseSet,
    model: report.model,
    caseCount: report.summary?.caseCount,
    generatedPredictionCount: report.summary?.generatedPredictionCount,
    reportJson: path.relative(repoRoot, reportPath)
  },
  notes: [
    'Instrumented local GPT-2 witness; not a black-box solver result.',
    'Source prompts and ordered-path nodes are read unchanged from the existing 50-fact report.',
    'Free-running interventions are measured on baseline contexts and do not feed changed tokens into later steps.',
    'Control groups match path component count, type, layer, and token positions; activation magnitude match coverage is recorded.'
  ]
});
console.log('GPT2_AUTOREGRESSIVE_CAUSAL_TRACE_EXPERIMENT_WRITTEN');
