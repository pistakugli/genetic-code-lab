#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-gpt2-attention-path-circuit-finder',
  'gpt2-50-facts-attention-path-layer-scan'
);

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const gpuPython = path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
  if (fs.existsSync(gpuPython)) return gpuPython;
  const projectPython = path.join(repoRoot, '.gcl-python', 'bin', 'python');
  if (fs.existsSync(projectPython)) return projectPython;
  return 'python3';
}

function envValue(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

fs.mkdirSync(output.outputDir, { recursive: true });

const toolPath = path.join(repoRoot, 'tools', 'llm-reader-gpt2-attention-path-sweep.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing attention path sweep tool: ${toolPath}`);

const args = [
  toolPath,
  '--model', envValue('GCL_LIVE_GPT2_MODEL', 'gpt2'),
  '--case-count', envValue('GCL_LIVE_GPT2_ATTENTION_FACT_COUNT', '50'),
  '--min-valid-cases', envValue('GCL_LIVE_GPT2_ATTENTION_MIN_VALID_CASES', '50'),
  '--only-subjects', envValue('GCL_LIVE_GPT2_ONLY_SUBJECTS', ''),
  '--layers', envValue('GCL_LIVE_GPT2_ATTENTION_LAYERS', '6,9,11'),
  '--fallback-layers', envValue('GCL_LIVE_GPT2_ATTENTION_FALLBACK_LAYERS', '0,1,2,3,4,5,6,7,8,9,10,11'),
  '--control-count', envValue('GCL_LIVE_GPT2_ATTENTION_CONTROL_COUNT', '8'),
  '--control-prompt', envValue('GCL_LIVE_GPT2_CONTROL_PROMPT', 'The weather in Berlin is'),
  '--control-seed', envValue('GCL_LIVE_GPT2_ATTENTION_CONTROL_SEED', '260605'),
  '--max-graph-heads', envValue('GCL_LIVE_GPT2_ATTENTION_MAX_GRAPH_HEADS', '3'),
  '--min-pass-rate', envValue('GCL_LIVE_GPT2_ATTENTION_MIN_PASS_RATE', '0.90'),
  '--min-graph-drop', envValue('GCL_LIVE_GPT2_ATTENTION_MIN_GRAPH_DROP', '0.05'),
  '--min-margin-over-control-p95', envValue('GCL_LIVE_GPT2_ATTENTION_MIN_MARGIN_OVER_CONTROL_P95', '0.02'),
  '--wrong-answer-count', envValue('GCL_LIVE_GPT2_ATTENTION_WRONG_ANSWER_COUNT', '3'),
  '--max-wrong-answer-drop-ratio', envValue('GCL_LIVE_GPT2_ATTENTION_MAX_WRONG_ANSWER_DROP_RATIO', '0.80'),
  '--max-target-rank', envValue('GCL_LIVE_GPT2_ATTENTION_MAX_TARGET_RANK', '100'),
  '--device', envValue('GCL_LIVE_GPT2_DEVICE', fs.existsSync(path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python')) ? 'cuda' : 'cpu'),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_LIVE_GPT2_ATTENTION_TIMEOUT_MS', '1800000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-gpt2-attention-path-circuit-finder',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Real GPT-2 live factual-prompt attention-head/path circuit finder.',
      'This uses automatic layer/head scanning, token/head/target path graph construction, target-rank validity filtering, joint attention-head ablation, attention-head controls, wrong-answer collateral reporting, and explicit CUDA runtime support when available.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-gpt2-attention-path-circuit-finder',
    status: 'fail',
    summary: { missingReport: true },
    notes: ['Python attention path circuit finder did not produce report.json.']
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
if (summary?.pass !== true) {
  console.error(JSON.stringify({ pass: false, reason: 'report does not report pass', summary }, null, 2));
  process.exit(1);
}
