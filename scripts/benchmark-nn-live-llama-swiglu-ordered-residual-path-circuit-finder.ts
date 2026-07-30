#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-llama-swiglu-ordered-residual-path-circuit-finder',
  'llama-3.2-3b-20-facts-ordered-residual-path'
);

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const gpuPython = path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
  if (fs.existsSync(gpuPython)) return gpuPython;
  return 'python3';
}

function envValue(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

fs.mkdirSync(output.outputDir, { recursive: true });

const toolPath = path.join(repoRoot, 'tools', 'llm-reader-gpt2-ordered-residual-path-sweep.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing ordered residual path sweep tool: ${toolPath}`);

const args = [
  toolPath,
  '--adapter', 'llama-swiglu',
  '--model', envValue(
    'GCL_LIVE_LLAMA_MODEL',
    'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit'
  ),
  '--case-count', envValue('GCL_LIVE_LLAMA_ORDERED_FACT_COUNT', '20'),
  '--min-valid-cases', envValue('GCL_LIVE_LLAMA_ORDERED_MIN_VALID_CASES', '20'),
  '--only-subjects', envValue('GCL_LIVE_LLAMA_ONLY_SUBJECTS', ''),
  '--mlp-layers', envValue('GCL_LIVE_LLAMA_LAYERS', '20,24,27'),
  '--mlp-fallback-layers', envValue('GCL_LIVE_LLAMA_FALLBACK_LAYERS', '16,18,20,22,24,26,27'),
  '--attention-layers', envValue('GCL_LIVE_LLAMA_ATTENTION_LAYERS', '20,24,27'),
  '--attention-fallback-layers', envValue(
    'GCL_LIVE_LLAMA_ATTENTION_FALLBACK_LAYERS',
    '16,18,20,22,24,26,27'
  ),
  '--top-neurons', envValue('GCL_LIVE_LLAMA_TOP_NEURONS', '8'),
  '--top-logits', envValue('GCL_LIVE_LLAMA_TOP_LOGITS', '8'),
  '--control-count', envValue('GCL_LIVE_LLAMA_ORDERED_CONTROL_COUNT', '3'),
  '--control-prompt', envValue('GCL_LIVE_LLAMA_CONTROL_PROMPT', 'The weather in Berlin is'),
  '--control-seed', envValue('GCL_LIVE_LLAMA_ORDERED_CONTROL_SEED', '260610'),
  '--max-graph-nodes', envValue('GCL_LIVE_LLAMA_MAX_GRAPH_NODES', '3'),
  '--max-graph-heads', envValue('GCL_LIVE_LLAMA_ATTENTION_MAX_GRAPH_HEADS', '2'),
  '--min-pass-rate', envValue('GCL_LIVE_LLAMA_ORDERED_MIN_PASS_RATE', '0.90'),
  '--min-component-pass-rate', envValue('GCL_LIVE_LLAMA_ORDERED_MIN_COMPONENT_PASS_RATE', '0.90'),
  '--min-target-drop', envValue('GCL_LIVE_LLAMA_COMBINED_MIN_TARGET_DROP', '0.01'),
  '--min-margin-over-control-p95', envValue(
    'GCL_LIVE_LLAMA_COMBINED_MIN_MARGIN_OVER_CONTROL_P95',
    '0.001'
  ),
  '--min-path-activation-delta', envValue(
    'GCL_LIVE_LLAMA_ORDERED_MIN_PATH_ACTIVATION_DELTA',
    '0.001'
  ),
  '--min-path-margin-over-control-p95', envValue(
    'GCL_LIVE_LLAMA_ORDERED_MIN_PATH_MARGIN_OVER_CONTROL_P95',
    '0.0001'
  ),
  '--wrong-answer-count', envValue('GCL_LIVE_LLAMA_WRONG_ANSWER_COUNT', '2'),
  '--attention-max-target-rank', envValue('GCL_LIVE_LLAMA_ATTENTION_MAX_TARGET_RANK', '200'),
  '--paraphrase-count', envValue('GCL_LIVE_LLAMA_PARAPHRASE_COUNT', '1'),
  '--negative-fact-count', envValue('GCL_LIVE_LLAMA_NEGATIVE_FACT_COUNT', '1'),
  '--unrelated-logit-count', envValue('GCL_LIVE_LLAMA_UNRELATED_LOGIT_COUNT', '4'),
  '--no-require-component-hardening',
  '--allow-target-causal-mlp-fallback',
  '--device', 'cuda',
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_LIVE_LLAMA_ORDERED_TIMEOUT_MS', '1800000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-llama-swiglu-ordered-residual-path-circuit-finder',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Real Llama-3.2-3B-Instruct 4-bit CUDA gate through the model-agnostic LLMOracleAdapter.',
      'The addressable MLP unit is the real SiLU(gate_proj(x))*up_proj(x) pair before down_proj. The benchmark uses causal pair ablations, attention output-path interventions, ordered path controls, and no synthetic teacher or fitted surrogate.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-llama-swiglu-ordered-residual-path-circuit-finder',
    status: 'fail',
    summary: { missingReport: true },
    notes: ['Python ordered residual path circuit finder did not produce report.json.']
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
