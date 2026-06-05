#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-gpt2-combined-circuit-finder',
  'gpt2-50-facts-attention-mlp-combined-graph'
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

const toolPath = path.join(repoRoot, 'tools', 'llm-reader-gpt2-combined-circuit-sweep.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing combined circuit sweep tool: ${toolPath}`);

const args = [
  toolPath,
  '--model', envValue('GCL_LIVE_GPT2_MODEL', 'gpt2'),
  '--case-count', envValue('GCL_LIVE_GPT2_COMBINED_FACT_COUNT', '50'),
  '--min-valid-cases', envValue('GCL_LIVE_GPT2_COMBINED_MIN_VALID_CASES', '50'),
  '--only-subjects', envValue('GCL_LIVE_GPT2_ONLY_SUBJECTS', ''),
  '--mlp-layers', envValue('GCL_LIVE_GPT2_LAYERS', '6,9,11'),
  '--mlp-fallback-layers', envValue('GCL_LIVE_GPT2_FALLBACK_LAYERS', '0,1,2,3,4,5,6,7,8,9,10,11'),
  '--attention-layers', envValue('GCL_LIVE_GPT2_ATTENTION_LAYERS', '6,9,11'),
  '--attention-fallback-layers', envValue('GCL_LIVE_GPT2_ATTENTION_FALLBACK_LAYERS', '0,1,2,3,4,5,6,7,8,9,10,11'),
  '--top-neurons', envValue('GCL_LIVE_GPT2_TOP_NEURONS', '50'),
  '--top-logits', envValue('GCL_LIVE_GPT2_TOP_LOGITS', '8'),
  '--control-count', envValue('GCL_LIVE_GPT2_COMBINED_CONTROL_COUNT', '8'),
  '--control-prompt', envValue('GCL_LIVE_GPT2_CONTROL_PROMPT', 'The weather in Berlin is'),
  '--control-seed', envValue('GCL_LIVE_GPT2_COMBINED_CONTROL_SEED', '260606'),
  '--max-graph-nodes', envValue('GCL_LIVE_GPT2_MAX_GRAPH_NODES', '5'),
  '--max-graph-heads', envValue('GCL_LIVE_GPT2_ATTENTION_MAX_GRAPH_HEADS', '3'),
  '--min-pass-rate', envValue('GCL_LIVE_GPT2_COMBINED_MIN_PASS_RATE', '0.90'),
  '--min-component-pass-rate', envValue('GCL_LIVE_GPT2_COMBINED_MIN_COMPONENT_PASS_RATE', '0.90'),
  '--min-target-drop', envValue('GCL_LIVE_GPT2_COMBINED_MIN_TARGET_DROP', '0.05'),
  '--min-margin-over-control-p95', envValue('GCL_LIVE_GPT2_COMBINED_MIN_MARGIN_OVER_CONTROL_P95', '0.02'),
  '--wrong-answer-count', envValue('GCL_LIVE_GPT2_WRONG_ANSWER_COUNT', '3'),
  '--paraphrase-count', envValue('GCL_LIVE_GPT2_PARAPHRASE_COUNT', '2'),
  '--negative-fact-count', envValue('GCL_LIVE_GPT2_NEGATIVE_FACT_COUNT', '2'),
  '--unrelated-logit-count', envValue('GCL_LIVE_GPT2_UNRELATED_LOGIT_COUNT', '8'),
  '--device', envValue('GCL_LIVE_GPT2_DEVICE', fs.existsSync(path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python')) ? 'cuda' : 'cpu'),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_LIVE_GPT2_COMBINED_TIMEOUT_MS', '2700000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-gpt2-combined-circuit-finder',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Real GPT-2 live factual-prompt combined attention+MLP circuit finder.',
      'This uses automatic MLP neuron scanning, attention-head scanning, one token/head/neuron/target graph, simultaneous attention+MLP causal ablation, combined controls, and explicit CUDA runtime support when available.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-gpt2-combined-circuit-finder',
    status: 'fail',
    summary: { missingReport: true },
    notes: ['Python combined circuit finder did not produce report.json.']
  });
}

if (result.status !== 0) {
  console.error(JSON.stringify({ pass: false, summary }, null, 2));
  process.exit(result.status ?? 1);
}
