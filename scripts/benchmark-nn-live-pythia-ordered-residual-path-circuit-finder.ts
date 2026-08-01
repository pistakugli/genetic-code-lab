#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-pythia-ordered-residual-path-circuit-finder',
  'pythia-70m-20-facts-ordered-residual-path'
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

const toolPath = path.join(repoRoot, 'tools', 'llm-reader-gpt2-ordered-residual-path-sweep.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing ordered residual path sweep tool: ${toolPath}`);

const args = [
  toolPath,
  '--adapter', 'gpt-neox',
  '--model', envValue('GCL_LIVE_PYTHIA_MODEL', 'EleutherAI/pythia-70m-deduped'),
  '--case-count', envValue('GCL_LIVE_PYTHIA_ORDERED_FACT_COUNT', '20'),
  '--min-valid-cases', envValue('GCL_LIVE_PYTHIA_ORDERED_MIN_VALID_CASES', '20'),
  '--only-subjects', envValue('GCL_LIVE_PYTHIA_ONLY_SUBJECTS', ''),
  '--mlp-layers', envValue('GCL_LIVE_PYTHIA_LAYERS', '0,1,2,3,4,5'),
  '--mlp-fallback-layers', envValue('GCL_LIVE_PYTHIA_FALLBACK_LAYERS', '0,1,2,3,4,5'),
  '--attention-layers', envValue('GCL_LIVE_PYTHIA_ATTENTION_LAYERS', '0,1,2,3,4,5'),
  '--attention-fallback-layers', envValue('GCL_LIVE_PYTHIA_ATTENTION_FALLBACK_LAYERS', '0,1,2,3,4,5'),
  '--top-neurons', envValue('GCL_LIVE_PYTHIA_TOP_NEURONS', '30'),
  '--top-logits', envValue('GCL_LIVE_PYTHIA_TOP_LOGITS', '8'),
  '--control-count', envValue('GCL_LIVE_PYTHIA_ORDERED_CONTROL_COUNT', '4'),
  '--control-prompt', envValue('GCL_LIVE_PYTHIA_CONTROL_PROMPT', 'The weather in Berlin is'),
  '--control-seed', envValue('GCL_LIVE_PYTHIA_ORDERED_CONTROL_SEED', '260607'),
  '--max-graph-nodes', envValue('GCL_LIVE_PYTHIA_MAX_GRAPH_NODES', '3'),
  '--max-graph-heads', envValue('GCL_LIVE_PYTHIA_ATTENTION_MAX_GRAPH_HEADS', '2'),
  '--min-pass-rate', envValue('GCL_LIVE_PYTHIA_ORDERED_MIN_PASS_RATE', '1.0'),
  '--min-component-pass-rate', envValue('GCL_LIVE_PYTHIA_ORDERED_MIN_COMPONENT_PASS_RATE', '1.0'),
  '--min-target-drop', envValue('GCL_LIVE_PYTHIA_COMBINED_MIN_TARGET_DROP', '0.05'),
  '--min-margin-over-control-p95', envValue('GCL_LIVE_PYTHIA_COMBINED_MIN_MARGIN_OVER_CONTROL_P95', '0.02'),
  '--min-path-activation-delta', envValue('GCL_LIVE_PYTHIA_ORDERED_MIN_PATH_ACTIVATION_DELTA', '0.005'),
  '--min-path-margin-over-control-p95', envValue('GCL_LIVE_PYTHIA_ORDERED_MIN_PATH_MARGIN_OVER_CONTROL_P95', '0.001'),
  '--ordered-rescue-candidate-count', envValue('GCL_LIVE_PYTHIA_ORDERED_RESCUE_CANDIDATE_COUNT', '24'),
  '--ordered-rescue-max-pair-attempts', envValue('GCL_LIVE_PYTHIA_ORDERED_RESCUE_MAX_PAIR_ATTEMPTS', '64'),
  '--ordered-rescue-set-candidate-count', envValue('GCL_LIVE_PYTHIA_ORDERED_RESCUE_SET_CANDIDATE_COUNT', '12'),
  '--ordered-rescue-max-set-attempts', envValue('GCL_LIVE_PYTHIA_ORDERED_RESCUE_MAX_SET_ATTEMPTS', '144'),
  '--wrong-answer-count', envValue('GCL_LIVE_PYTHIA_WRONG_ANSWER_COUNT', '3'),
  '--attention-max-target-rank', envValue('GCL_LIVE_PYTHIA_ATTENTION_MAX_TARGET_RANK', '200'),
  '--paraphrase-count', envValue('GCL_LIVE_PYTHIA_PARAPHRASE_COUNT', '2'),
  '--negative-fact-count', envValue('GCL_LIVE_PYTHIA_NEGATIVE_FACT_COUNT', '2'),
  '--unrelated-logit-count', envValue('GCL_LIVE_PYTHIA_UNRELATED_LOGIT_COUNT', '8'),
  '--no-require-component-hardening',
  '--allow-target-causal-mlp-fallback',
  '--device', envValue('GCL_LIVE_PYTHIA_DEVICE', fs.existsSync(path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python')) ? 'cuda' : 'cpu'),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_LIVE_PYTHIA_ORDERED_TIMEOUT_MS', '1800000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-pythia-ordered-residual-path-circuit-finder',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Second real-model adapter preview gate using EleutherAI/pythia-70m-deduped / GPT-NeoX.',
      'This reuses the ordered residual path finder through the LLMOracleAdapter contract and validates real attention->MLP activation dependency plus target-logit causal effectiveness. It is not synthetic and does not fit a surrogate.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-pythia-ordered-residual-path-circuit-finder',
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
if ((summary as { pass?: unknown } | undefined)?.pass !== true) {
  console.error(JSON.stringify({ pass: false, reason: 'report does not report pass', summary }, null, 2));
  process.exit(1);
}
