#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-gpt2-fact-circuit-finder',
  'gpt2-50-facts-graph-layer-scan'
);

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const projectPython = path.join(repoRoot, '.gcl-python', 'bin', 'python');
  if (fs.existsSync(projectPython)) return projectPython;
  return 'python3';
}

function envValue(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

fs.mkdirSync(output.outputDir, { recursive: true });

const toolPath = path.join(repoRoot, 'tools', 'llm-reader-gpt2-factual-sweep.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing factual sweep tool: ${toolPath}`);

const args = [
  toolPath,
  '--model', envValue('GCL_LIVE_GPT2_MODEL', 'gpt2'),
  '--case-count', envValue('GCL_LIVE_GPT2_FACT_COUNT', '50'),
  '--min-valid-cases', envValue('GCL_LIVE_GPT2_MIN_VALID_CASES', '50'),
  '--only-subjects', envValue('GCL_LIVE_GPT2_ONLY_SUBJECTS', ''),
  '--layers', envValue('GCL_LIVE_GPT2_LAYERS', '6,9,11'),
  '--fallback-layers', envValue('GCL_LIVE_GPT2_FALLBACK_LAYERS', '0,1,2,3,4,5,6,7,8,9,10,11'),
  '--top-neurons', envValue('GCL_LIVE_GPT2_TOP_NEURONS', '50'),
  '--top-logits', envValue('GCL_LIVE_GPT2_TOP_LOGITS', '8'),
  '--control-count', envValue('GCL_LIVE_GPT2_CONTROL_COUNT', '8'),
  '--control-prompt', envValue('GCL_LIVE_GPT2_CONTROL_PROMPT', 'The weather in Berlin is'),
  '--control-seed', envValue('GCL_LIVE_GPT2_CONTROL_SEED', '260603'),
  '--max-graph-nodes', envValue('GCL_LIVE_GPT2_MAX_GRAPH_NODES', '5'),
  '--min-pass-rate', envValue('GCL_LIVE_GPT2_MIN_PASS_RATE', '0.55'),
  '--min-target-drop', envValue('GCL_LIVE_GPT2_MIN_TARGET_DROP', '0.05'),
  '--min-margin-over-control-p95', envValue('GCL_LIVE_GPT2_MIN_MARGIN_OVER_CONTROL_P95', '0.02'),
  '--paraphrase-count', envValue('GCL_LIVE_GPT2_PARAPHRASE_COUNT', '2'),
  '--negative-fact-count', envValue('GCL_LIVE_GPT2_NEGATIVE_FACT_COUNT', '2'),
  '--wrong-answer-count', envValue('GCL_LIVE_GPT2_WRONG_ANSWER_COUNT', '3'),
  '--unrelated-logit-count', envValue('GCL_LIVE_GPT2_UNRELATED_LOGIT_COUNT', '8'),
  '--min-paraphrase-pass-rate', envValue('GCL_LIVE_GPT2_MIN_PARAPHRASE_PASS_RATE', '0.70'),
  '--min-wrong-answer-specificity-rate', envValue('GCL_LIVE_GPT2_MIN_WRONG_ANSWER_SPECIFICITY_RATE', '0.85'),
  '--min-negative-fact-specificity-rate', envValue('GCL_LIVE_GPT2_MIN_NEGATIVE_FACT_SPECIFICITY_RATE', '0.85'),
  '--min-unrelated-preservation-rate', envValue('GCL_LIVE_GPT2_MIN_UNRELATED_PRESERVATION_RATE', '0.85'),
  '--max-wrong-answer-drop-ratio', envValue('GCL_LIVE_GPT2_MAX_WRONG_ANSWER_DROP_RATIO', '0.55'),
  '--max-candidate-wrong-answer-drop-ratio', envValue('GCL_LIVE_GPT2_MAX_CANDIDATE_WRONG_ANSWER_DROP_RATIO', '0.70'),
  '--max-negative-fact-drop-ratio', envValue('GCL_LIVE_GPT2_MAX_NEGATIVE_FACT_DROP_RATIO', '0.55'),
  '--max-unrelated-abs-drop-ratio', envValue('GCL_LIVE_GPT2_MAX_UNRELATED_ABS_DROP_RATIO', '0.65'),
  '--strict-rescue-candidate-count', envValue('GCL_LIVE_GPT2_STRICT_RESCUE_CANDIDATE_COUNT', '14'),
  '--strict-rescue-pair-candidate-count', envValue('GCL_LIVE_GPT2_STRICT_RESCUE_PAIR_CANDIDATE_COUNT', '12'),
  '--strict-rescue-max-combo-size', envValue('GCL_LIVE_GPT2_STRICT_RESCUE_MAX_COMBO_SIZE', '2'),
  '--strict-rescue-max-attempts', envValue('GCL_LIVE_GPT2_STRICT_RESCUE_MAX_ATTEMPTS', '72'),
  '--device', envValue('GCL_LIVE_GPT2_DEVICE', 'cpu'),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_LIVE_GPT2_TIMEOUT_MS', '1800000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-gpt2-fact-circuit-finder',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Real GPT-2 live factual-prompt graph-level circuit finder.',
      'This uses automatic layer/neuron scanning, token/neuron/target graph construction, joint graph ablation, graph-level controls, paraphrase validation, wrong-answer targets, negative facts, and unrelated-logit preservation checks.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-gpt2-fact-circuit-finder',
    status: 'fail',
    summary: { missingReport: true },
    notes: ['Python fact circuit finder did not produce report.json.']
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
