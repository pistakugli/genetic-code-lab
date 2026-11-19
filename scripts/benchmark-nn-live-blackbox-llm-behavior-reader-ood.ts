#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-blackbox-llm-behavior-reader',
  'gpt2-output-only-ood-20-facts-behavior-graph'
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

const toolPath = path.join(repoRoot, 'tools', 'llm-reader-blackbox-behavior.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing black-box behavior reader: ${toolPath}`);

// Roadmap output-only rail, next gates: larger fact set (the full 20-fact pool)
// and the OOD paraphrase families, through the unchanged prompt-to-logits-only
// solver boundary. The paraphrase protocol is the fixed six-family answer-blind
// OOD set from llm_reader/ood_paraphrases.py: every family prompt is scored and
// kept, there is no margin-based selection, and per-family retention is
// aggregated over all cases. Brazil/Brasilia is not a single GPT-2 token, so
// 19 of the 20 pool facts can be valid; the skipped fact is recorded in the
// artifact (skippedCases) instead of being silently dropped.
const args = [
  toolPath,
  '--model', envValue('GCL_BLACKBOX_LLM_MODEL', 'gpt2'),
  '--case-count', envValue('GCL_BLACKBOX_LLM_CASE_COUNT', '20'),
  '--min-valid-cases', envValue('GCL_BLACKBOX_LLM_MIN_VALID_CASES', '19'),
  '--only-subjects', envValue('GCL_BLACKBOX_LLM_ONLY_SUBJECTS', ''),
  '--control-count', envValue('GCL_BLACKBOX_LLM_CONTROL_COUNT', '8'),
  '--paraphrase-count', envValue('GCL_BLACKBOX_LLM_PARAPHRASE_COUNT', '6'),
  '--paraphrase-profile', envValue('GCL_BLACKBOX_LLM_PARAPHRASE_PROFILE', 'ood'),
  '--wrong-answer-count', envValue('GCL_BLACKBOX_LLM_WRONG_ANSWER_COUNT', '4'),
  '--max-target-rank', envValue('GCL_BLACKBOX_LLM_MAX_TARGET_RANK', '100'),
  '--min-pass-rate', envValue('GCL_BLACKBOX_LLM_MIN_PASS_RATE', '1.0'),
  '--min-subject-lift', envValue('GCL_BLACKBOX_LLM_MIN_SUBJECT_LIFT', '0.05'),
  '--min-relation-lift', envValue('GCL_BLACKBOX_LLM_MIN_RELATION_LIFT', '0.02'),
  '--min-paraphrase-retention-rate', envValue('GCL_BLACKBOX_LLM_MIN_PARAPHRASE_RETENTION_RATE', '0.50'),
  '--control-seed', envValue('GCL_BLACKBOX_LLM_CONTROL_SEED', '260709'),
  '--device', envValue(
    'GCL_BLACKBOX_LLM_DEVICE',
    fs.existsSync(path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python')) ? 'cuda' : 'cpu'
  ),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_BLACKBOX_LLM_TIMEOUT_MS', '1800000')),
  maxBuffer: 128 * 1024 * 1024
});

if (result.error) throw result.error;

let summary: unknown = undefined;
if (fs.existsSync(output.jsonOutputPath)) {
  const report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8'));
  summary = report.summary;
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-blackbox-llm-behavior-reader',
    status: report.summary?.pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: [
      'Output/logit-only whole-model factual behavior reader, OOD paraphrase profile.',
      'Solver receives prompt strings, target token ids, and next-token logits only.',
      'No weights, hooks, hidden activations, source ids, or ablation hooks are used by the solver.',
      'Paraphrase protocol: the fixed six answer-blind OOD families (service-question, possessive-question, imperative, embedded-question, sentence-completion, context-prefixed-question); every family prompt is scored and kept with no margin-based selection.',
      'Full 20-fact pool; one GPT-2 non-single-token answer (Brasilia) is skipped and recorded.'
    ]
  });
} else {
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-live-blackbox-llm-behavior-reader',
    status: 'fail',
    summary: { missingReport: true },
    notes: ['Python black-box behavior reader did not produce report.json.']
  });
}

if (result.status !== 0) {
  console.error(JSON.stringify({ pass: false, summary }, null, 2));
  process.exit(result.status ?? 1);
}

// The wrapper's verdict must never be the child's exit code alone (see the
// recorded wrapper-fix note): a child that exits 0 while writing summary.pass
// false, or that exits 0 without writing report.json at all, must fail here.
if (!fs.existsSync(output.jsonOutputPath)) {
  console.error(JSON.stringify({ pass: false, reason: 'missing report.json', summary }, null, 2));
  process.exit(1);
}
if ((summary as { pass?: unknown } | undefined)?.pass !== true) {
  console.error(JSON.stringify({ pass: false, reason: 'report does not report pass', summary }, null, 2));
  process.exit(1);
}
