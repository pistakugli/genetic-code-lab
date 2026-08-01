#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-blackbox-llm-behavior-reader',
  'gpt2-output-only-10-facts-behavior-graph'
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

const args = [
  toolPath,
  '--model', envValue('GCL_BLACKBOX_LLM_MODEL', 'gpt2'),
  '--case-count', envValue('GCL_BLACKBOX_LLM_CASE_COUNT', '10'),
  '--min-valid-cases', envValue('GCL_BLACKBOX_LLM_MIN_VALID_CASES', '10'),
  '--only-subjects', envValue('GCL_BLACKBOX_LLM_ONLY_SUBJECTS', ''),
  '--control-count', envValue('GCL_BLACKBOX_LLM_CONTROL_COUNT', '8'),
  '--paraphrase-count', envValue('GCL_BLACKBOX_LLM_PARAPHRASE_COUNT', '3'),
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
  timeout: Number(envValue('GCL_BLACKBOX_LLM_TIMEOUT_MS', '900000')),
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
      'Output/logit-only whole-model factual behavior reader.',
      'Solver receives prompt strings, target token ids, and next-token logits only.',
      'No weights, hooks, hidden activations, source ids, or ablation hooks are used by the solver.'
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
