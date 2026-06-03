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
  '--layers', envValue('GCL_LIVE_GPT2_LAYERS', '0,3,6,9,11'),
  '--top-neurons', envValue('GCL_LIVE_GPT2_TOP_NEURONS', '6'),
  '--top-logits', envValue('GCL_LIVE_GPT2_TOP_LOGITS', '8'),
  '--control-count', envValue('GCL_LIVE_GPT2_CONTROL_COUNT', '8'),
  '--control-prompt', envValue('GCL_LIVE_GPT2_CONTROL_PROMPT', 'The weather in Berlin is'),
  '--control-seed', envValue('GCL_LIVE_GPT2_CONTROL_SEED', '260603'),
  '--max-graph-nodes', envValue('GCL_LIVE_GPT2_MAX_GRAPH_NODES', '4'),
  '--min-pass-rate', envValue('GCL_LIVE_GPT2_MIN_PASS_RATE', '0.55'),
  '--min-target-drop', envValue('GCL_LIVE_GPT2_MIN_TARGET_DROP', '0.05'),
  '--min-margin-over-control-p95', envValue('GCL_LIVE_GPT2_MIN_MARGIN_OVER_CONTROL_P95', '0.02'),
  '--device', envValue('GCL_LIVE_GPT2_DEVICE', 'cpu'),
  '--local-files-only',
  '--output-json', output.jsonOutputPath,
  '--output-md', output.markdownOutputPath
];

const result = spawnSync(resolveGclPython(), args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(envValue('GCL_LIVE_GPT2_TIMEOUT_MS', '900000')),
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
      'This uses automatic layer/neuron scanning, token/neuron/target graph construction, joint graph ablation, and graph-level controls.'
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
