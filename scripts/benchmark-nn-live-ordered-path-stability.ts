#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type StabilityCase = {
  id: string;
  label: string;
  adapter: string;
  model: string;
  inputReport: string;
  subjects: string;
  caseCount: string;
  paraphraseCount: string;
  minOriginalDrop: string;
  minParaphraseDropRatio: string;
  minParaphrasePassRate: string;
  layerShifts: string;
  minLayerControlCount: string;
  minLayerMargin: string;
  minLayerSpecificityRate: string;
  timeoutMs: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-ordered-path-stability',
  'real-model-paraphrase-layer-replay'
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

function runCase(testCase: StabilityCase): { status: number | null; reportPath: string; summary: unknown } {
  const caseDir = path.join(output.outputDir, testCase.id);
  fs.mkdirSync(caseDir, { recursive: true });
  const reportPath = path.join(caseDir, 'report.json');
  const markdownPath = path.join(caseDir, 'report.md');
  const toolPath = path.join(repoRoot, 'tools', 'llm-reader-ordered-path-stability.py');
  if (!fs.existsSync(toolPath)) throw new Error(`Missing stability tool: ${toolPath}`);
  if (!fs.existsSync(path.join(repoRoot, testCase.inputReport))) {
    throw new Error(`Missing source report for ${testCase.id}: ${testCase.inputReport}`);
  }
  const args = [
    toolPath,
    '--adapter', testCase.adapter,
    '--model', testCase.model,
    '--input-report', path.join(repoRoot, testCase.inputReport),
    '--subjects', testCase.subjects,
    '--case-count', testCase.caseCount,
    '--paraphrase-count', testCase.paraphraseCount,
    '--min-original-drop', testCase.minOriginalDrop,
    '--min-paraphrase-drop-ratio', testCase.minParaphraseDropRatio,
    '--min-paraphrase-pass-rate', testCase.minParaphrasePassRate,
    `--layer-shifts=${testCase.layerShifts}`,
    '--min-layer-control-count', testCase.minLayerControlCount,
    '--min-layer-margin', testCase.minLayerMargin,
    '--min-layer-specificity-rate', testCase.minLayerSpecificityRate,
    '--device', 'cuda',
    '--local-files-only',
    '--output-json', reportPath,
    '--output-md', markdownPath
  ];
  const result = spawnSync(resolveGclPython(), args, {
    cwd: repoRoot,
    stdio: 'inherit',
    env: process.env,
    timeout: testCase.timeoutMs,
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;
  let summary: unknown = { missingReport: true };
  if (fs.existsSync(reportPath)) {
    const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
    summary = report.summary;
  }
  return { status: result.status, reportPath, summary };
}

fs.mkdirSync(output.outputDir, { recursive: true });

const cases: StabilityCase[] = [
  {
    id: 'gpt2',
    label: 'GPT-2',
    adapter: 'gpt2',
    model: envValue('GCL_LIVE_STABILITY_GPT2_MODEL', 'gpt2'),
    inputReport: 'benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json',
    subjects: envValue('GCL_LIVE_STABILITY_GPT2_SUBJECTS', 'France,Germany,Italy,Portugal,Greece'),
    caseCount: envValue('GCL_LIVE_STABILITY_GPT2_CASE_COUNT', '5'),
    paraphraseCount: envValue('GCL_LIVE_STABILITY_GPT2_PARAPHRASE_COUNT', '3'),
    minOriginalDrop: envValue('GCL_LIVE_STABILITY_GPT2_MIN_ORIGINAL_DROP', '0.05'),
    minParaphraseDropRatio: envValue('GCL_LIVE_STABILITY_GPT2_MIN_PARAPHRASE_DROP_RATIO', '0.10'),
    minParaphrasePassRate: envValue('GCL_LIVE_STABILITY_GPT2_MIN_PARAPHRASE_PASS_RATE', '0.66'),
    layerShifts: envValue('GCL_LIVE_STABILITY_GPT2_LAYER_SHIFTS', '-2,-1,1,2'),
    minLayerControlCount: envValue('GCL_LIVE_STABILITY_GPT2_MIN_LAYER_CONTROL_COUNT', '2'),
    minLayerMargin: envValue('GCL_LIVE_STABILITY_GPT2_MIN_LAYER_MARGIN', '0.001'),
    minLayerSpecificityRate: envValue('GCL_LIVE_STABILITY_GPT2_MIN_LAYER_SPECIFICITY_RATE', '0.8'),
    timeoutMs: Number(envValue('GCL_LIVE_STABILITY_GPT2_TIMEOUT_MS', '300000'))
  },
  {
    id: 'pythia',
    label: 'Pythia/GPT-NeoX',
    adapter: 'gpt-neox',
    model: envValue('GCL_LIVE_STABILITY_PYTHIA_MODEL', 'EleutherAI/pythia-70m-deduped'),
    inputReport: 'benchmarks/release/nn-live-pythia-ordered-residual-path-circuit-finder/pythia-70m-20-facts-ordered-residual-path/report.json',
    subjects: envValue('GCL_LIVE_STABILITY_PYTHIA_SUBJECTS', 'Spain,Japan,Russia,Norway,Austria'),
    caseCount: envValue('GCL_LIVE_STABILITY_PYTHIA_CASE_COUNT', '5'),
    paraphraseCount: envValue('GCL_LIVE_STABILITY_PYTHIA_PARAPHRASE_COUNT', '3'),
    minOriginalDrop: envValue('GCL_LIVE_STABILITY_PYTHIA_MIN_ORIGINAL_DROP', '0.05'),
    minParaphraseDropRatio: envValue('GCL_LIVE_STABILITY_PYTHIA_MIN_PARAPHRASE_DROP_RATIO', '0.10'),
    minParaphrasePassRate: envValue('GCL_LIVE_STABILITY_PYTHIA_MIN_PARAPHRASE_PASS_RATE', '0.66'),
    layerShifts: envValue('GCL_LIVE_STABILITY_PYTHIA_LAYER_SHIFTS', '-2,-1,1,2'),
    minLayerControlCount: envValue('GCL_LIVE_STABILITY_PYTHIA_MIN_LAYER_CONTROL_COUNT', '2'),
    minLayerMargin: envValue('GCL_LIVE_STABILITY_PYTHIA_MIN_LAYER_MARGIN', '0.001'),
    minLayerSpecificityRate: envValue('GCL_LIVE_STABILITY_PYTHIA_MIN_LAYER_SPECIFICITY_RATE', '0.8'),
    timeoutMs: Number(envValue('GCL_LIVE_STABILITY_PYTHIA_TIMEOUT_MS', '420000'))
  },
  {
    id: 'llama',
    label: 'Llama SwiGLU',
    adapter: 'llama-swiglu',
    model: envValue('GCL_LIVE_STABILITY_LLAMA_MODEL', 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit'),
    inputReport: 'benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-20-facts-ordered-residual-path/report.json',
    subjects: envValue('GCL_LIVE_STABILITY_LLAMA_SUBJECTS', 'France,Germany,Italy,Spain'),
    caseCount: envValue('GCL_LIVE_STABILITY_LLAMA_CASE_COUNT', '4'),
    paraphraseCount: envValue('GCL_LIVE_STABILITY_LLAMA_PARAPHRASE_COUNT', '3'),
    minOriginalDrop: envValue('GCL_LIVE_STABILITY_LLAMA_MIN_ORIGINAL_DROP', '0.005'),
    minParaphraseDropRatio: envValue('GCL_LIVE_STABILITY_LLAMA_MIN_PARAPHRASE_DROP_RATIO', '0.05'),
    minParaphrasePassRate: envValue('GCL_LIVE_STABILITY_LLAMA_MIN_PARAPHRASE_PASS_RATE', '0.66'),
    layerShifts: envValue('GCL_LIVE_STABILITY_LLAMA_LAYER_SHIFTS', '-2,-1,1,2'),
    minLayerControlCount: envValue('GCL_LIVE_STABILITY_LLAMA_MIN_LAYER_CONTROL_COUNT', '2'),
    minLayerMargin: envValue('GCL_LIVE_STABILITY_LLAMA_MIN_LAYER_MARGIN', '0.0001'),
    minLayerSpecificityRate: envValue('GCL_LIVE_STABILITY_LLAMA_MIN_LAYER_SPECIFICITY_RATE', '0.75'),
    timeoutMs: Number(envValue('GCL_LIVE_STABILITY_LLAMA_TIMEOUT_MS', '900000'))
  },
  {
    id: 'mistral',
    label: 'Mistral SwiGLU',
    adapter: 'mistral-swiglu',
    model: envValue('GCL_LIVE_STABILITY_MISTRAL_MODEL', 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'),
    inputReport: 'benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-10-facts-ordered-residual-path/report.json',
    subjects: envValue('GCL_LIVE_STABILITY_MISTRAL_SUBJECTS', 'France,Germany,Spain,Japan'),
    caseCount: envValue('GCL_LIVE_STABILITY_MISTRAL_CASE_COUNT', '4'),
    paraphraseCount: envValue('GCL_LIVE_STABILITY_MISTRAL_PARAPHRASE_COUNT', '3'),
    minOriginalDrop: envValue('GCL_LIVE_STABILITY_MISTRAL_MIN_ORIGINAL_DROP', '0.005'),
    minParaphraseDropRatio: envValue('GCL_LIVE_STABILITY_MISTRAL_MIN_PARAPHRASE_DROP_RATIO', '0.05'),
    minParaphrasePassRate: envValue('GCL_LIVE_STABILITY_MISTRAL_MIN_PARAPHRASE_PASS_RATE', '0.66'),
    layerShifts: envValue('GCL_LIVE_STABILITY_MISTRAL_LAYER_SHIFTS', '-2,-1,1,2'),
    minLayerControlCount: envValue('GCL_LIVE_STABILITY_MISTRAL_MIN_LAYER_CONTROL_COUNT', '2'),
    minLayerMargin: envValue('GCL_LIVE_STABILITY_MISTRAL_MIN_LAYER_MARGIN', '0.0001'),
    minLayerSpecificityRate: envValue('GCL_LIVE_STABILITY_MISTRAL_MIN_LAYER_SPECIFICITY_RATE', '0.75'),
    timeoutMs: Number(envValue('GCL_LIVE_STABILITY_MISTRAL_TIMEOUT_MS', '1200000'))
  }
];

const results = cases.map((testCase) => {
  console.log(JSON.stringify({ stabilityCase: testCase.id, model: testCase.model }));
  const result = runCase(testCase);
  return {
    id: testCase.id,
    label: testCase.label,
    status: result.status === 0 ? 'pass' : 'fail',
    reportPath: path.relative(repoRoot, result.reportPath),
    summary: result.summary
  };
});

const pass = results.every((result) => result.status === 'pass');
const suiteReport = {
  schemaVersion: 1,
  kind: 'nn-live-ordered-path-stability-suite',
  synthetic: false,
  surrogate: false,
  generatedAt: new Date().toISOString(),
  summary: {
    pass,
    cases: results.length,
    passed: results.filter((result) => result.status === 'pass').length,
    failed: results.filter((result) => result.status !== 'pass').map((result) => result.id)
  },
  results
};

fs.writeFileSync(output.jsonOutputPath, JSON.stringify(suiteReport, null, 2) + '\n');
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# Live Ordered Path Stability Suite',
    '',
    'This suite replays discovered real-model circuits across paraphrase and neighboring-layer controls. It does not fit a surrogate.',
    '',
    `- Pass: \`${pass}\``,
    `- Passed: \`${suiteReport.summary.passed}/${suiteReport.summary.cases}\``,
    '',
    '| model | status | cases | paraphrase pass | layer specificity | failed subjects | report |',
    '|---|---:|---:|---:|---:|---|---|',
    ...results.map((result) => {
      const summary = result.summary as {
        caseCount?: number;
        passedCases?: number;
        paraphrasePassRate?: number;
        layerSpecificityRate?: number;
        failedSubjects?: string[];
      };
      return `| ${result.label} | ${result.status} | ${summary.passedCases ?? '?'}/${summary.caseCount ?? '?'} | ${summary.paraphrasePassRate ?? '?'} | ${summary.layerSpecificityRate ?? '?'} | ${(summary.failedSubjects ?? []).join(',') || '[]'} | ${result.reportPath} |`;
    })
  ].join('\n') + '\n'
);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-live-ordered-path-stability',
  status: pass ? 'pass' : 'fail',
  summary: suiteReport.summary,
  notes: [
    'Real-model circuit replay across GPT-2, Pythia/GPT-NeoX, Llama SwiGLU, and Mistral SwiGLU.',
    'Checks exact original replay, broader paraphrase persistence, and multi-neighborhood layer-shift controls without fitting a surrogate.'
  ]
});

console.log(JSON.stringify({ pass, output: output.jsonOutputPath, results }, null, 2));
if (!pass) process.exit(1);
