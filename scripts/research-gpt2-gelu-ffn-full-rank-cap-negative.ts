#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

type FullFfnFixture = {
  type: string;
  dimension: number;
  num_channels: number;
  description: string;
  W_fc_T: number[][];
  b_fc: number[];
  w_proj: number[];
  b_proj: number;
};

const repoRoot = path.resolve(__dirname, '..');
const dataDir = process.env.GCL_GPT2_FFN_DATA_DIR
  ? path.resolve(process.env.GCL_GPT2_FFN_DATA_DIR)
  : path.resolve(repoRoot, '..', 'external-pytorch-test');
const fixturePath = process.env.GCL_GPT2_FULL_FFN_FIXTURE
  ? path.resolve(process.env.GCL_GPT2_FULL_FFN_FIXTURE)
  : path.join(dataDir, 'gpt2_full_ffn_v2.json');
const rankCap = Math.max(1, Math.floor(Number(process.env.GCL_GPT2_FULL_RANK_CAP ?? 20)));
const sampleCount = Math.max(32, Math.floor(Number(process.env.GCL_GPT2_FULL_AUDIT_SAMPLES ?? 96)));
const outputDir = path.join(repoRoot, 'benchmarks', 'research', 'nn-gpt2-gelu-ffn-full-rank-cap-negative', 'local-full-fixture-audit');
const jsonOutputPath = path.join(outputDir, 'report.json');
const markdownOutputPath = path.join(outputDir, 'report.md');

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function dot(left: number[], right: number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index++) sum += left[index] * right[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * 0.5)
  ));
}

function assertFixture(fixture: FullFfnFixture): void {
  if (!Number.isInteger(fixture.dimension) || fixture.dimension <= 0) throw new Error('GPT-2 full FFN fixture has invalid dimension');
  if (!Number.isInteger(fixture.num_channels) || fixture.num_channels <= 0) throw new Error('GPT-2 full FFN fixture has invalid num_channels');
  if (fixture.W_fc_T.length !== fixture.num_channels) throw new Error('GPT-2 full FFN fixture W_fc_T channel count mismatch');
  if (fixture.b_fc.length !== fixture.num_channels) throw new Error('GPT-2 full FFN fixture b_fc channel count mismatch');
  if (fixture.w_proj.length !== fixture.num_channels) throw new Error('GPT-2 full FFN fixture w_proj channel count mismatch');
  for (const [index, weights] of fixture.W_fc_T.entries()) {
    if (weights.length !== fixture.dimension) throw new Error(`GPT-2 W_fc_T[${index}] dimension mismatch`);
    if (!weights.every(Number.isFinite)) throw new Error(`GPT-2 W_fc_T[${index}] contains non-finite values`);
  }
}

function predictFixture(fixture: FullFfnFixture, input: number[], rankedIndices?: number[]): number {
  let value = fixture.b_proj;
  const indices = rankedIndices ?? fixture.w_proj.map((_unused, index) => index);
  for (const sourceIndex of indices) {
    value += fixture.w_proj[sourceIndex] * gelu(dot(fixture.W_fc_T[sourceIndex], input) + fixture.b_fc[sourceIndex]);
  }
  return value;
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function makeMarkdown(report: Record<string, any>): string {
  return [
    '# Real GPT-2 Full-FFN Rank-Cap Audit',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    '## Summary',
    '',
    `- Pass: ${report.summary.pass}`,
    `- Claim decision: ${report.summary.claimDecision}`,
    `- Fixture: ${report.fixture.path}`,
    `- Dimension: ${report.fixture.dimension}`,
    `- Source channels: ${report.fixture.numChannels}`,
    `- Rank cap: ${report.summary.rankCap}`,
    `- Top-rank source coverage: ${report.summary.topRankCoverage}`,
    `- R2 top-rank vs full: ${report.summary.metrics.r2}`,
    `- MAE top-rank vs full: ${report.summary.metrics.mae}`,
    `- Worst error: ${report.summary.worstError}`,
    '',
    '## Decision Reasons',
    '',
    ...report.summary.claimReasons.map((reason: string) => `- ${reason}`),
    '',
    '## Boundary',
    '',
    'This is a local research audit over the full GPT-2 fixture. It is not a release-gated full-FFN extraction claim.'
  ].join('\n');
}

if (!fs.existsSync(fixturePath)) {
  throw new Error(`Missing GPT-2 full FFN fixture: ${fixturePath}. Set GCL_GPT2_FULL_FFN_FIXTURE or GCL_GPT2_FFN_DATA_DIR.`);
}

const started = Date.now();
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as FullFfnFixture;
assertFixture(fixture);
if (rankCap >= fixture.num_channels) throw new Error(`Rank cap ${rankCap} must be below full channel count ${fixture.num_channels} for a negative audit`);

const ranked = fixture.w_proj
  .map((outputWeight, index) => ({
    index,
    contributionScore: Math.abs(outputWeight) * norm(fixture.W_fc_T[index])
  }))
  .sort((left, right) => right.contributionScore - left.contributionScore);
const totalContributionScore = ranked.reduce((sum, item) => sum + item.contributionScore, 0);
const selected = ranked.slice(0, rankCap);
const selectedIndices = selected.map(item => item.index);
const topRankCoverage = selected.reduce((sum, item) => sum + item.contributionScore, 0) / Math.max(1e-12, totalContributionScore);
const inputs = makeInputs(903031, fixture.dimension, sampleCount);
const full = inputs.map(input => predictFixture(fixture, input));
const capped = inputs.map(input => predictFixture(fixture, input, selectedIndices));
const metrics = regressionMetrics(full, capped);
const worstError = worstAbsoluteError(full, capped);
const claimReasons = [
  `rank cap ${rankCap} is below full GPT-2 source channel count ${fixture.num_channels}`,
  `top-rank contribution coverage is ${topRankCoverage.toFixed(6)}`,
  `held-out capped-vs-full R2 is ${metrics.r2.toFixed(6)}`
];
const claimDecision = (topRankCoverage < 0.70 || metrics.r2 < 0.995)
  ? 'insufficient-rank'
  : 'surrogate-only';
const gates = {
  fullFixtureLoaded: fixture.dimension === 768 && fixture.num_channels === 3072,
  expectedNegativeDecision: claimDecision === 'insufficient-rank',
  rankCapBelowFullChannelCount: rankCap < fixture.num_channels,
  notReleaseClaim: true,
  finiteMetrics: Number.isFinite(metrics.r2) && Number.isFinite(metrics.mae) && Number.isFinite(worstError)
};
const pass = Object.values(gates).every(Boolean);
const summary = {
  pass,
  claimDecision,
  claimReasons,
  rankCap,
  selectedSourceIndices: selectedIndices,
  topRankCoverage,
  metrics,
  worstError,
  sampleCount
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'Local real GPT-2 full FFN rank-cap negative audit',
  fixture: {
    path: fixturePath,
    type: fixture.type,
    dimension: fixture.dimension,
    numChannels: fixture.num_channels
  },
  claimScope: {
    provesIfPasses: 'A small rank cap is correctly rejected as insufficient for a full GPT-2 layer-0 FFN output slice on this local fixture.',
    doesNotClaim: 'output-only full GPT-2 FFN decompilation, original-neuron identity, package-backed release proof, or broad SOTA status.'
  },
  summary,
  gates,
  topSources: selected.map(item => ({
    index: item.index,
    contributionScore: item.contributionScore
  })),
  runtimeMs: Date.now() - started
};

fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(markdownOutputPath, `${makeMarkdown(report)}\n`);

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
