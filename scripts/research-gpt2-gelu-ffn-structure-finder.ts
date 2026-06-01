#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileGeluFfnStructure,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { writeBenchmarkManifest } from './benchmark-paths';

type FullFfn = {
  dimension: number;
  W_fc_T: number[][];
  b_fc: number[];
  w_proj: number[];
  b_proj: number;
};

type SingleChannel = {
  dimension: number;
  weight: number[];
  bias: number;
  weight_norm?: number;
};

const repoRoot = path.resolve(__dirname, '..');
const defaultExternalDir = path.resolve(repoRoot, '..', 'external-pytorch-test');
const dataDir = process.env.GCL_GPT2_FFN_DATA_DIR
  ? path.resolve(process.env.GCL_GPT2_FFN_DATA_DIR)
  : defaultExternalDir;

const researchRoot = path.join(repoRoot, 'benchmarks', 'research');
const output = {
  rootDir: researchRoot,
  outputDir: path.join(researchRoot, 'nn-gpt2-gelu-ffn-structure-finder', 'real-gpt2-layer0-topk'),
  jsonOutputPath: path.join(researchRoot, 'nn-gpt2-gelu-ffn-structure-finder', 'real-gpt2-layer0-topk', 'report.json'),
  markdownOutputPath: path.join(researchRoot, 'nn-gpt2-gelu-ffn-structure-finder', 'real-gpt2-layer0-topk', 'report.md'),
  manifestOutputPath: path.join(researchRoot, 'nn-gpt2-gelu-ffn-structure-finder', 'real-gpt2-layer0-topk', 'manifest.json'),
  variant: 'real-gpt2-layer0-topk'
};

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

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

function makeInputs(seed: number, dimension: number, count: number, scale = 0.5): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * scale)
  ));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function bestProjection(trueWeight: number[], recoveredWeights: number[][]): number {
  const trueUnit = normalize(trueWeight);
  return Math.max(...recoveredWeights.map(weight => Math.abs(dot(trueUnit, normalize(weight)))));
}

function readJson<T>(fileName: string): T {
  const filePath = path.join(dataDir, fileName);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing GPT-2 FFN fixture: ${filePath}. Set GCL_GPT2_FFN_DATA_DIR to the folder containing gpt2_single_channel_v2.json and gpt2_full_ffn_v2.json.`);
  }
  return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
}

const started = Date.now();
const single = readJson<SingleChannel>('gpt2_single_channel_v2.json');
const full = readJson<FullFfn>('gpt2_full_ffn_v2.json');
const dimension = full.dimension;
const ranges = Array.from({ length: dimension }, () => [-1, 1] as [number, number]);

if (single.dimension !== dimension) throw new Error('Single-channel and full-FFN fixtures disagree on dimension');

const results: Array<Record<string, unknown>> = [];

{
  const oracle = (input: number[]) => gelu(dot(single.weight, input) + single.bias);
  let rawOracleCalls = 0;
  const countedOracle = (input: number[]) => {
    rawOracleCalls++;
    return oracle(input);
  };
  const model = decompileGeluFfnStructure(countedOracle, {
    dimension,
    ranges,
    channelCount: 1,
    activeRank: 1,
    gradientProbeCount: 10,
    trainingSampleCount: 2600,
    validationSampleCount: 700,
    randomSeed: 6101,
    restarts: 10,
    iterations: 1100,
    learningRate: 0.03
  });
  const validationInputs = makeInputs(6102, dimension, 1200);
  const actual = validationInputs.map(oracle);
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const atom = model.atoms[0];
  results.push({
    id: 'real-gpt2-single-gelu-768d',
    kind: 'single-channel-gelu',
    pass: metrics.r2 > 0.999 && metrics.mae < 0.01 && bestProjection(single.weight, [atom.inputWeight]) > 0.9999,
    metrics,
    rawOracleCalls,
    projection: bestProjection(single.weight, [atom.inputWeight]),
    trueNorm: norm(single.weight),
    recoveredNorm: atom.inputWeightNorm,
    trueBias: single.bias,
    recoveredBias: atom.inputBias,
    outputWeight: atom.outputWeight
  });
}

const contributions = full.w_proj.map((outputWeight, index) => ({
  index,
  score: Math.abs(outputWeight) * norm(full.W_fc_T[index])
})).sort((a, b) => b.score - a.score);

for (const topK of [1, 2, 3]) {
  const channels = contributions.slice(0, topK).map(item => item.index);
  const oracle = (input: number[]) => channels.reduce((sum, channel) => (
    sum + full.w_proj[channel] * gelu(dot(full.W_fc_T[channel], input) + full.b_fc[channel])
  ), full.b_proj);
  let rawOracleCalls = 0;
  const countedOracle = (input: number[]) => {
    rawOracleCalls++;
    return oracle(input);
  };
  const model = decompileGeluFfnStructure(countedOracle, {
    dimension,
    ranges,
    channelCount: topK,
    activeRank: topK,
    gradientProbeCount: Math.max(12, topK * 7),
    trainingSampleCount: topK === 1 ? 3000 : topK * 1900,
    validationSampleCount: 800,
    randomSeed: 6200 + topK,
    restarts: topK === 1 ? 10 : 16,
    iterations: topK === 1 ? 1100 : 1300,
    learningRate: topK === 1 ? 0.03 : 0.024
  });
  const validationInputs = makeInputs(6300 + topK, dimension, 1200);
  const actual = validationInputs.map(oracle);
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const recoveredWeights = model.atoms.map(atom => atom.inputWeight);
  const projections = channels.map(channel => bestProjection(full.W_fc_T[channel], recoveredWeights));
  results.push({
    id: `real-gpt2-top${topK}-gelu-ffn-768d`,
    kind: 'topk-output-channel-gelu-sum',
    pass: metrics.r2 > 0.995 && metrics.mae < 0.06 && Math.min(...projections) > 0.9998,
    metrics,
    rawOracleCalls,
    channels,
    minProjection: Math.min(...projections),
    projections,
    atoms: model.atoms.map(atom => ({
      inputWeightNorm: atom.inputWeightNorm,
      inputBias: atom.inputBias,
      outputWeight: atom.outputWeight
    }))
  });
}

const pass = results.every(result => result.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'Research-only GPT-2 GELU FFN structure finder stress',
  blackBox: {
    type: 'real GPT-2 layer-0 FFN weights serialized as local JSON fixtures',
    dataDir,
    visibilityToGcl: 'oracle calls only: f(vector), no weights, no hidden activations'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit GELU atom directions and high-fidelity outputs for a real GPT-2 single neuron and top-1/top-3 truncated output-channel sums on this fixture',
    doesNotClaim: 'full GPT-2 FFN extraction, tokenizer integration, arbitrary layer extraction, arbitrary channel-count recovery, SwiGLU recovery, or release-gated production status'
  },
  gates: {
    pass,
    allCasesPassed: pass,
    minR2: Math.min(...results.map(result => (result.metrics as { r2: number }).r2)),
    maxMae: Math.max(...results.map(result => (result.metrics as { mae: number }).mae)),
    minProjection: Math.min(...results.map(result => (
      typeof result.projection === 'number' ? result.projection : result.minProjection as number
    )))
  },
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# GPT-2 GELU FFN Structure Finder Research Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Scope',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Summary',
  '',
  `- Pass: ${pass}`,
  `- Cases: ${results.filter(result => result.pass).length}/${results.length}`,
  `- Min R2: ${report.gates.minR2}`,
  `- Max MAE: ${report.gates.maxMae}`,
  `- Min direction projection: ${report.gates.minProjection}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.id}`,
    '',
    `- Pass: ${result.pass}`,
    `- R2: ${(result.metrics as { r2: number }).r2}`,
    `- MAE: ${(result.metrics as { mae: number }).mae}`,
    `- Raw oracle calls: ${result.rawOracleCalls}`,
    `- Projection: ${typeof result.projection === 'number' ? result.projection : result.minProjection}`,
    ''
  ].join('\n'))
].join('\n').trimEnd();

fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gpt2-gelu-ffn-structure-finder',
  status: pass ? 'pass' : 'fail',
  summary: report.gates,
  notes: [
    'Research-only external fixture benchmark for the GELU structure finder.',
    'Depends on GPT-2 JSON fixtures outside the package; intentionally not part of release:gate.',
    'This is structural atom recovery, not active-subspace KNN surrogate recovery.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.gates
}, null, 2));

if (!pass) process.exit(1);
