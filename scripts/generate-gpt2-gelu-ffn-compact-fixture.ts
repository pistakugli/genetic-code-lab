#!/usr/bin/env ts-node

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

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

type CompactSource = {
  id: string;
  layer: string;
  index: number;
  origin: string;
  activation: 'gelu';
  inputWeight: number[];
  inputBias: number;
  outputWeight: number;
  contributionRank: number;
  contributionScore: number;
};

type CompactFixture = {
  schemaVersion: 1;
  type: string;
  generatedAt: string;
  sourceFixtureType: string;
  sourceFixtureSha256: string;
  dimension: number;
  outputChannelIndex: number;
  selectedBy: 'abs(w_proj[j]) * ||W_fc_T[j]||';
  topK: number;
  outputBias: number;
  sources: CompactSource[];
  notes: string[];
};

const repoRoot = path.resolve(__dirname, '..');
const defaultExternalDir = path.resolve(repoRoot, '..', 'external-pytorch-test');
const dataDir = process.env.GCL_GPT2_FFN_DATA_DIR
  ? path.resolve(process.env.GCL_GPT2_FFN_DATA_DIR)
  : defaultExternalDir;
const topK = Math.max(1, Math.floor(Number(process.env.GCL_GPT2_FFN_TOPK ?? 3)));
const outputPath = process.env.GCL_GPT2_COMPACT_FIXTURE
  ? path.resolve(process.env.GCL_GPT2_COMPACT_FIXTURE)
  : path.join(repoRoot, 'benchmarks', 'fixtures', `gpt2-layer0-output0-top${topK}-gelu.json`);

function dot(left: number[], right: number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index++) sum += left[index] * right[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function assertFixture(fixture: FullFfnFixture): void {
  if (!Number.isInteger(fixture.dimension) || fixture.dimension <= 0) throw new Error('GPT-2 full FFN fixture has invalid dimension');
  if (!Number.isInteger(fixture.num_channels) || fixture.num_channels <= 0) throw new Error('GPT-2 full FFN fixture has invalid num_channels');
  if (fixture.W_fc_T.length !== fixture.num_channels) throw new Error('GPT-2 full FFN fixture W_fc_T channel count mismatch');
  if (fixture.b_fc.length !== fixture.num_channels) throw new Error('GPT-2 full FFN fixture b_fc channel count mismatch');
  if (fixture.w_proj.length !== fixture.num_channels) throw new Error('GPT-2 full FFN fixture w_proj channel count mismatch');
  for (const [index, weights] of fixture.W_fc_T.entries()) {
    if (weights.length !== fixture.dimension) throw new Error(`GPT-2 full FFN fixture W_fc_T[${index}] dimension mismatch`);
    if (!weights.every(Number.isFinite)) throw new Error(`GPT-2 full FFN fixture W_fc_T[${index}] contains non-finite values`);
  }
}

const sourcePath = path.join(dataDir, 'gpt2_full_ffn_v2.json');
if (!fs.existsSync(sourcePath)) {
  throw new Error(`Missing GPT-2 full FFN fixture: ${sourcePath}. Set GCL_GPT2_FFN_DATA_DIR to the folder containing gpt2_full_ffn_v2.json.`);
}

const raw = fs.readFileSync(sourcePath);
const full = JSON.parse(raw.toString('utf8')) as FullFfnFixture;
assertFixture(full);
if (topK > full.num_channels) throw new Error(`Requested topK ${topK} exceeds GPT-2 fixture channel count ${full.num_channels}`);

const ranked = full.w_proj
  .map((outputWeight, index) => ({
    index,
    contributionScore: Math.abs(outputWeight) * norm(full.W_fc_T[index])
  }))
  .sort((left, right) => right.contributionScore - left.contributionScore);

const sources: CompactSource[] = ranked.slice(0, topK).map((item, rank) => ({
  id: `gpt2.transformer.h0.mlp.c_fc[${item.index}]->c_proj[0]`,
  layer: 'gpt2.transformer.h0.mlp.c_fc',
  index: item.index,
  origin: `real GPT-2 layer 0 FFN expansion channel ${item.index} feeding output channel 0`,
  activation: 'gelu',
  inputWeight: full.W_fc_T[item.index],
  inputBias: full.b_fc[item.index],
  outputWeight: full.w_proj[item.index],
  contributionRank: rank + 1,
  contributionScore: item.contributionScore
}));

const compact: CompactFixture = {
  schemaVersion: 1,
  type: `gpt2-layer0-output0-top${topK}-gelu-source-fixture`,
  generatedAt: new Date().toISOString(),
  sourceFixtureType: full.type,
  sourceFixtureSha256: crypto.createHash('sha256').update(raw).digest('hex'),
  dimension: full.dimension,
  outputChannelIndex: 0,
  selectedBy: 'abs(w_proj[j]) * ||W_fc_T[j]||',
  topK,
  outputBias: full.b_proj,
  sources,
  notes: [
    'Compact fixture extracted from real GPT-2 layer-0 FFN weights.',
    'Only the selected top-K source channels are stored so release benchmarks do not commit the full 49MB fixture.',
    'The benchmark must recover explicit GELU source structure; output-only surrogate fit is not sufficient.'
  ]
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(compact, null, 2)}\n`);
console.log(JSON.stringify({
  wrote: path.relative(repoRoot, outputPath),
  topK,
  dimension: compact.dimension,
  sourceFixtureSha256: compact.sourceFixtureSha256,
  channels: sources.map(source => source.index)
}, null, 2));
