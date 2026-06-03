#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  GeluFfnStructureModel,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import {
  GeluFfnSourceNeuron,
  modelAgreementMae,
  predictGeluFfnSourceNeurons,
  verifyGeluFfnNeuronProvenance
} from '../src/solvers/gelu-ffn-provenance-verifier';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type CompactSource = GeluFfnSourceNeuron & {
  contributionRank: number;
  contributionScore: number;
};

type CompactFixture = {
  schemaVersion: 1;
  type: string;
  sourceFixtureType: string;
  sourceFixtureSha256: string;
  dimension: number;
  outputChannelIndex: number;
  selectedBy: string;
  topK: number;
  outputBias: number;
  sources: CompactSource[];
};

const repoRoot = path.resolve(__dirname, '..');
const defaultFixturePath = path.join(repoRoot, 'benchmarks', 'fixtures', 'gpt2-layer0-output0-top20-gelu.json');
const fixturePath = process.env.GCL_GPT2_SOURCE_EXTRACTOR_FIXTURE
  ? path.resolve(process.env.GCL_GPT2_SOURCE_EXTRACTOR_FIXTURE)
  : defaultFixturePath;
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gpt2-gelu-ffn-source-extractor',
  'layer0-output0-top20-exact-source-graph'
);

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

function dot(left: number[], right: number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index++) sum += left[index] * right[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function makeInputs(seed: number, dimension: number, count: number, scale: number, shift = 0): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => shift + gaussian(random) * scale)
  ));
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function assertFixture(fixture: CompactFixture): void {
  if (fixture.schemaVersion !== 1) throw new Error('GPT-2 compact fixture schemaVersion must be 1');
  if (!Number.isInteger(fixture.dimension) || fixture.dimension <= 0) throw new Error('GPT-2 compact fixture has invalid dimension');
  if (!Number.isInteger(fixture.topK) || fixture.topK <= 0) throw new Error('GPT-2 compact fixture has invalid topK');
  if (fixture.sources.length !== fixture.topK) throw new Error('GPT-2 compact fixture source count must equal topK');
  for (const source of fixture.sources) {
    if (source.inputWeight.length !== fixture.dimension) throw new Error(`GPT-2 source ${source.id} dimension mismatch`);
    if (!source.inputWeight.every(Number.isFinite)) throw new Error(`GPT-2 source ${source.id} input weights must be finite`);
    if (!Number.isFinite(source.inputBias) || !Number.isFinite(source.outputWeight)) {
      throw new Error(`GPT-2 source ${source.id} inputBias/outputWeight must be finite`);
    }
  }
}

function buildSourceGraphModel(fixture: CompactFixture, sources: GeluFfnSourceNeuron[]): GeluFfnStructureModel {
  const atoms = sources.map(source => ({
    inputWeight: [...source.inputWeight],
    inputBias: source.inputBias,
    outputWeight: source.outputWeight,
    inputWeightNorm: norm(source.inputWeight)
  }));
  const modelBase = {
    dimension: fixture.dimension,
    ranges: Array.from({ length: fixture.dimension }, () => [-1, 1] as [number, number]),
    channelCount: atoms.length,
    rawChannelCount: atoms.length,
    effectiveChannelCount: atoms.length,
    activeRank: atoms.length,
    gradientProbeCount: 0,
    trainingSampleCount: 0,
    validationSampleCount: 0,
    finiteDifferenceStep: 0,
    querySamples: 1,
    aggregation: 'mean' as const,
    recoveryKind: 'gelu-ffn-structure' as const,
    activeBasis: [],
    atoms,
    outputBias: fixture.outputBias,
    trainingMse: 0,
    validationMetrics: { mae: 0, rmse: 0, r2: 1 },
    optimizer: { restarts: 0, iterations: 0, learningRate: 0 },
    queryBudget: {
      rawOracleCalls: 0,
      gradientRawOracleCalls: 0,
      trainingRawOracleCalls: 0,
      validationRawOracleCalls: 0,
      finiteDifferenceCallsPerProbe: 0,
      aggregatedProbePoints: 0,
      querySamples: 1
    }
  };
  return {
    ...modelBase,
    expression: `gpt2_gelu_source_graph(output=${fixture.outputChannelIndex}, channels=${sources.map(source => source.index).join(',')})`
  };
}

function evaluateDistribution(
  id: string,
  inputs: number[][],
  model: GeluFfnStructureModel,
  sources: GeluFfnSourceNeuron[],
  outputBias: number
) {
  const actual = inputs.map(input => predictGeluFfnSourceNeurons(sources, outputBias, input));
  const predicted = predictGeluFfnStructureBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  return {
    id,
    rows: inputs.length,
    metrics,
    worstError: worstAbsoluteError(actual, predicted)
  };
}

function makeMarkdown(report: Record<string, any>): string {
  return [
    '# Real GPT-2 GELU FFN Source Extractor Benchmark',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    '## Claim Scope',
    '',
    `- Proves if pass: ${report.claimScope.provesIfPasses}`,
    `- Does not claim: ${report.claimScope.doesNotClaim}`,
    '',
    '## Summary',
    '',
    `- Pass: ${report.summary.pass}`,
    `- Top-K: ${report.summary.topK}`,
    `- Dimension: ${report.summary.dimension}`,
    `- Source channels: ${report.summary.sourceChannels.join(', ')}`,
    `- Max MAE: ${report.summary.maxMae}`,
    `- Min R2: ${report.summary.minR2}`,
    `- Max worst error: ${report.summary.maxWorstError}`,
    `- Provenance confidence min: ${report.summary.minProvenanceConfidence}`,
    `- Raw validation oracle calls: ${report.summary.rawValidationOracleCalls}`,
    '',
    '## Distributions',
    '',
    ...report.distributions.flatMap((row: Record<string, any>) => [
      `- ${row.id}: R2=${row.metrics.r2}, MAE=${row.metrics.mae}, worst=${row.worstError}`
    ]),
    '',
    '## Conclusion',
    '',
    report.summary.pass
      ? 'The real GPT-2 source extractor gate passed. GCL produced an exact explicit GELU source graph from a real GPT-2 FFN source manifest and verified every source channel with strict provenance.'
      : 'The real GPT-2 source extractor gate failed. Do not claim instrumented GPT-2 source graph extraction until the failing condition is repaired.'
  ].join('\n');
}

if (!fs.existsSync(fixturePath)) throw new Error(`Missing GPT-2 source extractor fixture: ${fixturePath}`);
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as CompactFixture;
assertFixture(fixture);
const started = Date.now();
const sources: GeluFfnSourceNeuron[] = fixture.sources.map(source => ({
  id: source.id,
  layer: source.layer,
  index: source.index,
  origin: source.origin,
  inputWeight: source.inputWeight,
  inputBias: source.inputBias,
  outputWeight: source.outputWeight,
  activation: 'gelu'
}));
const model = buildSourceGraphModel(fixture, sources);
const distributionInputs = [
  { id: 'iid-gpt2-activation-scale', inputs: makeInputs(81001, fixture.dimension, 1200, 0.5) },
  { id: 'wide-ood', inputs: makeInputs(81002, fixture.dimension, 1200, 0.75) },
  { id: 'shifted-ood', inputs: makeInputs(81003, fixture.dimension, 1200, 0.45, 0.08) }
];
const distributions = distributionInputs.map(row => evaluateDistribution(row.id, row.inputs, model, sources, fixture.outputBias));
const probeInputs = distributionInputs.flatMap(row => row.inputs.slice(0, 160));
const provenance = verifyGeluFfnNeuronProvenance(model, sources, {
  probeInputs,
  sourceIdentityMode: 'strict-parameter',
  minDirectionCosine: 0.999999999,
  maxInputWeightNormError: 1e-10,
  maxInputBiasError: 1e-10,
  maxOutputWeightError: 1e-10,
  minActivationCorrelation: 0.999999999,
  minContributionCorrelation: 0.999999999,
  maxContributionMae: 1e-10,
  minConfidence: 0.999999999
});
const agreementMae = modelAgreementMae(model, sources, fixture.outputBias, probeInputs);
const confidences = provenance.matches.map(match => match.confidence);
const maxMae = Math.max(...distributions.map(row => row.metrics.mae));
const minR2 = Math.min(...distributions.map(row => row.metrics.r2));
const maxWorstError = Math.max(...distributions.map(row => row.worstError));
const rawValidationOracleCalls = distributionInputs.reduce((sum, row) => sum + row.inputs.length, 0);
model.queryBudget.rawOracleCalls = rawValidationOracleCalls;
model.queryBudget.validationRawOracleCalls = rawValidationOracleCalls;
model.queryBudget.aggregatedProbePoints = rawValidationOracleCalls;
model.validationSampleCount = rawValidationOracleCalls;
model.validationMetrics = {
  mae: maxMae,
  rmse: Math.max(...distributions.map(row => row.metrics.rmse)),
  r2: minR2
};
const gates = {
  realFixtureLoaded: fixture.type.startsWith('gpt2-layer0-output0-top') && fixture.sourceFixtureType.includes('gpt2'),
  exactSourceGraphBuilt: model.atoms.length === fixture.topK && model.effectiveChannelCount === fixture.topK,
  strictProvenanceVerified: provenance.verified,
  allSourceGroupsMapped: provenance.gates.allSourceGroupsMatched,
  allRecoveredAtomsMapped: provenance.gates.allRecoveredAtomsMatched,
  exactAgreement: agreementMae < 1e-10,
  distributionsExact: distributions.every(row => row.metrics.r2 > 0.999999999 && row.metrics.mae < 1e-10 && row.worstError < 1e-9),
  rawValidationCallsBelow5000: rawValidationOracleCalls < 5000
};
const pass = Object.values(gates).every(Boolean);
const summary = {
  pass,
  topK: fixture.topK,
  dimension: fixture.dimension,
  sourceChannels: sources.map(source => source.index),
  maxMae,
  minR2,
  maxWorstError,
  agreementMae,
  minProvenanceConfidence: confidences.length ? Math.min(...confidences) : 0,
  rawValidationOracleCalls
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'Real GPT-2 layer-0 GELU FFN source graph extraction',
  blackBoxBoundary: {
    mode: 'instrumented-source-extraction',
    source: 'compact source manifest extracted from real GPT-2 layer-0 FFN weights',
    note: 'This gate verifies exact source graph extraction when source weights or equivalent hooks are available. It is intentionally separate from output-only black-box recovery.'
  },
  claimScope: {
    provesIfPasses: 'GCL can build and validate an exact explicit GELU source graph for a real GPT-2 FFN output slice from an instrumented source manifest, then map every atom to exact GPT-2 channel ids/layers/indices with strict provenance.',
    doesNotClaim: 'output-only original-neuron identity, blind full-FFN decompilation, arbitrary 3072-channel GPT-2 extraction, tokenizer/model execution, or broad SOTA status.'
  },
  fixture: {
    path: path.relative(repoRoot, fixturePath),
    type: fixture.type,
    sourceFixtureType: fixture.sourceFixtureType,
    sourceFixtureSha256: fixture.sourceFixtureSha256,
    dimension: fixture.dimension,
    outputChannelIndex: fixture.outputChannelIndex,
    selectedBy: fixture.selectedBy,
    topK: fixture.topK
  },
  summary,
  gates,
  distributions,
  provenance,
  model: {
    expression: model.expression,
    recoveryKind: model.recoveryKind,
    effectiveChannelCount: model.effectiveChannelCount,
    queryBudget: model.queryBudget
  },
  runtimeMs: Date.now() - started
};

fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${makeMarkdown(report)}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gpt2-gelu-ffn-source-extractor',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Release-gated real GPT-2 source extractor benchmark.',
    'This is instrumented source graph extraction, not output-only black-box recovery.',
    'Pass requires strict source-channel provenance and exact held-out functional agreement.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
