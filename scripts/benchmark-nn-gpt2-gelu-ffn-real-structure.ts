#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileRankAdaptiveGeluFfnStructure,
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
  sources: Array<GeluFfnSourceNeuron & {
    contributionRank: number;
    contributionScore: number;
  }>;
};

const repoRoot = path.resolve(__dirname, '..');
const fixturePath = process.env.GCL_GPT2_COMPACT_FIXTURE
  ? path.resolve(process.env.GCL_GPT2_COMPACT_FIXTURE)
  : path.join(repoRoot, 'benchmarks', 'fixtures', 'gpt2-layer0-output0-top3-gelu.json');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gpt2-gelu-ffn-real-structure',
  'layer0-output0-topk-source-mapping'
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

function makeInputs(seed: number, dimension: number, count: number, scale: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * scale)
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
  if (!Number.isFinite(fixture.outputBias)) throw new Error('GPT-2 compact fixture outputBias must be finite');
  for (const source of fixture.sources) {
    if (source.inputWeight.length !== fixture.dimension) throw new Error(`GPT-2 source ${source.id} dimension mismatch`);
    if (!source.inputWeight.every(Number.isFinite)) throw new Error(`GPT-2 source ${source.id} contains non-finite input weights`);
    if (!Number.isFinite(source.inputBias) || !Number.isFinite(source.outputWeight)) {
      throw new Error(`GPT-2 source ${source.id} inputBias/outputWeight must be finite`);
    }
    if ((source.activation ?? 'gelu') !== 'gelu') throw new Error(`GPT-2 source ${source.id} has unsupported activation`);
  }
}

function makeMarkdown(report: Record<string, any>): string {
  const matches = report.provenance.matches as Array<Record<string, any>>;
  const sourceRows = report.sources as Array<Record<string, any>>;
  return [
    '# Real GPT-2 GELU FFN Structure Benchmark',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    '## Claim Scope',
    '',
    `- Proves if pass: ${report.claimScope.provesIfPasses}`,
    `- Does not claim: ${report.claimScope.doesNotClaim}`,
    '',
    '## Fixture',
    '',
    `- Source fixture type: ${report.fixture.sourceFixtureType}`,
    `- Source fixture SHA-256: ${report.fixture.sourceFixtureSha256}`,
    `- Dimension: ${report.fixture.dimension}`,
    `- Output channel: ${report.fixture.outputChannelIndex}`,
    `- Top-K: ${report.fixture.topK}`,
    `- Selected by: ${report.fixture.selectedBy}`,
    '',
    '## Summary',
    '',
    `- Pass: ${report.summary.pass}`,
    `- Effective channels: ${report.summary.effectiveChannelCount}`,
    `- Source groups: ${report.summary.sourceGroupCount}`,
    `- R2: ${report.summary.metrics.r2}`,
    `- MAE: ${report.summary.metrics.mae}`,
    `- Worst error: ${report.summary.worstError}`,
    `- Agreement MAE: ${report.summary.agreementMae}`,
    `- Min provenance confidence: ${report.summary.minProvenanceConfidence}`,
    `- Min direction cosine: ${report.summary.minDirectionCosine}`,
    `- Raw oracle calls: ${report.summary.rawOracleCalls}`,
    '',
    '## Real GPT-2 Source Channels',
    '',
    ...sourceRows.flatMap(source => [
      `- ${source.id}`,
      `  - contribution rank: ${source.contributionRank}`,
      `  - GPT-2 source index: ${source.index}`,
      `  - input norm: ${source.inputWeightNorm}`,
      `  - input bias: ${source.inputBias}`,
      `  - output weight: ${source.outputWeight}`
    ]),
    '',
    '## Provenance Matches',
    '',
    ...matches.flatMap(match => [
      `- recovered atom ${match.recoveredAtomIndex} -> ${match.sourceGroupId}`,
      `  - source ids: ${(match.sourceIds as string[]).join(', ')}`,
      `  - confidence: ${match.confidence}`,
      `  - direction cosine: ${match.directionCosine}`,
      `  - norm error: ${match.inputWeightNormError}`,
      `  - bias error: ${match.inputBiasError}`,
      `  - output weight error: ${match.outputWeightError}`,
      `  - contribution correlation: ${match.contributionCorrelation}`,
      `  - why: ${match.whyReacts.responseRule}`
    ]),
    '',
    '## Conclusion',
    '',
    report.summary.pass
      ? 'The real GPT-2 compact FFN structure gate passed. GCL recovered explicit GELU source atoms from oracle calls over real GPT-2 layer-0 weights and verified source-channel provenance against the compact source manifest.'
      : 'The real GPT-2 compact FFN structure gate failed. Do not claim real GPT-2 source-structure recovery until the failing condition is repaired.'
  ].join('\n');
}

if (!fs.existsSync(fixturePath)) {
  throw new Error(`Missing compact GPT-2 GELU FFN fixture: ${fixturePath}. Run npm run generate:gpt2-gelu-ffn-compact-fixture first.`);
}

const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as CompactFixture;
assertFixture(fixture);
const started = Date.now();
const sources: GeluFfnSourceNeuron[] = fixture.sources.map(source => ({
  id: source.id,
  layer: source.layer,
  index: source.index,
  origin: source.origin,
  activation: 'gelu',
  inputWeight: source.inputWeight,
  inputBias: source.inputBias,
  outputWeight: source.outputWeight
}));
const oracle = (input: number[]): number => predictGeluFfnSourceNeurons(sources, fixture.outputBias, input);
const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
  dimension: fixture.dimension,
  ranges: Array.from({ length: fixture.dimension }, () => [-1, 1] as [number, number]),
  maxChannelCount: fixture.topK,
  minEffectiveChannelCount: 1,
  activeRank: fixture.topK,
  selectionRestarts: 1,
  gradientProbeCount: Math.max(14, fixture.topK * 7),
  trainingSampleCount: Math.max(3200, fixture.topK * 1900),
  validationSampleCount: 780,
  randomSeed: 66203 + fixture.topK,
  restarts: fixture.topK === 1 ? 10 : 16,
  iterations: fixture.topK === 1 ? 1100 : 1350,
  learningRate: fixture.topK === 1 ? 0.03 : 0.024,
  atomRelativeStrengthThreshold: 0.012,
  atomAbsoluteStrengthThreshold: 0.0005,
  pruneValidationToleranceRatio: 0.08,
  pruneValidationToleranceAbsolute: 0.006
});
const validationInputs = makeInputs(66303 + fixture.topK, fixture.dimension, 1400, 0.5);
const actual = validationInputs.map(oracle);
const predicted = predictGeluFfnStructureBatch(model, validationInputs);
const metrics = regressionMetrics(actual, predicted);
const worstError = worstAbsoluteError(actual, predicted);
const agreementMae = modelAgreementMae(model, sources, fixture.outputBias, validationInputs);
const provenance = verifyGeluFfnNeuronProvenance(model, sources, {
  probeInputs: validationInputs.slice(0, 420),
  sourceIdentityMode: 'functional-equivalence',
  minDirectionCosine: 0.9975,
  maxInputWeightNormError: 0.34,
  maxInputBiasError: 0.18,
  maxOutputWeightError: 0.18,
  minActivationCorrelation: 0.996,
  minContributionCorrelation: 0.996,
  maxContributionMae: 0.065,
  minConfidence: 0.70
});
const confidences = provenance.matches.map(match => match.confidence);
const directionCosines = provenance.matches.map(match => match.directionCosine);
const gates = {
  realFixtureLoaded: fixture.type.startsWith('gpt2-layer0-output0-top') && fixture.sourceFixtureType.includes('gpt2'),
  recoversRankAdaptiveStructure: model.recoveryKind === 'rank-adaptive-gelu-ffn-structure',
  recoveredExpectedEffectiveChannels: model.effectiveChannelCount === fixture.topK,
  outputFaithful: metrics.r2 > 0.994 && metrics.mae < 0.065 && worstError < 0.45,
  modelAgreementSmall: agreementMae < 0.065,
  provenanceVerified: provenance.verified,
  allSourceGroupsMapped: provenance.gates.allSourceGroupsMatched,
  allRecoveredAtomsMapped: provenance.gates.allRecoveredAtomsMatched,
  whyReactEvidencePresent: provenance.matches.every(match => match.whyReacts.topTriggers.length >= 3),
  rawOracleCallsBelow260000: model.queryBudget.rawOracleCalls < 260000
};
const pass = Object.values(gates).every(Boolean);
const summary = {
  pass,
  topK: fixture.topK,
  dimension: fixture.dimension,
  effectiveChannelCount: model.effectiveChannelCount,
  rawChannelCount: model.rawChannelCount,
  sourceGroupCount: provenance.sourceGroupCount,
  metrics,
  worstError,
  agreementMae,
  minProvenanceConfidence: confidences.length ? Math.min(...confidences) : 0,
  minDirectionCosine: directionCosines.length ? Math.min(...directionCosines) : 0,
  rawOracleCalls: model.queryBudget.rawOracleCalls
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'Real GPT-2 layer-0 GELU FFN compact source-structure recovery',
  blackBoxBoundary: {
    oracle: 'GCL receives f(vector) for the compact real GPT-2 FFN output slice during recovery',
    provenance: 'Source manifest is used only after recovery to verify exact GPT-2 channel ids, weights, and why-reacts evidence.',
    notASurrogateGate: 'Pass requires recovered atom count and per-source provenance matches, not only output R2.'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit GELU atoms from a real GPT-2 layer-0 FFN top-K output-channel slice and map them to exact GPT-2 source channels through provenance verification.',
    doesNotClaim: 'full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary tokenizer/model execution, output-only original-neuron identity without source manifest/hooks, or broad SOTA status.'
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
  sources: fixture.sources.map(source => ({
    id: source.id,
    layer: source.layer,
    index: source.index,
    origin: source.origin,
    contributionRank: source.contributionRank,
    contributionScore: source.contributionScore,
    inputWeightNorm: norm(source.inputWeight),
    inputBias: source.inputBias,
    outputWeight: source.outputWeight
  })),
  model: {
    recoveryKind: model.recoveryKind,
    effectiveChannelCount: model.effectiveChannelCount,
    rawChannelCount: model.rawChannelCount,
    selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
    validationMetrics: model.validationMetrics,
    queryBudget: model.queryBudget,
    atoms: model.atoms.map(atom => ({
      inputWeightNorm: atom.inputWeightNorm,
      inputBias: atom.inputBias,
      outputWeight: atom.outputWeight,
      strength: Math.abs(atom.outputWeight) * atom.inputWeightNorm
    }))
  },
  provenance,
  runtimeMs: Date.now() - started
};

fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${makeMarkdown(report)}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gpt2-gelu-ffn-real-structure',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Release-gated compact real GPT-2 layer-0 FFN source-structure benchmark.',
    'Uses real GPT-2 weights extracted into a compact top-K fixture.',
    'Pass requires explicit GELU atom recovery plus source-channel provenance; output-only surrogate fit is insufficient.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
