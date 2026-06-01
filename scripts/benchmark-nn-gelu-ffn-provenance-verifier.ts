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

type TeacherCase = {
  id: string;
  dimension: number;
  outputBias: number;
  sources: GeluFfnSourceNeuron[];
  expectedSourceGroups: number;
  maxChannelCount: number;
  seed: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-provenance-verifier',
  'source-neuron-mapping'
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
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

function normalize(values: number[]): number[] {
  const length = Math.sqrt(Math.max(1e-12, dot(values, values)));
  return values.map(value => value / length);
}

function makeDirection(seed: number, dimension: number, previous: number[][] = []): number[] {
  const random = makeRandom(seed);
  let vector = Array.from({ length: dimension }, () => gaussian(random));
  for (const direction of previous) {
    const projection = dot(vector, direction);
    vector = vector.map((value, index) => value - projection * direction[index]);
  }
  return normalize(vector);
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => -1.5 + 3 * random())
  ));
}

function makeSource(
  layer: string,
  index: number,
  origin: string,
  direction: number[],
  inputScale: number,
  inputBias: number,
  outputWeight: number
): GeluFfnSourceNeuron {
  return {
    id: `${layer}[${index}]`,
    layer,
    index,
    origin,
    inputWeight: direction.map(value => value * inputScale),
    inputBias,
    outputWeight,
    activation: 'gelu'
  };
}

function makeDistinctCase(): TeacherCase {
  const dimension = 24;
  const directions: number[][] = [];
  for (const seed of [32011, 32112, 32213]) directions.push(makeDirection(seed, dimension, directions));
  return {
    id: 'distinct-source-neurons-24d',
    dimension,
    outputBias: 0.06,
    maxChannelCount: 4,
    expectedSourceGroups: 3,
    seed: 32300,
    sources: [
      makeSource('transformer.h0.mlp.c_fc', 17, 'synthetic FFN expansion neuron 17', directions[0], 1.18, -0.2, 0.82),
      makeSource('transformer.h0.mlp.c_fc', 42, 'synthetic FFN expansion neuron 42', directions[1], 0.96, 0.24, -0.61),
      makeSource('transformer.h0.mlp.c_fc', 203, 'synthetic FFN expansion neuron 203', directions[2], 0.82, -0.08, 0.37)
    ]
  };
}

function makeDuplicateSplitCase(): TeacherCase {
  const dimension = 24;
  const directions: number[][] = [];
  for (const seed of [33011, 33112]) directions.push(makeDirection(seed, dimension, directions));
  return {
    id: 'duplicate-split-source-group-24d',
    dimension,
    outputBias: -0.03,
    maxChannelCount: 3,
    expectedSourceGroups: 2,
    seed: 33300,
    sources: [
      makeSource('transformer.h1.mlp.c_fc', 8, 'duplicate split source group A part 1', directions[0], 1.05, -0.18, 0.31),
      makeSource('transformer.h1.mlp.c_fc', 9, 'duplicate split source group A part 2', directions[0], 1.05, -0.18, 0.43),
      makeSource('transformer.h1.mlp.c_fc', 77, 'separate negative FFN expansion neuron', directions[1], 0.9, 0.14, -0.52)
    ]
  };
}

function runCase(testCase: TeacherCase) {
  const oracle = (input: number[]): number => predictGeluFfnSourceNeurons(testCase.sources, testCase.outputBias, input);
  const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
    dimension: testCase.dimension,
    ranges: Array.from({ length: testCase.dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: testCase.maxChannelCount,
    minEffectiveChannelCount: 1,
    selectionRestarts: 2,
    gradientProbeCount: 16,
    trainingSampleCount: 1100,
    validationSampleCount: 360,
    randomSeed: testCase.seed,
    restarts: 7,
    iterations: 480,
    learningRate: 0.028,
    atomRelativeStrengthThreshold: 0.02,
    atomAbsoluteStrengthThreshold: 0.004,
    pruneValidationToleranceRatio: 0.22,
    pruneValidationToleranceAbsolute: 0.009
  });
  const validationInputs = makeInputs(testCase.seed + 1000, testCase.dimension, 900);
  const actual = validationInputs.map(oracle);
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const report = verifyGeluFfnNeuronProvenance(model, testCase.sources, {
    probeInputs: validationInputs.slice(0, 300),
    minDirectionCosine: 0.992,
    maxInputWeightNormError: 0.18,
    maxInputBiasError: 0.16,
    maxOutputWeightError: 0.22,
    minActivationCorrelation: 0.997,
    minContributionCorrelation: 0.997,
    maxContributionMae: 0.018,
    minConfidence: 0.78
  });
  const agreementMae = modelAgreementMae(model, testCase.sources, testCase.outputBias, validationInputs);
  const gates = {
    outputFaithful: metrics.r2 > 0.998 && metrics.mae < 0.018,
    modelAgreementMaeSmall: agreementMae < 0.018,
    expectedSourceGroupCount: report.sourceGroupCount === testCase.expectedSourceGroups,
    allRecoveredAtomsMapped: report.gates.allRecoveredAtomsMatched,
    allSourceGroupsMapped: report.gates.allSourceGroupsMatched,
    provenanceVerified: report.verified,
    whyReactEvidencePresent: report.matches.every(match => (
      match.whyReacts.formula.includes('GELU')
      && match.whyReacts.topTriggers.length >= 3
      && match.sourceLayer.startsWith('transformer.')
      && match.sourceIds.length >= 1
    )),
    rawOracleCallsBelow120000: model.queryBudget.rawOracleCalls < 120000
  };
  return {
    id: testCase.id,
    pass: Object.values(gates).every(Boolean),
    teacher: {
      dimension: testCase.dimension,
      outputBias: testCase.outputBias,
      sourceNeuronCount: testCase.sources.length,
      expectedSourceGroups: testCase.expectedSourceGroups,
      sources: testCase.sources.map(source => ({
        id: source.id,
        layer: source.layer,
        index: source.index,
        origin: source.origin,
        inputWeightNorm: Math.sqrt(dot(source.inputWeight, source.inputWeight)),
        inputBias: source.inputBias,
        outputWeight: source.outputWeight
      }))
    },
    gcl: {
      effectiveChannelCount: model.effectiveChannelCount,
      rawChannelCount: model.rawChannelCount,
      selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
      metrics,
      agreementMae,
      queryBudget: model.queryBudget
    },
    provenance: report,
    gates
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = [makeDistinctCase(), makeDuplicateSplitCase()].map(runCase);
const pass = rows.every(row => row.pass);
const matchConfidences = rows.flatMap(row => row.provenance.matches.map(match => match.confidence));
const summary = {
  pass,
  caseCount: rows.length,
  passedCases: rows.filter(row => row.pass).length,
  failedCases: rows.filter(row => !row.pass).map(row => row.id),
  maxMae: Math.max(...rows.map(row => row.gcl.metrics.mae)),
  minR2: Math.min(...rows.map(row => row.gcl.metrics.r2)),
  maxAgreementMae: Math.max(...rows.map(row => row.gcl.agreementMae)),
  minConfidence: matchConfidences.length ? Math.min(...matchConfidences) : 0,
  duplicateGroupCases: rows.filter(row => row.provenance.duplicateSourceGroups.length > 0).length,
  maxRawOracleCalls: Math.max(...rows.map(row => row.gcl.queryBudget.rawOracleCalls))
};
const gates = {
  pass,
  allCasesPassed: rows.every(row => row.pass),
  coversDistinctOneToOneMapping: rows.some(row => row.id.includes('distinct') && row.provenance.matches.every(match => match.sourceIds.length === 1)),
  coversDuplicateSplitGroupMapping: rows.some(row => row.id.includes('duplicate') && row.provenance.duplicateSourceGroups.some(group => group.sourceIds.length === 2)),
  allWhyReactEvidencePresent: rows.every(row => row.gates.whyReactEvidencePresent)
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'GELU FFN source-neuron provenance verifier',
  blackBoxBoundary: {
    solverInput: 'oracle f(vector) plus source-neuron manifest for provenance verification',
    note: 'Exact source identity is a provenance-mode claim. Output-only recovery still produces canonical function atoms; source-neuron identity requires source weights, activation hooks, or an equivalent signed provenance manifest.'
  },
  claimScope: {
    provesIfPasses: 'GCL can map recovered GELU atoms to exact source neuron ids/layers/indices for distinguishable source neurons, and to exact source groups for duplicate-split neurons, while recording why each mapped source reacts.',
    doesNotClaim: 'Exact original-neuron identity from output-only access without provenance information, mathematical uniqueness for arbitrary neural networks, full GPT-2 FFN extraction, or arbitrary 3072-channel LLM block decompilation.'
  },
  summary,
  gates,
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# GELU FFN Source-Neuron Provenance Verifier Benchmark',
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
  `- Pass: ${pass}`,
  `- Cases: ${summary.caseCount}`,
  `- Passed cases: ${summary.passedCases}`,
  `- Failed cases: ${summary.failedCases.join(', ') || 'none'}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Max model agreement MAE: ${summary.maxAgreementMae}`,
  `- Min provenance confidence: ${summary.minConfidence}`,
  `- Duplicate-group cases: ${summary.duplicateGroupCases}`,
  '',
  '## Cases',
  '',
  '| Case | Pass | Source neurons | Source groups | Effective channels | MAE | R2 |',
  '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
  ...rows.map(row => `| ${row.id} | ${row.pass} | ${row.teacher.sourceNeuronCount} | ${row.provenance.sourceGroupCount} | ${row.gcl.effectiveChannelCount} | ${row.gcl.metrics.mae} | ${row.gcl.metrics.r2} |`),
  '',
  '## Provenance Matches',
  '',
  ...rows.flatMap(row => [
    `### ${row.id}`,
    '',
    ...row.provenance.matches.flatMap(match => [
      `- recovered atom ${match.recoveredAtomIndex} -> ${match.sourceGroupId}`,
      `  - source ids: ${match.sourceIds.join(', ')}`,
      `  - layer: ${match.sourceLayer}`,
      `  - indices: ${match.sourceIndices.join(', ')}`,
      `  - duplicate split: ${match.duplicateSplit}`,
      `  - confidence: ${match.confidence}`,
      `  - direction cosine: ${match.directionCosine}`,
      `  - contribution correlation: ${match.contributionCorrelation}`,
      `  - contribution MAE: ${match.contributionMae}`,
      `  - why: ${match.whyReacts.responseRule}`,
      `  - formula: \`${match.whyReacts.formula}\``
    ]),
    ''
  ]),
  '## Conclusion',
  '',
  pass
    ? 'The provenance verifier gate passed. GCL can now make source-neuron identity claims only in provenance mode, with exact source ids/groups and why-reacts evidence.'
    : 'The provenance verifier gate failed. Do not claim exact source-neuron provenance until the failing case is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-provenance-verifier',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Provenance-mode source-neuron identity verifier for GELU FFN structure recovery.',
    'Maps recovered atoms to source ids, layers, indices, duplicate source groups, and why-reacts evidence.',
    'Does not claim source identity from output-only access without source provenance.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
