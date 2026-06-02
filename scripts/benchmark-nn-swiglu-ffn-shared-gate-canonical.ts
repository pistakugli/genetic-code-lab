#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileRankAdaptiveSwigluFfnStructure,
  predictSwigluFfnStructureBatch,
  SwigluFfnStructurePair
} from '../src/solvers/swiglu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type SourceBlock = {
  valueDirection: number[];
  valueScale: number;
  valueBias: number;
  outputWeight: number;
};

type TeacherCase = {
  id: string;
  profile: 'shared-gate-clean-collapse' | 'shared-gate-noisy-cancellation';
  dimension: number;
  gateDirection: number[];
  gateScale: number;
  gateBias: number;
  sourceBlocks: SourceBlock[];
  outputBias: number;
  noiseStd: number;
  quantization: number;
  querySamples: number;
  ranges: Array<[number, number]>;
  randomSeed: number;
  solver: {
    maxPairCount: number;
    activeRank: number;
    gradientProbeCount: number;
    trainingSampleCount: number;
    validationSampleCount: number;
    restarts: number;
    iterations: number;
    learningRate: number;
    pruneValidationToleranceRatio: number;
    pruneValidationToleranceAbsolute: number;
    pairRelativeStrengthThreshold: number;
    pairAbsoluteStrengthThreshold: number;
  };
};

type EvaluationSet = {
  id: 'iid-uniform' | 'wide-ood' | 'shifted-ood';
  kind: 'iid' | 'ood';
  inputs: number[][];
  actual: number[];
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-swiglu-ffn-shared-gate-canonical',
  'canonical-collapse'
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

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(1e-12, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = norm(values);
  return values.map(value => value / length);
}

function subtractProjection(vector: number[], basis: number[][]): number[] {
  let outputVector = [...vector];
  for (const direction of basis) {
    const projection = dot(outputVector, direction);
    outputVector = outputVector.map((value, index) => value - projection * direction[index]);
  }
  return outputVector;
}

function makeDirection(random: () => number, dimension: number, previous: number[][] = []): number[] {
  return normalize(subtractProjection(
    Array.from({ length: dimension }, () => gaussian(random)),
    previous
  ));
}

function silu(value: number): number {
  if (value > 40) return value;
  if (value < -40) return 0;
  return value / (1 + Math.exp(-value));
}

function makeInputs(seed: number, dimension: number, count: number, range: [number, number] = [-1.45, 1.45]): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => range[0] + (range[1] - range[0]) * random())
  ));
}

function makeShiftedOodInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, (_unused, axis) => {
      const shift = axis % 4 === 0 ? 0.42 : axis % 4 === 1 ? -0.31 : axis % 4 === 2 ? 0.18 : -0.14;
      return -1.35 + 2.7 * random() + shift;
    })
  ));
}

function predictCleanTeacher(teacher: TeacherCase, input: number[]): number {
  const gate = teacher.gateScale * dot(teacher.gateDirection, input) + teacher.gateBias;
  return teacher.outputBias + teacher.sourceBlocks.reduce((sum, block) => (
    sum + block.outputWeight * (block.valueScale * dot(block.valueDirection, input) + block.valueBias) * silu(gate)
  ), 0);
}

function makeNoisyOracle(teacher: TeacherCase, seedOffset: number): (input: number[]) => number {
  if (teacher.querySamples === 1) return input => predictCleanTeacher(teacher, input);
  const random = makeRandom(teacher.randomSeed + seedOffset);
  return input => {
    const noisy = predictCleanTeacher(teacher, input) + gaussian(random) * teacher.noiseStd;
    return Math.round(noisy / teacher.quantization) * teacher.quantization;
  };
}

function aggregateCanonicalValue(teacher: TeacherCase): { direction: number[]; weight: number[]; bias: number; cancellationRatio: number } {
  const weight = new Array(teacher.dimension).fill(0);
  let bias = 0;
  let sourceNormSum = 0;
  for (const block of teacher.sourceBlocks) {
    const sourceWeight = block.outputWeight * block.valueScale;
    sourceNormSum += Math.abs(sourceWeight);
    for (let axis = 0; axis < teacher.dimension; axis++) weight[axis] += sourceWeight * block.valueDirection[axis];
    bias += block.outputWeight * block.valueBias;
  }
  const weightNorm = norm(weight);
  return {
    direction: normalize(weight),
    weight,
    bias,
    cancellationRatio: 1 - weightNorm / Math.max(1e-12, sourceNormSum)
  };
}

function makeEvaluationSets(teacher: TeacherCase): EvaluationSet[] {
  const specs: Array<Omit<EvaluationSet, 'actual'>> = [
    {
      id: 'iid-uniform',
      kind: 'iid',
      inputs: makeInputs(teacher.randomSeed + 501, teacher.dimension, 900)
    },
    {
      id: 'wide-ood',
      kind: 'ood',
      inputs: makeInputs(teacher.randomSeed + 502, teacher.dimension, 900, [-1.95, 1.95])
    },
    {
      id: 'shifted-ood',
      kind: 'ood',
      inputs: makeShiftedOodInputs(teacher.randomSeed + 503, teacher.dimension, 900)
    }
  ];
  return specs.map(spec => ({
    ...spec,
    actual: spec.inputs.map(input => predictCleanTeacher(teacher, input))
  }));
}

function directionProjection(trueDirection: number[], recoveredWeight: number[]): number {
  return Math.abs(dot(trueDirection, normalize(recoveredWeight)));
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function countNearDuplicatePairs(pairs: SwigluFfnStructurePair[], cosine: number): number {
  let count = 0;
  for (let left = 0; left < pairs.length; left++) {
    for (let right = left + 1; right < pairs.length; right++) {
      const valueCosine = Math.abs(dot(pairs[left].valueWeight, pairs[right].valueWeight) / Math.max(1e-12, norm(pairs[left].valueWeight) * norm(pairs[right].valueWeight)));
      const gateCosine = Math.abs(dot(pairs[left].gateWeight, pairs[right].gateWeight) / Math.max(1e-12, norm(pairs[left].gateWeight) * norm(pairs[right].gateWeight)));
      if (valueCosine >= cosine && gateCosine >= cosine) count++;
    }
  }
  return count;
}

function makeTeacherCases(): TeacherCase[] {
  const cleanDimension = 14;
  const cleanRandom = makeRandom(5501);
  const cleanGate = makeDirection(cleanRandom, cleanDimension);
  const cleanValue1 = makeDirection(cleanRandom, cleanDimension, [cleanGate]);
  const cleanPerturbation = makeDirection(cleanRandom, cleanDimension, [cleanGate, cleanValue1]);
  const cleanValue2 = normalize(cleanValue1.map((value, index) => -0.92 * value + 0.28 * cleanPerturbation[index]));
  const cleanValue3 = makeDirection(cleanRandom, cleanDimension, [cleanGate, cleanValue1, cleanPerturbation]);

  const noisyDimension = 16;
  const noisyRandom = makeRandom(6501);
  const noisyGate = makeDirection(noisyRandom, noisyDimension);
  const noisyValue1 = makeDirection(noisyRandom, noisyDimension, [noisyGate]);
  const noisyPerturbation = makeDirection(noisyRandom, noisyDimension, [noisyGate, noisyValue1]);
  const noisyValue2 = normalize(noisyValue1.map((value, index) => 0.98 * value + 0.18 * noisyPerturbation[index]));
  const noisyValue3 = normalize(noisyValue1.map((value, index) => 0.38 * value + 0.92 * noisyPerturbation[index]));

  return [
    {
      id: 'shared-gate-three-source-clean-collapse-14d',
      profile: 'shared-gate-clean-collapse',
      dimension: cleanDimension,
      gateDirection: cleanGate,
      gateScale: 1.17,
      gateBias: 0.21,
      sourceBlocks: [
        { valueDirection: cleanValue1, valueScale: 1.1, valueBias: -0.2, outputWeight: 1.0 },
        { valueDirection: cleanValue2, valueScale: 0.95, valueBias: 0.17, outputWeight: -0.82 },
        { valueDirection: cleanValue3, valueScale: 0.55, valueBias: 0.06, outputWeight: 0.18 }
      ],
      outputBias: 0.035,
      noiseStd: 0,
      quantization: 0,
      querySamples: 1,
      ranges: Array.from({ length: cleanDimension }, () => [-1.45, 1.45] as [number, number]),
      randomSeed: 5601,
      solver: {
        maxPairCount: 4,
        activeRank: 4,
        gradientProbeCount: 24,
        trainingSampleCount: 2600,
        validationSampleCount: 520,
        restarts: 12,
        iterations: 820,
        learningRate: 0.024,
        pruneValidationToleranceRatio: 0.18,
        pruneValidationToleranceAbsolute: 0.006,
        pairRelativeStrengthThreshold: 0.02,
        pairAbsoluteStrengthThreshold: 0.002
      }
    },
    {
      id: 'shared-gate-noisy-cancellation-heavy-collapse-16d',
      profile: 'shared-gate-noisy-cancellation',
      dimension: noisyDimension,
      gateDirection: noisyGate,
      gateScale: 1.08,
      gateBias: -0.16,
      sourceBlocks: [
        { valueDirection: noisyValue1, valueScale: 1.05, valueBias: -0.18, outputWeight: 1.0 },
        { valueDirection: noisyValue2, valueScale: 1.0, valueBias: 0.16, outputWeight: -0.92 },
        { valueDirection: noisyValue3, valueScale: 0.62, valueBias: -0.04, outputWeight: 0.25 }
      ],
      outputBias: -0.02,
      noiseStd: 0.0018,
      quantization: 0.00075,
      querySamples: 11,
      ranges: Array.from({ length: noisyDimension }, () => [-1.45, 1.45] as [number, number]),
      randomSeed: 6701,
      solver: {
        maxPairCount: 4,
        activeRank: 4,
        gradientProbeCount: 34,
        trainingSampleCount: 4200,
        validationSampleCount: 800,
        restarts: 16,
        iterations: 1100,
        learningRate: 0.022,
        pruneValidationToleranceRatio: 0.28,
        pruneValidationToleranceAbsolute: 0.012,
        pairRelativeStrengthThreshold: 0.016,
        pairAbsoluteStrengthThreshold: 0.0015
      }
    }
  ];
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = makeTeacherCases().map(teacher => {
  const oracle = makeNoisyOracle(teacher, 101);
  const canonical = aggregateCanonicalValue(teacher);
  const model = decompileRankAdaptiveSwigluFfnStructure(oracle, {
    dimension: teacher.dimension,
    ranges: teacher.ranges,
    maxPairCount: teacher.solver.maxPairCount,
    minEffectivePairCount: 1,
    activeRank: teacher.solver.activeRank,
    gradientProbeCount: teacher.solver.gradientProbeCount,
    trainingSampleCount: teacher.solver.trainingSampleCount,
    validationSampleCount: teacher.solver.validationSampleCount,
    randomSeed: teacher.randomSeed,
    restarts: teacher.solver.restarts,
    iterations: teacher.solver.iterations,
    learningRate: teacher.solver.learningRate,
    querySamples: teacher.querySamples,
    aggregation: 'median',
    selectionRestarts: 1,
    pruneValidationToleranceRatio: teacher.solver.pruneValidationToleranceRatio,
    pruneValidationToleranceAbsolute: teacher.solver.pruneValidationToleranceAbsolute,
    pairRelativeStrengthThreshold: teacher.solver.pairRelativeStrengthThreshold,
    pairAbsoluteStrengthThreshold: teacher.solver.pairAbsoluteStrengthThreshold,
    duplicateDirectionCosine: 0.998
  });

  const evaluationSets = makeEvaluationSets(teacher);
  const evaluations = evaluationSets.map(evaluation => {
    const predicted = predictSwigluFfnStructureBatch(model, evaluation.inputs);
    return {
      id: evaluation.id,
      kind: evaluation.kind,
      metrics: regressionMetrics(evaluation.actual, predicted),
      worstError: worstAbsoluteError(evaluation.actual, predicted)
    };
  });
  const iid = evaluations.find(evaluation => evaluation.id === 'iid-uniform') ?? evaluations[0];
  const pair = model.pairs[0];
  const valueProjection = pair ? directionProjection(canonical.direction, pair.valueWeight) : 0;
  const gateProjection = pair ? directionProjection(teacher.gateDirection, pair.gateWeight) : 0;
  const duplicatePairs = countNearDuplicatePairs(model.rawPairs ?? model.pairs, 0.998);
  const maxDistributionMae = Math.max(...evaluations.map(evaluation => evaluation.metrics.mae));
  const minDistributionR2 = Math.min(...evaluations.map(evaluation => evaluation.metrics.r2));
  const maxDistributionWorstError = Math.max(...evaluations.map(evaluation => evaluation.worstError));
  const gates = {
    recoversRankAdaptiveStructure: model.recoveryKind === 'rank-adaptive-swiglu-ffn-structure',
    sourcePairCountExceedsCanonicalCount: teacher.sourceBlocks.length > 1,
    canonicalEffectivePairCountIsOne: model.effectivePairCount === 1 && model.pairs.length === 1,
    doesNotUseTrueSourcePairCount: model.rawPairCount === teacher.solver.maxPairCount && model.maxPairCount === teacher.solver.maxPairCount,
    canonicalValueDirectionRecovered: valueProjection > (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.995 : 0.999),
    sharedGateDirectionRecovered: gateProjection > (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.995 : 0.999),
    cancellationPressurePresent: teacher.profile !== 'shared-gate-noisy-cancellation' || canonical.cancellationRatio > 0.72,
    iidR2AboveThreshold: iid.metrics.r2 > (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.9995 : 0.9999),
    iidMaeBelowThreshold: iid.metrics.mae < (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.006 : 0.002),
    allDistributionR2AboveThreshold: minDistributionR2 > (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.9992 : 0.9998),
    allDistributionMaeBelowThreshold: maxDistributionMae < (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.012 : 0.004),
    allDistributionWorstErrorBelowThreshold: maxDistributionWorstError < (teacher.profile === 'shared-gate-noisy-cancellation' ? 0.12 : 0.04),
    duplicateRawPairsNotPromoted: duplicatePairs === 0 || (model.effectivePairCount ?? 0) < (model.rawPairs?.length ?? model.pairs.length),
    rawOracleCallsBelowBudget: model.queryBudget.rawOracleCalls < (teacher.profile === 'shared-gate-noisy-cancellation' ? 320000 : 60000)
  };

  return {
    id: teacher.id,
    profile: teacher.profile,
    summary: {
      pass: Object.values(gates).every(Boolean),
      trueSourcePairCount: teacher.sourceBlocks.length,
      canonicalPairCount: 1,
      rawPairBudgetGivenToGcl: teacher.solver.maxPairCount,
      effectivePairCount: model.effectivePairCount,
      rawPairCount: model.rawPairCount,
      prunedWeakPairCount: model.adaptiveSelection?.prunedWeakPairCount ?? 0,
      duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
      canonicalValueDirectionProjection: valueProjection,
      sharedGateDirectionProjection: gateProjection,
      cancellationRatio: canonical.cancellationRatio,
      mae: iid.metrics.mae,
      r2: iid.metrics.r2,
      worstError: iid.worstError,
      maxDistributionMae,
      minDistributionR2,
      maxDistributionWorstError,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    teacher: {
      dimension: teacher.dimension,
      sourcePairCount: teacher.sourceBlocks.length,
      canonicalPairCount: 1,
      noiseStd: teacher.noiseStd,
      quantization: teacher.quantization,
      querySamples: teacher.querySamples
    },
    canonicalTarget: {
      aggregateValueWeightNorm: norm(canonical.weight),
      aggregateValueBias: canonical.bias,
      cancellationRatio: canonical.cancellationRatio,
      sharedGateScale: teacher.gateScale,
      sharedGateBias: teacher.gateBias
    },
    gates,
    evaluations,
    structure: {
      adaptiveSelection: model.adaptiveSelection,
      queryBudget: model.queryBudget,
      expression: model.expression,
      pairs: model.pairs.map(recoveredPair => ({
        valueWeightNorm: recoveredPair.valueWeightNorm,
        valueBias: recoveredPair.valueBias,
        gateWeightNorm: recoveredPair.gateWeightNorm,
        gateBias: recoveredPair.gateBias,
        outputWeight: recoveredPair.outputWeight
      }))
    }
  };
});

const pass = rows.every(row => row.summary.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'shared-gate cancellation-heavy SwiGLU canonical structure recovery',
  blackBox: {
    visibilityToGcl: 'oracle calls only: f(vector); source pair count, shared gate, source value directions, cancellation pattern, clean target, and noise seed are hidden from the solver',
    oracleNoise: 'clean case plus stochastic Gaussian output noise and finite-precision quantization case'
  },
  claimScope: {
    provesIfPasses: 'GCL canonicalizes shared-gate/cancellation-heavy SwiGLU degeneracy into the minimal functionally equivalent value/gate pair instead of promoting duplicate source channels',
    doesNotClaim: 'unique original source-neuron identity from output-only access, exact hidden weights under non-identifiability, full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D transformer block decompilation, or broad SOTA status'
  },
  config: {
    caseCount: rows.length,
    evaluationDistributions: ['iid-uniform', 'wide-ood', 'shifted-ood']
  },
  gates: {
    pass,
    allCasesPassed: rows.every(row => row.summary.pass),
    allCanonicalCountsOne: rows.every(row => row.gates.canonicalEffectivePairCountIsOne),
    allSourcePairCountsCollapsed: rows.every(row => row.gates.sourcePairCountExceedsCanonicalCount),
    allValueDirectionsRecovered: rows.every(row => row.gates.canonicalValueDirectionRecovered),
    allSharedGateDirectionsRecovered: rows.every(row => row.gates.sharedGateDirectionRecovered),
    cancellationPressureCovered: rows.some(row => row.profile === 'shared-gate-noisy-cancellation' && row.gates.cancellationPressurePresent),
    allOutputMetricsPass: rows.every(row => (
      row.gates.iidR2AboveThreshold
      && row.gates.iidMaeBelowThreshold
      && row.gates.allDistributionR2AboveThreshold
      && row.gates.allDistributionMaeBelowThreshold
      && row.gates.allDistributionWorstErrorBelowThreshold
    )),
    noDuplicatePairsPromoted: rows.every(row => row.gates.duplicateRawPairsNotPromoted),
    allRawOracleCallsBelowBudget: rows.every(row => row.gates.rawOracleCallsBelowBudget),
    coversCleanSharedGateCollapse: rows.some(row => row.profile === 'shared-gate-clean-collapse'),
    coversNoisyCancellationCollapse: rows.some(row => row.profile === 'shared-gate-noisy-cancellation')
  },
  summary: {
    pass,
    caseCount: rows.length,
    passedCases: rows.filter(row => row.summary.pass).length,
    failedCases: rows.filter(row => !row.summary.pass).map(row => row.id),
    profiles: rows.map(row => row.profile),
    sourcePairCounts: rows.map(row => row.summary.trueSourcePairCount),
    effectivePairCounts: rows.map(row => row.summary.effectivePairCount),
    minCanonicalValueDirectionProjection: Math.min(...rows.map(row => row.summary.canonicalValueDirectionProjection)),
    minSharedGateDirectionProjection: Math.min(...rows.map(row => row.summary.sharedGateDirectionProjection)),
    minCancellationRatio: Math.min(...rows.map(row => row.summary.cancellationRatio)),
    maxMae: Math.max(...rows.map(row => row.summary.mae)),
    minR2: Math.min(...rows.map(row => row.summary.r2)),
    maxWorstError: Math.max(...rows.map(row => row.summary.worstError)),
    maxDistributionMae: Math.max(...rows.map(row => row.summary.maxDistributionMae)),
    minDistributionR2: Math.min(...rows.map(row => row.summary.minDistributionR2)),
    maxDistributionWorstError: Math.max(...rows.map(row => row.summary.maxDistributionWorstError)),
    maxRawOracleCalls: Math.max(...rows.map(row => row.summary.rawOracleCalls))
  },
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Shared-Gate Cancellation SwiGLU Canonical Benchmark',
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
  `- Cases: ${report.summary.caseCount}`,
  `- Passed cases: ${report.summary.passedCases}`,
  `- Failed cases: ${report.summary.failedCases.join(', ') || 'none'}`,
  `- Profiles: ${report.summary.profiles.join(', ')}`,
  `- Source pair counts: ${report.summary.sourcePairCounts.join(', ')}`,
  `- Effective canonical pair counts: ${report.summary.effectivePairCounts.join(', ')}`,
  `- Min canonical value-direction projection: ${report.summary.minCanonicalValueDirectionProjection}`,
  `- Min shared-gate direction projection: ${report.summary.minSharedGateDirectionProjection}`,
  `- Min cancellation ratio: ${report.summary.minCancellationRatio}`,
  `- Max MAE: ${report.summary.maxMae}`,
  `- Min R2: ${report.summary.minR2}`,
  `- Max worst error: ${report.summary.maxWorstError}`,
  `- Max distribution MAE: ${report.summary.maxDistributionMae}`,
  `- Min distribution R2: ${report.summary.minDistributionR2}`,
  `- Max distribution worst error: ${report.summary.maxDistributionWorstError}`,
  `- Max raw oracle calls: ${report.summary.maxRawOracleCalls}`,
  '',
  '## Cases',
  '',
  '| Case | Profile | Pass | Source pairs | Effective pairs | Cancellation | MAE | R2 | Value projection | Gate projection | Raw calls |',
  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
  ...rows.map(row => (
    `| ${row.id} | ${row.profile} | ${row.summary.pass} | ${row.summary.trueSourcePairCount} | ${row.summary.effectivePairCount} | ${row.summary.cancellationRatio} | ${row.summary.mae} | ${row.summary.r2} | ${row.summary.canonicalValueDirectionProjection} | ${row.summary.sharedGateDirectionProjection} | ${row.summary.rawOracleCalls} |`
  )),
  '',
  '## Distribution Holdout',
  '',
  '| Case | Distribution | Kind | MAE | R2 | Worst error |',
  '| --- | --- | --- | --- | --- | --- |',
  ...rows.flatMap(row => row.evaluations.map(item => (
    `| ${row.id} | ${item.id} | ${item.kind} | ${item.metrics.mae} | ${item.metrics.r2} | ${item.worstError} |`
  ))),
  '',
  '## Gate Details',
  '',
  ...rows.flatMap(row => [
    `### ${row.id}`,
    '',
    `- Summary: \`${JSON.stringify(row.summary)}\``,
    ...Object.entries(row.gates).map(([key, value]) => `- ${key}: ${value}`),
    ''
  ]),
  '## Conclusion',
  '',
  pass
    ? 'The shared-gate/cancellation-heavy SwiGLU hard-case passed. GCL collapsed multiple source channels into the minimal canonical value/gate structure and did not promote duplicate non-identifiable source channels.'
    : 'The shared-gate/cancellation-heavy SwiGLU hard-case failed. Do not promote canonical shared-gate recovery until the recorded failure is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-swiglu-ffn-shared-gate-canonical',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Release-gated shared-gate/cancellation-heavy SwiGLU canonicalization.',
    'Multiple source SwiGLU value channels share one gate and must collapse to one canonical effective pair.',
    'Fails if prediction is good but canonical value/gate directions or effective pair count are wrong.',
    'Keeps source-neuron identity without provenance, full GPT-2 extraction, and broad SOTA out of scope.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
