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

type TeacherBlock = {
  valueDirection: number[];
  valueScale: number;
  valueBias: number;
  gateDirection: number[];
  gateScale: number;
  gateBias: number;
  outputWeight: number;
};

type TeacherCase = {
  id: string;
  profile: 'balanced-noisy' | 'near-parallel-values-noisy' | 'weak-resolvable-noisy';
  dimension: number;
  blocks: TeacherBlock[];
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
    pairRelativeStrengthThreshold: number;
    pairAbsoluteStrengthThreshold: number;
    pruneValidationToleranceRatio: number;
    pruneValidationToleranceAbsolute: number;
    duplicateDirectionCosine: number;
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
  'nn-swiglu-ffn-noisy-mixed-structure',
  'quantized-adversarial-family'
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
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
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
      const shift = axis % 4 === 0 ? 0.45 : axis % 4 === 1 ? -0.35 : axis % 4 === 2 ? 0.2 : -0.15;
      return -1.35 + 2.7 * random() + shift;
    })
  ));
}

function predictTeacher(teacher: TeacherCase, input: number[]): number {
  return teacher.outputBias + teacher.blocks.reduce((sum, block) => {
    const value = block.valueScale * dot(block.valueDirection, input) + block.valueBias;
    const gate = block.gateScale * dot(block.gateDirection, input) + block.gateBias;
    return sum + block.outputWeight * value * silu(gate);
  }, 0);
}

function makeNoisyOracle(teacher: TeacherCase, seedOffset: number): (input: number[]) => number {
  const random = makeRandom(teacher.randomSeed + seedOffset);
  return input => {
    const noisy = predictTeacher(teacher, input) + gaussian(random) * teacher.noiseStd;
    return Math.round(noisy / teacher.quantization) * teacher.quantization;
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
    actual: spec.inputs.map(input => predictTeacher(teacher, input))
  }));
}

function directionProjection(trueDirection: number[], recoveredWeight: number[]): number {
  return Math.abs(dot(trueDirection, normalize(recoveredWeight)));
}

function bestAssignment(
  blocks: TeacherBlock[],
  pairs: SwigluFfnStructurePair[],
  row = 0,
  used: boolean[] = new Array(pairs.length).fill(false),
  selected: Array<{ trueIndex: number; recoveredIndex: number; valueProjection: number; gateProjection: number; score: number }> = []
): { score: number; selected: Array<{ trueIndex: number; recoveredIndex: number; valueProjection: number; gateProjection: number; score: number }> } {
  if (row >= blocks.length) {
    return {
      score: selected.reduce((sum, item) => sum + item.score, 0),
      selected: [...selected]
    };
  }
  let best = { score: -Infinity, selected: [] as Array<{ trueIndex: number; recoveredIndex: number; valueProjection: number; gateProjection: number; score: number }> };
  for (let col = 0; col < pairs.length; col++) {
    if (used[col]) continue;
    const valueProjection = directionProjection(blocks[row].valueDirection, pairs[col].valueWeight);
    const gateProjection = directionProjection(blocks[row].gateDirection, pairs[col].gateWeight);
    const score = (valueProjection + gateProjection) / 2;
    used[col] = true;
    selected.push({ trueIndex: row, recoveredIndex: col, valueProjection, gateProjection, score });
    const candidate = bestAssignment(blocks, pairs, row + 1, used, selected);
    if (candidate.score > best.score) best = candidate;
    selected.pop();
    used[col] = false;
  }
  return best;
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function countNearParallelPairs(pairs: SwigluFfnStructurePair[], field: 'valueWeight' | 'gateWeight', cosine: number): number {
  let count = 0;
  for (let left = 0; left < pairs.length; left++) {
    for (let right = left + 1; right < pairs.length; right++) {
      const leftWeight = pairs[left][field];
      const rightWeight = pairs[right][field];
      const similarity = Math.abs(dot(leftWeight, rightWeight) / Math.max(1e-12, norm(leftWeight) * norm(rightWeight)));
      if (similarity >= cosine) count++;
    }
  }
  return count;
}

function makeTeacherCases(): TeacherCase[] {
  const balancedRandom = makeRandom(410101);
  const balancedDirections: number[][] = [];
  for (let index = 0; index < 4; index++) balancedDirections.push(makeDirection(balancedRandom, 24, balancedDirections));

  const nearRandom = makeRandom(420202);
  const valueBase = makeDirection(nearRandom, 18);
  const valuePerturbation = makeDirection(nearRandom, 18, [valueBase]);
  const nearValueDirections = [
    valueBase,
    normalize(valueBase.map((value, index) => value + 0.08 * valuePerturbation[index]))
  ];
  const nearGateDirections: number[][] = [];
  for (let index = 0; index < 2; index++) {
    nearGateDirections.push(makeDirection(nearRandom, 18, [...nearValueDirections, ...nearGateDirections]));
  }

  const weakRandom = makeRandom(430303);
  const weakDirections: number[][] = [];
  for (let index = 0; index < 6; index++) weakDirections.push(makeDirection(weakRandom, 16, weakDirections));

  return [
    {
      id: 'balanced-two-pair-noisy-24d',
      profile: 'balanced-noisy',
      dimension: 24,
      blocks: [
        {
          valueDirection: balancedDirections[0],
          valueScale: 1.04,
          valueBias: -0.16,
          gateDirection: balancedDirections[1],
          gateScale: 1.22,
          gateBias: 0.2,
          outputWeight: 1.08
        },
        {
          valueDirection: balancedDirections[2],
          valueScale: 0.86,
          valueBias: 0.18,
          gateDirection: balancedDirections[3],
          gateScale: 1.05,
          gateBias: -0.18,
          outputWeight: -0.82
        }
      ],
      outputBias: 0.035,
      noiseStd: 0.002,
      quantization: 0.00075,
      querySamples: 11,
      ranges: Array.from({ length: 24 }, () => [-1.45, 1.45] as [number, number]),
      randomSeed: 411001,
      solver: {
        maxPairCount: 3,
        activeRank: 4,
        gradientProbeCount: 26,
        trainingSampleCount: 3000,
        validationSampleCount: 560,
        restarts: 12,
        iterations: 820,
        learningRate: 0.024,
        pairRelativeStrengthThreshold: 0.028,
        pairAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.2,
        pruneValidationToleranceAbsolute: 0.01,
        duplicateDirectionCosine: 0.998
      }
    },
    {
      id: 'near-parallel-values-two-pair-noisy-18d',
      profile: 'near-parallel-values-noisy',
      dimension: 18,
      blocks: [
        {
          valueDirection: nearValueDirections[0],
          valueScale: 1.08,
          valueBias: -0.28,
          gateDirection: nearGateDirections[0],
          gateScale: 1.12,
          gateBias: 0.24,
          outputWeight: 0.92
        },
        {
          valueDirection: nearValueDirections[1],
          valueScale: 0.94,
          valueBias: 0.22,
          gateDirection: nearGateDirections[1],
          gateScale: 1.04,
          gateBias: -0.2,
          outputWeight: -0.68
        }
      ],
      outputBias: -0.025,
      noiseStd: 0.0025,
      quantization: 0.001,
      querySamples: 13,
      ranges: Array.from({ length: 18 }, () => [-1.45, 1.45] as [number, number]),
      randomSeed: 422001,
      solver: {
        maxPairCount: 3,
        activeRank: 6,
        gradientProbeCount: 34,
        trainingSampleCount: 4200,
        validationSampleCount: 700,
        restarts: 18,
        iterations: 1100,
        learningRate: 0.023,
        pairRelativeStrengthThreshold: 0.022,
        pairAbsoluteStrengthThreshold: 0.0035,
        pruneValidationToleranceRatio: 0.38,
        pruneValidationToleranceAbsolute: 0.018,
        duplicateDirectionCosine: 0.9995
      }
    },
    {
      id: 'weak-resolvable-three-pair-noisy-16d',
      profile: 'weak-resolvable-noisy',
      dimension: 16,
      blocks: [
        {
          valueDirection: weakDirections[0],
          valueScale: 1.05,
          valueBias: -0.2,
          gateDirection: weakDirections[1],
          gateScale: 1.12,
          gateBias: 0.18,
          outputWeight: 0.9
        },
        {
          valueDirection: weakDirections[2],
          valueScale: 0.92,
          valueBias: 0.2,
          gateDirection: weakDirections[3],
          gateScale: 0.98,
          gateBias: -0.16,
          outputWeight: -0.66
        },
        {
          valueDirection: weakDirections[4],
          valueScale: 0.82,
          valueBias: -0.08,
          gateDirection: weakDirections[5],
          gateScale: 0.9,
          gateBias: 0.1,
          outputWeight: 0.24
        }
      ],
      outputBias: 0.02,
      noiseStd: 0.0025,
      quantization: 0.001,
      querySamples: 15,
      ranges: Array.from({ length: 16 }, () => [-1.45, 1.45] as [number, number]),
      randomSeed: 433001,
      solver: {
        maxPairCount: 4,
        activeRank: 8,
        gradientProbeCount: 38,
        trainingSampleCount: 3600,
        validationSampleCount: 760,
        restarts: 14,
        iterations: 980,
        learningRate: 0.025,
        pairRelativeStrengthThreshold: 0.014,
        pairAbsoluteStrengthThreshold: 0.0025,
        pruneValidationToleranceRatio: 0.22,
        pruneValidationToleranceAbsolute: 0.012,
        duplicateDirectionCosine: 0.998
      }
    }
  ];
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = makeTeacherCases().map(teacher => {
  const noisyOracle = makeNoisyOracle(teacher, 101);
  const model = decompileRankAdaptiveSwigluFfnStructure(noisyOracle, {
    dimension: teacher.dimension,
    ranges: teacher.ranges,
    maxPairCount: teacher.solver.maxPairCount,
    minEffectivePairCount: 1,
    activeRank: teacher.solver.activeRank,
    gradientProbeCount: teacher.solver.gradientProbeCount,
    trainingSampleCount: teacher.solver.trainingSampleCount,
    validationSampleCount: teacher.solver.validationSampleCount,
    randomSeed: teacher.randomSeed + 201,
    restarts: teacher.solver.restarts,
    iterations: teacher.solver.iterations,
    learningRate: teacher.solver.learningRate,
    querySamples: teacher.querySamples,
    aggregation: 'median',
    selectionRestarts: 1,
    pairRelativeStrengthThreshold: teacher.solver.pairRelativeStrengthThreshold,
    pairAbsoluteStrengthThreshold: teacher.solver.pairAbsoluteStrengthThreshold,
    pruneValidationToleranceRatio: teacher.solver.pruneValidationToleranceRatio,
    pruneValidationToleranceAbsolute: teacher.solver.pruneValidationToleranceAbsolute,
    duplicateDirectionCosine: teacher.solver.duplicateDirectionCosine
  });

  const evaluationSets = makeEvaluationSets(teacher);
  const iidEvaluation = evaluationSets.find(evaluation => evaluation.id === 'iid-uniform') ?? evaluationSets[0];
  const iidPredicted = predictSwigluFfnStructureBatch(model, iidEvaluation.inputs);
  const iidMetrics = regressionMetrics(iidEvaluation.actual, iidPredicted);
  const iidWorstError = worstAbsoluteError(iidEvaluation.actual, iidPredicted);
  const evaluations = evaluationSets.map(evaluation => {
    const predicted = predictSwigluFfnStructureBatch(model, evaluation.inputs);
    return {
      id: evaluation.id,
      kind: evaluation.kind,
      metrics: regressionMetrics(evaluation.actual, predicted),
      worstError: worstAbsoluteError(evaluation.actual, predicted)
    };
  });
  const assignment = bestAssignment(teacher.blocks, model.pairs);
  const minValueDirectionProjection = Math.min(...assignment.selected.map(item => item.valueProjection));
  const minGateDirectionProjection = Math.min(...assignment.selected.map(item => item.gateProjection));
  const meanPairProjection = assignment.selected.reduce((sum, item) => sum + item.score, 0) / Math.max(1, assignment.selected.length);
  const recoveredNearParallelGatePairs = countNearParallelPairs(model.pairs, 'gateWeight', 0.99);
  const recoveredNearParallelValuePairs = countNearParallelPairs(model.pairs, 'valueWeight', 0.99);
  const strongOnlyPredict = (input: number[]): number => teacher.outputBias + teacher.blocks.slice(0, Math.max(1, teacher.blocks.length - 1)).reduce((sum, block) => {
    const value = block.valueScale * dot(block.valueDirection, input) + block.valueBias;
    const gate = block.gateScale * dot(block.gateDirection, input) + block.gateBias;
    return sum + block.outputWeight * value * silu(gate);
  }, 0);
  const strongOnlyMetrics = regressionMetrics(iidEvaluation.actual, iidEvaluation.inputs.map(strongOnlyPredict));
  const maxDistributionMae = Math.max(...evaluations.map(evaluation => evaluation.metrics.mae));
  const minDistributionR2 = Math.min(...evaluations.map(evaluation => evaluation.metrics.r2));
  const maxDistributionWorstError = Math.max(...evaluations.map(evaluation => evaluation.worstError));
  const gates = {
    recoversRankAdaptiveStructure: model.recoveryKind === 'rank-adaptive-swiglu-ffn-structure',
    doesNotUseTruePairCount: model.rawPairCount === teacher.solver.maxPairCount && model.maxPairCount === teacher.solver.maxPairCount,
    effectivePairCountMatches: model.effectivePairCount === teacher.blocks.length,
    duplicatePairsAbsent: (model.adaptiveSelection?.duplicatePairCount ?? 0) === 0,
    valueDirectionRecoveryAboveThreshold: minValueDirectionProjection > (teacher.profile === 'weak-resolvable-noisy' ? 0.982 : 0.99),
    gateDirectionRecoveryAboveThreshold: minGateDirectionProjection > (teacher.profile === 'weak-resolvable-noisy' ? 0.982 : 0.99),
    meanPairProjectionAboveThreshold: meanPairProjection > (teacher.profile === 'weak-resolvable-noisy' ? 0.988 : 0.993),
    iidR2AboveThreshold: iidMetrics.r2 > (teacher.profile === 'weak-resolvable-noisy' ? 0.994 : 0.996),
    iidMaeBelowThreshold: iidMetrics.mae < (teacher.profile === 'weak-resolvable-noisy' ? 0.035 : 0.026),
    iidWorstErrorBelowThreshold: iidWorstError < (teacher.profile === 'weak-resolvable-noisy' ? 0.18 : 0.14),
    allDistributionsR2AboveThreshold: minDistributionR2 > (teacher.profile === 'weak-resolvable-noisy' ? 0.988 : 0.992),
    allDistributionsMaeBelowThreshold: maxDistributionMae < (teacher.profile === 'weak-resolvable-noisy' ? 0.055 : 0.04),
    allDistributionsWorstErrorBelowThreshold: maxDistributionWorstError < (teacher.profile === 'weak-resolvable-noisy' ? 0.3 : 0.22),
    nearParallelValueDiagnosticPresent: teacher.profile !== 'near-parallel-values-noisy' || recoveredNearParallelValuePairs >= 1,
    weakResolvableSignalNotPruned: teacher.profile !== 'weak-resolvable-noisy'
      || ((model.effectivePairCount ?? 0) === 3 && iidMetrics.mae < strongOnlyMetrics.mae * 0.7),
    rawOracleCallsBelowBudget: model.queryBudget.rawOracleCalls < (teacher.profile === 'weak-resolvable-noisy' ? 520000 : 330000)
  };
  return {
    id: teacher.id,
    profile: teacher.profile,
    teacher: {
      dimension: teacher.dimension,
      truePairCount: teacher.blocks.length,
      maxPairBudgetGivenToGcl: teacher.solver.maxPairCount,
      noiseStd: teacher.noiseStd,
      quantization: teacher.quantization,
      querySamples: teacher.querySamples
    },
    summary: {
      pass: Object.values(gates).every(Boolean),
      mae: iidMetrics.mae,
      rmse: iidMetrics.rmse,
      r2: iidMetrics.r2,
      worstError: iidWorstError,
      effectivePairCount: model.effectivePairCount,
      rawPairCount: model.rawPairCount,
      prunedWeakPairCount: model.adaptiveSelection?.prunedWeakPairCount ?? 0,
      duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
      minValueDirectionProjection,
      minGateDirectionProjection,
      meanPairProjection,
      recoveredNearParallelGatePairs,
      recoveredNearParallelValuePairs,
      strongOnlyMae: strongOnlyMetrics.mae,
      maxDistributionMae,
      minDistributionR2,
      maxDistributionWorstError,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates,
    evaluations,
    structure: {
      assignment: assignment.selected,
      adaptiveSelection: model.adaptiveSelection,
      queryBudget: model.queryBudget,
      expression: model.expression,
      pairs: model.pairs.map(pair => ({
        valueWeightNorm: pair.valueWeightNorm,
        valueBias: pair.valueBias,
        gateWeightNorm: pair.gateWeightNorm,
        gateBias: pair.gateBias,
        outputWeight: pair.outputWeight,
        strength: Math.abs(pair.outputWeight) * pair.valueWeightNorm * pair.gateWeightNorm
      }))
    }
  };
});

const pass = rows.every(row => row.summary.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy adversarial mixed-block SwiGLU FFN structure recovery',
  blackBox: {
    visibilityToGcl: 'oracle calls only: f(vector); true pair count, value/gate directions, biases, readout weights, noise seed, quantization, and clean target are hidden from the solver',
    oracleNoise: 'stochastic Gaussian output noise plus finite-precision quantization; solver uses repeated median queries'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit mixed SwiGLU value/gate pair structure under controlled stochastic noise and quantization, including balanced, near-parallel value, and weak-but-resolvable pair profiles',
    doesNotClaim: 'full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production transformer block decompilation, semantic original-neuron identity from output-only access, exact hidden weights under non-identifiability, XGBoost/RF/MLP structure-decompiler superiority, or broad SOTA status'
  },
  config: {
    caseCount: rows.length,
    evaluationDistributions: ['iid-uniform', 'wide-ood', 'shifted-ood']
  },
  gates: {
    pass,
    allCasesPassed: rows.every(row => row.summary.pass),
    allEffectivePairCountsMatch: rows.every(row => row.gates.effectivePairCountMatches),
    allValueDirectionsRecovered: rows.every(row => row.gates.valueDirectionRecoveryAboveThreshold),
    allGateDirectionsRecovered: rows.every(row => row.gates.gateDirectionRecoveryAboveThreshold),
    allOutputMetricsPass: rows.every(row => (
      row.gates.iidR2AboveThreshold
      && row.gates.iidMaeBelowThreshold
      && row.gates.allDistributionsR2AboveThreshold
      && row.gates.allDistributionsMaeBelowThreshold
      && row.gates.allDistributionsWorstErrorBelowThreshold
    )),
    allDuplicatePairsAbsent: rows.every(row => row.gates.duplicatePairsAbsent),
    nearParallelValueCaseCovered: rows.some(row => row.profile === 'near-parallel-values-noisy' && row.gates.nearParallelValueDiagnosticPresent),
    weakResolvableCaseCovered: rows.some(row => row.profile === 'weak-resolvable-noisy' && row.gates.weakResolvableSignalNotPruned),
    allRawOracleCallsBelowBudget: rows.every(row => row.gates.rawOracleCallsBelowBudget),
    coversIidUniformDistribution: rows.every(row => row.evaluations.some(item => item.id === 'iid-uniform')),
    coversWideOodDistribution: rows.every(row => row.evaluations.some(item => item.id === 'wide-ood')),
    coversShiftedOodDistribution: rows.every(row => row.evaluations.some(item => item.id === 'shifted-ood'))
  },
  summary: {
    pass,
    caseCount: rows.length,
    passedCases: rows.filter(row => row.summary.pass).length,
    failedCases: rows.filter(row => !row.summary.pass).map(row => row.id),
    profiles: rows.map(row => row.profile),
    effectivePairCounts: rows.map(row => row.summary.effectivePairCount),
    maxMae: Math.max(...rows.map(row => row.summary.mae)),
    minR2: Math.min(...rows.map(row => row.summary.r2)),
    maxWorstError: Math.max(...rows.map(row => row.summary.worstError)),
    minValueDirectionProjection: Math.min(...rows.map(row => row.summary.minValueDirectionProjection)),
    minGateDirectionProjection: Math.min(...rows.map(row => row.summary.minGateDirectionProjection)),
    minMeanPairProjection: Math.min(...rows.map(row => row.summary.meanPairProjection)),
    maxDistributionMae: Math.max(...rows.map(row => row.summary.maxDistributionMae)),
    minDistributionR2: Math.min(...rows.map(row => row.summary.minDistributionR2)),
    maxDistributionWorstError: Math.max(...rows.map(row => row.summary.maxDistributionWorstError)),
    maxRawOracleCalls: Math.max(...rows.map(row => row.summary.rawOracleCalls))
  },
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Noisy Mixed SwiGLU FFN Structure Benchmark',
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
  `- Effective pair counts: ${report.summary.effectivePairCounts.join(', ')}`,
  `- Max MAE: ${report.summary.maxMae}`,
  `- Min R2: ${report.summary.minR2}`,
  `- Max worst error: ${report.summary.maxWorstError}`,
  `- Min value-direction projection: ${report.summary.minValueDirectionProjection}`,
  `- Min gate-direction projection: ${report.summary.minGateDirectionProjection}`,
  `- Min mean pair projection: ${report.summary.minMeanPairProjection}`,
  `- Max distribution MAE: ${report.summary.maxDistributionMae}`,
  `- Min distribution R2: ${report.summary.minDistributionR2}`,
  `- Max distribution worst error: ${report.summary.maxDistributionWorstError}`,
  `- Max raw oracle calls: ${report.summary.maxRawOracleCalls}`,
  '',
  '## Cases',
  '',
  '| Case | Profile | Pass | MAE | R2 | Pairs | Min value projection | Min gate projection | Raw calls |',
  '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
  ...rows.map(row => (
    `| ${row.id} | ${row.profile} | ${row.summary.pass} | ${row.summary.mae} | ${row.summary.r2} | ${row.summary.effectivePairCount} | ${row.summary.minValueDirectionProjection} | ${row.summary.minGateDirectionProjection} | ${row.summary.rawOracleCalls} |`
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
    `- Profile: ${row.profile}`,
    `- Noise std: ${row.teacher.noiseStd}`,
    `- Quantization: ${row.teacher.quantization}`,
    `- Query samples: ${row.teacher.querySamples}`,
    `- Summary: \`${JSON.stringify(row.summary)}\``,
    ...Object.entries(row.gates).map(([key, value]) => `- ${key}: ${value}`),
    ''
  ]),
  '## Conclusion',
  '',
  pass
    ? 'The noisy mixed SwiGLU structure gate passed. GCL recovered explicit value/gate pairs and effective pair counts from repeated noisy/quantized oracle calls across balanced, near-parallel-value, and weak-resolvable profiles.'
    : 'The noisy mixed SwiGLU structure gate failed. Do not promote noisy mixed SwiGLU recovery until the recorded failure is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-swiglu-ffn-noisy-mixed-structure',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Release-gated noisy/quantized mixed SwiGLU structure recovery family.',
    'Fails if output fidelity is good but effective pair structure or value/gate directions are wrong.',
    'Includes balanced, near-parallel-value, and weak-but-resolvable profiles.',
    'Includes IID, wide-OOD, and shifted-OOD held-out clean-truth evaluations.',
    'Keeps full GPT-2 FFN extraction, arbitrary 3072-channel recovery, semantic original-neuron identity, and broad SOTA out of scope.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
