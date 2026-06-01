#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  canonicalizeGeluFfnStructure,
  decompileRankAdaptiveGeluFfnStructure,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type TeacherAtom = {
  direction: number[];
  inputScale: number;
  inputBias: number;
  outputWeight: number;
};

type TeacherCase = {
  id: string;
  profile: 'balanced-noisy' | 'near-parallel-noisy' | 'weak-resolvable-noisy';
  dimension: number;
  atoms: TeacherAtom[];
  outputBias: number;
  noiseStd: number;
  quantization: number;
  querySamples: number;
  randomSeed: number;
  ranges: Array<[number, number]>;
  solver: {
    gradientProbeCount: number;
    trainingSampleCount: number;
    validationSampleCount: number;
    restarts: number;
    iterations: number;
    learningRate: number;
    atomRelativeStrengthThreshold: number;
    atomAbsoluteStrengthThreshold: number;
    pruneValidationToleranceRatio: number;
    pruneValidationToleranceAbsolute: number;
    nearParallelDirectionCosine: number;
    duplicateDirectionCosine: number;
    duplicateBiasTolerance: number;
  };
};

type Predictor = (input: number[]) => number;

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-noisy-mixed-structure',
  'quantized-noisy-family'
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
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
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

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function makeDirections(dimension: number, count: number, seed: number): number[][] {
  const random = makeRandom(seed);
  const directions: number[][] = [];
  for (let index = 0; index < count; index++) {
    directions.push(normalize(subtractProjection(
      Array.from({ length: dimension }, () => gaussian(random)),
      directions
    )));
  }
  return directions;
}

function makeInputs(seed: number, dimension: number, count: number, range: [number, number] = [-1.5, 1.5]): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => range[0] + (range[1] - range[0]) * random())
  ));
}

function predictTeacher(teacher: TeacherCase, input: number[]): number {
  return teacher.outputBias + teacher.atoms.reduce((sum, atom) => (
    sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
  ), 0);
}

function makeNoisyOracle(teacher: TeacherCase, seedOffset: number): Predictor {
  const random = makeRandom(teacher.randomSeed + seedOffset);
  return input => {
    const noisy = predictTeacher(teacher, input) + gaussian(random) * teacher.noiseStd;
    return Math.round(noisy / teacher.quantization) * teacher.quantization;
  };
}

function aggregate(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function queryRepeated(oracle: Predictor, input: number[], count: number): number {
  return aggregate(Array.from({ length: count }, () => oracle(input)));
}

function bestDirectionProjection(trueDirection: number[], recoveredWeights: number[][]): number {
  return Math.max(0, ...recoveredWeights.map(weight => Math.abs(dot(trueDirection, normalize(weight)))));
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function solveSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) < 1e-12) throw new Error('baseline linear system is singular');
    for (let index = column; index <= n; index++) augmented[column][index] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      if (Math.abs(factor) < 1e-20) continue;
      for (let index = column; index <= n; index++) augmented[row][index] -= factor * augmented[column][index];
    }
  }
  return augmented.map(row => row[n]);
}

function trainRidge(
  inputs: number[][],
  targets: number[],
  feature: (input: number[]) => number[],
  ridge: number
): Predictor {
  const width = feature(inputs[0]).length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (let rowIndex = 0; rowIndex < inputs.length; rowIndex++) {
    const row = feature(inputs[rowIndex]);
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * targets[rowIndex];
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let index = 0; index < width; index++) xtx[index][index] += ridge;
  const weights = solveSystem(xtx, xty);
  return input => dot(weights, feature(input));
}

function makeQuadraticFeatures(input: number[]): number[] {
  return [1, ...input, ...input.map(value => value ** 2)];
}

function makeRandomGeluFeature(seed: number, dimension: number, count: number): (input: number[]) => number[] {
  const random = makeRandom(seed);
  const directions = Array.from({ length: count }, () => (
    normalize(Array.from({ length: dimension }, () => gaussian(random)))
  ));
  const biases = Array.from({ length: count }, () => -0.6 + 1.2 * random());
  return input => [
    1,
    ...input,
    ...directions.map((direction, index) => gelu(dot(direction, input) + biases[index]))
  ];
}

function makeTeacherCases(): TeacherCase[] {
  const balancedDirections = makeDirections(24, 3, 300101);
  const nearBase = normalize(Array.from({ length: 16 }, (_, index) => Math.sin(index * 1.7 + 0.2)));
  const nearPerturbation = normalize(subtractProjection(
    Array.from({ length: 16 }, (_, index) => Math.cos(index * 2.1 - 0.4)),
    [nearBase]
  ));
  const nearThird = makeDirections(16, 1, 300202)[0];
  const nearSecond = normalize(nearBase.map((value, index) => value + 0.018 * nearPerturbation[index]));
  const weakDirections = makeDirections(12, 3, 300303);

  return [
    {
      id: 'balanced-three-channel-noisy-24d',
      profile: 'balanced-noisy',
      dimension: 24,
      atoms: [
        { direction: balancedDirections[0], inputScale: 1.18, inputBias: -0.22, outputWeight: 0.82 },
        { direction: balancedDirections[1], inputScale: 0.96, inputBias: 0.18, outputWeight: -0.64 },
        { direction: balancedDirections[2], inputScale: 0.88, inputBias: 0.34, outputWeight: 0.48 }
      ],
      outputBias: 0.04,
      noiseStd: 0.0025,
      quantization: 0.001,
      querySamples: 15,
      randomSeed: 301001,
      ranges: Array.from({ length: 24 }, () => [-1.5, 1.5] as [number, number]),
      solver: {
        gradientProbeCount: 22,
        trainingSampleCount: 1450,
        validationSampleCount: 420,
        restarts: 8,
        iterations: 540,
        learningRate: 0.03,
        atomRelativeStrengthThreshold: 0.025,
        atomAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.22,
        pruneValidationToleranceAbsolute: 0.014,
        nearParallelDirectionCosine: 0.999,
        duplicateDirectionCosine: 0.998,
        duplicateBiasTolerance: 0.12
      }
    },
    {
      id: 'near-parallel-three-channel-noisy-16d',
      profile: 'near-parallel-noisy',
      dimension: 16,
      atoms: [
        { direction: nearBase, inputScale: 1.16, inputBias: -0.36, outputWeight: 0.74 },
        { direction: nearSecond, inputScale: 1.02, inputBias: 0.26, outputWeight: -0.58 },
        { direction: nearThird, inputScale: 0.9, inputBias: 0.28, outputWeight: 0.5 }
      ],
      outputBias: -0.03,
      noiseStd: 0.002,
      quantization: 0.001,
      querySamples: 11,
      randomSeed: 302001,
      ranges: Array.from({ length: 16 }, () => [-1.5, 1.5] as [number, number]),
      solver: {
        gradientProbeCount: 28,
        trainingSampleCount: 1900,
        validationSampleCount: 550,
        restarts: 12,
        iterations: 900,
        learningRate: 0.028,
        atomRelativeStrengthThreshold: 0.02,
        atomAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.45,
        pruneValidationToleranceAbsolute: 0.02,
        nearParallelDirectionCosine: 0.999,
        duplicateDirectionCosine: 0.9994,
        duplicateBiasTolerance: 0.16
      }
    },
    {
      id: 'weak-resolvable-three-channel-noisy-12d',
      profile: 'weak-resolvable-noisy',
      dimension: 12,
      atoms: [
        { direction: weakDirections[0], inputScale: 1.12, inputBias: -0.2, outputWeight: 0.78 },
        { direction: weakDirections[1], inputScale: 0.94, inputBias: 0.24, outputWeight: -0.57 },
        { direction: weakDirections[2], inputScale: 1.08, inputBias: -0.08, outputWeight: 0.09 }
      ],
      outputBias: 0.025,
      noiseStd: 0.003,
      quantization: 0.001,
      querySamples: 13,
      randomSeed: 303001,
      ranges: Array.from({ length: 12 }, () => [-1.5, 1.5] as [number, number]),
      solver: {
        gradientProbeCount: 32,
        trainingSampleCount: 2300,
        validationSampleCount: 680,
        restarts: 14,
        iterations: 1050,
        learningRate: 0.028,
        atomRelativeStrengthThreshold: 0.015,
        atomAbsoluteStrengthThreshold: 0.004,
        pruneValidationToleranceRatio: 0.2,
        pruneValidationToleranceAbsolute: 0.01,
        nearParallelDirectionCosine: 0.999,
        duplicateDirectionCosine: 0.998,
        duplicateBiasTolerance: 0.12
      }
    }
  ];
}

function runBaselines(teacher: TeacherCase, validationInputs: number[][], actual: number[]) {
  const trainInputs = makeInputs(teacher.randomSeed + 401, teacher.dimension, 1250);
  const noisyOracle = makeNoisyOracle(teacher, 402);
  let rawOracleCalls = 0;
  const trainTargets = trainInputs.map(input => {
    rawOracleCalls += teacher.querySamples;
    return queryRepeated(noisyOracle, input, teacher.querySamples);
  });
  const mean = trainTargets.reduce((sum, value) => sum + value, 0) / trainTargets.length;
  const baselines = [
    { id: 'mean-output', family: 'constant', predict: (() => mean) as Predictor, trainingRawOracleCalls: 0 },
    {
      id: 'linear-ridge',
      family: 'linear ridge',
      predict: trainRidge(trainInputs, trainTargets, input => [1, ...input], 1e-4),
      trainingRawOracleCalls: rawOracleCalls
    },
    {
      id: 'quadratic-ridge',
      family: 'linear + squared coordinates ridge',
      predict: trainRidge(trainInputs, trainTargets, makeQuadraticFeatures, 1e-3),
      trainingRawOracleCalls: rawOracleCalls
    },
    {
      id: 'random-gelu-ridge',
      family: 'random GELU feature ridge surrogate',
      predict: trainRidge(
        trainInputs,
        trainTargets,
        makeRandomGeluFeature(teacher.randomSeed + 403, teacher.dimension, 48),
        1e-3
      ),
      trainingRawOracleCalls: rawOracleCalls
    }
  ];
  return baselines.map(baseline => {
    const predicted = validationInputs.map(baseline.predict);
    return {
      id: baseline.id,
      family: baseline.family,
      metrics: regressionMetrics(actual, predicted),
      worstError: worstAbsoluteError(actual, predicted),
      trainingRawOracleCalls: baseline.trainingRawOracleCalls
    };
  });
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = makeTeacherCases().map((teacher, index) => {
  const noisyOracle = makeNoisyOracle(teacher, 101);
  const model = decompileRankAdaptiveGeluFfnStructure(noisyOracle, {
    dimension: teacher.dimension,
    ranges: teacher.ranges,
    maxChannelCount: 4,
    minEffectiveChannelCount: 2,
    selectionRestarts: 1,
    gradientProbeCount: teacher.solver.gradientProbeCount,
    trainingSampleCount: teacher.solver.trainingSampleCount,
    validationSampleCount: teacher.solver.validationSampleCount,
    randomSeed: teacher.randomSeed + 201,
    restarts: teacher.solver.restarts,
    iterations: teacher.solver.iterations,
    learningRate: teacher.solver.learningRate,
    querySamples: teacher.querySamples,
    aggregation: 'median',
    atomRelativeStrengthThreshold: teacher.solver.atomRelativeStrengthThreshold,
    atomAbsoluteStrengthThreshold: teacher.solver.atomAbsoluteStrengthThreshold,
    pruneValidationToleranceRatio: teacher.solver.pruneValidationToleranceRatio,
    pruneValidationToleranceAbsolute: teacher.solver.pruneValidationToleranceAbsolute,
    nearParallelDirectionCosine: teacher.solver.nearParallelDirectionCosine,
    duplicateDirectionCosine: teacher.solver.duplicateDirectionCosine,
    duplicateBiasTolerance: teacher.solver.duplicateBiasTolerance
  });
  const validationInputs = makeInputs(teacher.randomSeed + 501, teacher.dimension, 850);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const worstError = worstAbsoluteError(actual, predicted);
  const signature = canonicalizeGeluFfnStructure(model, {
    atomRelativeStrengthThreshold: teacher.profile === 'weak-resolvable-noisy' ? 0.012 : 0.018,
    atomAbsoluteStrengthThreshold: teacher.profile === 'weak-resolvable-noisy' ? 0.003 : 0.004,
    duplicateDirectionCosine: teacher.solver.duplicateDirectionCosine,
    duplicateBiasTolerance: teacher.solver.duplicateBiasTolerance
  });
  const recoveredWeights = model.atoms.map(atom => atom.inputWeight);
  const projectionScores = teacher.atoms.map(atom => bestDirectionProjection(atom.direction, recoveredWeights));
  const minDirectionProjection = Math.min(...projectionScores);
  const baselines = runBaselines(teacher, validationInputs, actual);
  const bestBaseline = baselines.reduce((best, baseline) => (
    baseline.metrics.mae < best.metrics.mae ? baseline : best
  ), baselines[0]);
  const strongOnlyPredict = (input: number[]): number => teacher.outputBias + teacher.atoms.slice(0, 2).reduce((sum, atom) => (
    sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
  ), 0);
  const strongOnlyMetrics = regressionMetrics(actual, validationInputs.map(strongOnlyPredict));
  const weakSignalResolvable = teacher.profile !== 'weak-resolvable-noisy'
    || strongOnlyMetrics.mae > Math.max(0.015, metrics.mae * 2.25);
  const gates = {
    outputR2AboveThreshold: metrics.r2 > (teacher.profile === 'weak-resolvable-noisy' ? 0.997 : 0.9985),
    outputMaeBelowThreshold: metrics.mae < (teacher.profile === 'weak-resolvable-noisy' ? 0.022 : 0.018),
    worstErrorBelowThreshold: worstError < (teacher.profile === 'weak-resolvable-noisy' ? 0.105 : 0.085),
    effectiveChannelCountMatches: model.effectiveChannelCount === teacher.atoms.length,
    canonicalAtomCountMatches: signature.canonicalAtomCount === teacher.atoms.length,
    directionRecoveryAboveThreshold: minDirectionProjection > (teacher.profile === 'near-parallel-noisy' ? 0.985 : 0.992),
    noDuplicateCollapse: (model.adaptiveSelection?.duplicatePairCount ?? 0) === 0,
    weakResolvableSignalNotPruned: weakSignalResolvable && (teacher.profile !== 'weak-resolvable-noisy' || (model.effectiveChannelCount ?? 0) === 3),
    beatsBestBaselineByMae: metrics.mae < bestBaseline.metrics.mae,
    rawOracleCallsBelowBudget: model.queryBudget.rawOracleCalls < 220000
  };
  return {
    id: teacher.id,
    profile: teacher.profile,
    teacher: {
      dimension: teacher.dimension,
      trueChannelCount: teacher.atoms.length,
      noiseStd: teacher.noiseStd,
      quantization: teacher.quantization,
      querySamples: teacher.querySamples
    },
    summary: {
      pass: Object.values(gates).every(Boolean),
      mae: metrics.mae,
      rmse: metrics.rmse,
      r2: metrics.r2,
      worstError,
      effectiveChannelCount: model.effectiveChannelCount,
      canonicalAtomCount: signature.canonicalAtomCount,
      minDirectionProjection,
      duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
      nearParallelPairCount: model.adaptiveSelection?.nearParallelPairCount ?? 0,
      prunedWeakAtomCount: model.adaptiveSelection?.prunedWeakAtomCount ?? 0,
      bestBaselineId: bestBaseline.id,
      bestBaselineMae: bestBaseline.metrics.mae,
      gclVsBestBaselineMaeRatio: metrics.mae / Math.max(1e-12, bestBaseline.metrics.mae),
      strongOnlyMae: strongOnlyMetrics.mae,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates,
    baselines,
    gcl: {
      metrics,
      worstError,
      projectionScores,
      canonicalSignature: signature,
      adaptiveSelection: model.adaptiveSelection,
      queryBudget: model.queryBudget,
      expression: model.expression,
      atoms: model.atoms.map(atom => ({
        inputWeightNorm: atom.inputWeightNorm,
        inputBias: atom.inputBias,
        outputWeight: atom.outputWeight,
        strength: Math.abs(atom.outputWeight) * atom.inputWeightNorm
      }))
    }
  };
});

const pass = rows.every(row => row.summary.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy quantized mixed GELU FFN structure recovery',
  blackBox: {
    visibilityToGcl: 'oracle calls only: f(vector); true channel count, directions, biases, readout weights, noise seed, and clean target are hidden from the solver',
    oracleNoise: 'stochastic Gaussian output noise plus finite-precision quantization; solver uses repeated median queries'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit mixed GELU atom structure under controlled stochastic noise and quantization, including balanced, near-parallel, and weak-but-resolvable channel profiles, while beating dependency-free surrogate baselines on held-out noiseless truth',
    doesNotClaim: 'full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM block decompilation, noisy SwiGLU mixed-block recovery, semantic original-neuron identity from output-only access, or broad SOTA status'
  },
  config: {
    maxChannelBudgetGivenToGcl: 4,
    caseCount: rows.length
  },
  gates: {
    pass,
    structureRecoveryPass: rows.every(row => (
      row.gates.effectiveChannelCountMatches
      && row.gates.canonicalAtomCountMatches
      && row.gates.directionRecoveryAboveThreshold
      && row.gates.noDuplicateCollapse
      && row.gates.weakResolvableSignalNotPruned
    )),
    accuracyCourtPass: rows.every(row => (
      row.gates.outputR2AboveThreshold
      && row.gates.outputMaeBelowThreshold
      && row.gates.worstErrorBelowThreshold
      && row.gates.beatsBestBaselineByMae
    )),
    allCasesPassed: rows.every(row => row.summary.pass),
    allEffectiveCountsMatch: rows.every(row => row.gates.effectiveChannelCountMatches),
    allCanonicalCountsMatch: rows.every(row => row.gates.canonicalAtomCountMatches),
    allDirectionRecoveryAboveThreshold: rows.every(row => row.gates.directionRecoveryAboveThreshold),
    allBeatBestBaselineByMae: rows.every(row => row.gates.beatsBestBaselineByMae),
    allRawOracleCallsBelowBudget: rows.every(row => row.gates.rawOracleCallsBelowBudget),
    coversBalancedNoise: rows.some(row => row.profile === 'balanced-noisy' && row.summary.pass),
    coversNearParallelNoise: rows.some(row => row.profile === 'near-parallel-noisy' && row.summary.pass),
    coversWeakResolvableNoise: rows.some(row => row.profile === 'weak-resolvable-noisy' && row.summary.pass)
  },
  summary: {
    pass,
    caseCount: rows.length,
    passedCases: rows.filter(row => row.summary.pass).length,
    failedCases: rows.filter(row => !row.summary.pass).map(row => row.id),
    maxMae: Math.max(...rows.map(row => row.summary.mae)),
    minR2: Math.min(...rows.map(row => row.summary.r2)),
    maxWorstError: Math.max(...rows.map(row => row.summary.worstError)),
    minDirectionProjection: Math.min(...rows.map(row => row.summary.minDirectionProjection)),
    effectiveChannelCounts: rows.map(row => row.summary.effectiveChannelCount),
    canonicalAtomCounts: rows.map(row => row.summary.canonicalAtomCount),
    maxGclVsBestBaselineMaeRatio: Math.max(...rows.map(row => row.summary.gclVsBestBaselineMaeRatio)),
    maxRawOracleCalls: Math.max(...rows.map(row => row.summary.rawOracleCalls))
  },
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Noisy Mixed GELU FFN Structure Benchmark',
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
  `- Max MAE: ${report.summary.maxMae}`,
  `- Min R2: ${report.summary.minR2}`,
  `- Max worst error: ${report.summary.maxWorstError}`,
  `- Min direction projection: ${report.summary.minDirectionProjection}`,
  `- Effective channel counts: ${report.summary.effectiveChannelCounts.join(', ')}`,
  `- Canonical atom counts: ${report.summary.canonicalAtomCounts.join(', ')}`,
  `- Max GCL/best-baseline MAE ratio: ${report.summary.maxGclVsBestBaselineMaeRatio}`,
  `- Max raw oracle calls: ${report.summary.maxRawOracleCalls}`,
  '',
  '## Cases',
  '',
  '| Case | Profile | Pass | MAE | R2 | Canonical atoms | Best baseline | GCL/baseline MAE |',
  '| --- | --- | --- | --- | --- | --- | --- | --- |',
  ...rows.map(row => (
    `| ${row.id} | ${row.profile} | ${row.summary.pass} | ${row.summary.mae} | ${row.summary.r2} | ${row.summary.canonicalAtomCount} | ${row.summary.bestBaselineId} | ${row.summary.gclVsBestBaselineMaeRatio} |`
  )),
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
    ? 'The noisy mixed GELU structure gate passed. GCL recovered explicit atom structure under stochastic noise/quantization and beat the dependency-free baseline court on held-out noiseless truth.'
    : 'The noisy mixed GELU structure gate failed. Do not promote noisy mixed GELU recovery until the failing case is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-noisy-mixed-structure',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Release-gated noisy/quantized mixed GELU structure recovery family.',
    'Fails if output fidelity is good but effective/canonical atom structure is wrong.',
    'Includes dependency-free baseline court on held-out noiseless truth.',
    'Keeps full GPT-2 FFN extraction, arbitrary 3072-channel recovery, and broad SOTA out of scope.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
