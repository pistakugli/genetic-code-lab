#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerBatch
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type Profile = 'near-parallel' | 'cancellation' | 'low-amplitude' | 'mixed-oblique';
type Unit = { w: number[]; b: number };
type Teacher = {
  id: string;
  seed: number;
  dimension: number;
  profile: Profile;
  l1: Unit[];
  l2: Unit[];
  out: { w: number[]; b: number; base: number[] };
};
type AttemptConfig = {
  label: string;
  predictionMode: 'region-vote' | 'local-affine-knn';
  sampleCount: number;
  querySamples: number;
  gradientStep: number;
  slopeTolerance: number;
  interceptTolerance: number;
  minRegionSupport: number;
  neighborCount: number;
  localAffineNeighborCount: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler',
  '3d-4d-adaptive-search'
);

function relu(value: number): number {
  return Math.max(0, value);
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
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
  return Math.sqrt(dot(values, values));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function makeTeacher(seed: number, dimension: number, profile: Profile): Teacher {
  const random = makeRandom(seed);
  const directions: number[][] = [];
  if (profile === 'near-parallel') {
    const base = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
    for (let index = 0; index < 4; index++) {
      const perturbation = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
        .map(value => value * (0.04 + index * 0.035));
      directions.push(normalize(base.map((value, axis) => value + perturbation[axis])));
    }
  } else {
    for (let index = 0; index < 4; index++) {
      directions.push(normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1))));
    }
  }

  const l1 = directions.map(direction => ({
    w: direction.map(value => value * randomBetween(
      random,
      profile === 'low-amplitude' ? 0.45 : 0.65,
      profile === 'low-amplitude' ? 0.9 : 1.45
    )),
    b: randomBetween(random, -0.75, 0.75)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      const amplitude = profile === 'cancellation'
        ? randomBetween(random, 0.45, 1.05)
        : randomBetween(random, 0.2, 1.0);
      return sign * amplitude;
    }),
    b: randomBetween(random, -0.55, 0.55)
  }));
  const outputScale = profile === 'low-amplitude' ? 0.65 : 1;
  return {
    id: `${profile}-${dimension}d-${seed}`,
    seed,
    dimension,
    profile,
    l1,
    l2,
    out: {
      w: Array.from({ length: 4 }, (_, index) => {
        const sign = index % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.3, 1.2) * outputScale;
      }),
      b: randomBetween(random, -0.1, 0.1),
      base: Array.from({ length: dimension }, () => randomBetween(random, -0.06, 0.06))
    }
  };
}

function predictTeacher(teacher: Teacher, input: number[]): number {
  const h1 = teacher.l1.map(unit => relu(dot(unit.w, input) + unit.b));
  const h2 = teacher.l2.map(unit => relu(dot(unit.w, h1) + unit.b));
  return teacher.out.b + dot(teacher.out.base, input) + dot(teacher.out.w, h2);
}

function makeNoisyOracle(
  teacher: Teacher,
  seed: number,
  noiseStd: number,
  quantization: number
): (input: number[]) => number {
  const random = makeRandom(seed);
  return (input: number[]) => {
    const noisy = predictTeacher(teacher, input) + gaussian(random) * noiseStd;
    return Math.round(noisy / quantization) * quantization;
  };
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => randomBetween(random, -1.75, 1.75))
  ));
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function aggregateNoisy(
  oracle: (input: number[]) => number,
  input: number[],
  querySamples: number
): number {
  return median(Array.from({ length: querySamples }, () => oracle(input)));
}

function solveSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, index) => [...row, rhs[index]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    if (Math.abs(divisor) < 1e-12) throw new Error('baseline linear system is singular');
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      if (Math.abs(factor) < 1e-20) continue;
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[n]);
}

function fitLinearBaseline(inputs: number[][], targets: number[]): number[] {
  const dimension = inputs[0].length;
  const width = dimension + 1;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (let sampleIndex = 0; sampleIndex < inputs.length; sampleIndex++) {
    const row = [1, ...inputs[sampleIndex]];
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * targets[sampleIndex];
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += 1e-10;
  return solveSystem(xtx, xty);
}

function predictLinear(coefficients: number[], input: number[]): number {
  return coefficients[0] + dot(coefficients.slice(1), input);
}

function attemptConfigs(dimension: number): AttemptConfig[] {
  return [
    {
      label: 'region-vote-screen',
      predictionMode: 'region-vote',
      sampleCount: dimension === 3 ? 4200 : 6200,
      querySamples: 13,
      gradientStep: 0.08,
      slopeTolerance: 0.18,
      interceptTolerance: 0.36,
      minRegionSupport: 8,
      neighborCount: dimension === 3 ? 13 : 17,
      localAffineNeighborCount: dimension === 3 ? 48 : 72
    },
    {
      label: 'local-affine-screen',
      predictionMode: 'local-affine-knn',
      sampleCount: dimension === 3 ? 4200 : 6200,
      querySamples: 13,
      gradientStep: 0.08,
      slopeTolerance: 0.18,
      interceptTolerance: 0.36,
      minRegionSupport: 8,
      neighborCount: dimension === 3 ? 13 : 17,
      localAffineNeighborCount: dimension === 3 ? 48 : 72
    },
    {
      label: 'local-affine-refined',
      predictionMode: 'local-affine-knn',
      sampleCount: dimension === 3 ? 5600 : 7800,
      querySamples: 15,
      gradientStep: 0.085,
      slopeTolerance: 0.22,
      interceptTolerance: 0.44,
      minRegionSupport: 6,
      neighborCount: dimension === 3 ? 15 : 19,
      localAffineNeighborCount: dimension === 3 ? 64 : 96
    }
  ];
}

function evaluateAttempt(teacher: Teacher, config: AttemptConfig, noise: { std: number; quantization: number }) {
  let rawOracleCalls = 0;
  const noisyOracle = makeNoisyOracle(teacher, 91000 + teacher.seed + teacher.dimension + config.label.length, noise.std, noise.quantization);
  const countedOracle = (input: number[]): number => {
    rawOracleCalls++;
    return noisyOracle(input);
  };
  const model = decompileDeepBlackBoxReluNdRegionSampler(countedOracle, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-2, 2] as [number, number]),
    sampleCount: config.sampleCount,
    gradientStep: config.gradientStep,
    slopeTolerance: config.slopeTolerance,
    interceptTolerance: config.interceptTolerance,
    minRegionSupport: config.minRegionSupport,
    querySamples: config.querySamples,
    aggregation: 'median',
    neighborCount: config.neighborCount,
    predictionMode: config.predictionMode,
    localAffineNeighborCount: config.localAffineNeighborCount,
    randomSeed: 73001 + teacher.seed + teacher.dimension + config.label.length,
    ridge: 1e-8
  });

  let selectionOracleCalls = 0;
  const selectionNoisyOracle = makeNoisyOracle(teacher, 130000 + teacher.seed + config.label.length, noise.std, noise.quantization);
  const selectionInputs = makeInputs(140000 + teacher.seed + teacher.dimension, teacher.dimension, 320);
  const selectionTargets = selectionInputs.map(input => {
    selectionOracleCalls += 9;
    return aggregateNoisy(selectionNoisyOracle, input, 9);
  });
  const selectionPredicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, selectionInputs);
  const selectionMetrics = regressionMetrics(selectionTargets, selectionPredicted);

  const validationInputs = makeInputs(150000 + teacher.seed + teacher.dimension, teacher.dimension, 1400);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const maxRegionFitRmse = Math.max(...model.regions.map(region => region.fitRmse));

  return {
    label: config.label,
    predictionMode: config.predictionMode,
    model,
    metrics,
    selectionMetrics,
    maxRegionFitRmse,
    queryBudget: {
      rawOracleCalls,
      selectionOracleCalls,
      totalRawOracleCalls: rawOracleCalls + selectionOracleCalls,
      finiteDifferenceCallsPerSample: 1 + 2 * teacher.dimension,
      querySamples: config.querySamples
    }
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const noise = { std: 0.0025, quantization: 0.001 };
const teachers = [
  makeTeacher(600, 3, 'near-parallel'),
  makeTeacher(617, 3, 'cancellation'),
  makeTeacher(634, 3, 'low-amplitude'),
  makeTeacher(700, 4, 'near-parallel'),
  makeTeacher(717, 4, 'cancellation'),
  makeTeacher(734, 4, 'low-amplitude'),
  makeTeacher(751, 4, 'mixed-oblique')
];

const results = teachers.map(teacher => {
  const attempts = attemptConfigs(teacher.dimension).map(config => evaluateAttempt(teacher, config, noise));
  const selected = [...attempts].sort((a, b) => a.selectionMetrics.mae - b.selectionMetrics.mae)[0];
  const baselineInputs = makeInputs(410000 + teacher.seed + teacher.dimension, teacher.dimension, 1600);
  const baselineTargets = baselineInputs.map(input => predictTeacher(teacher, input));
  const linear = fitLinearBaseline(baselineInputs, baselineTargets);
  const validationInputs = makeInputs(150000 + teacher.seed + teacher.dimension, teacher.dimension, 1400);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const linearPredicted = validationInputs.map(input => predictLinear(linear, input));
  const linearMetrics = regressionMetrics(actual, linearPredicted);
  const improvementRatio = linearMetrics.mae / Math.max(1e-12, selected.metrics.mae);
  const deepOnlyRegionEstimate = selected.model.regionCount - teacher.l1.length - 1;
  const pass = selected.metrics.mae < 0.075
    && selected.metrics.r2 > 0.975
    && improvementRatio > 2.5
    && selected.model.retainedSampleCount / selected.model.sampleCount > 0.85
    && selected.model.regionCount >= 5;
  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    dimension: teacher.dimension,
    profile: teacher.profile,
    pass,
    selectedAttempt: selected.label,
    selectedPredictionMode: selected.predictionMode,
    recovery: {
      recoveryKind: selected.model.recoveryKind,
      regionCount: selected.model.regionCount,
      firstLayerWidth: teacher.l1.length,
      deepOnlyRegionEstimate,
      retainedSampleCount: selected.model.retainedSampleCount,
      droppedSampleCount: selected.model.droppedSampleCount,
      retainedSampleRatio: selected.model.retainedSampleCount / selected.model.sampleCount,
      maxRegionFitRmse: selected.maxRegionFitRmse,
      neighborCount: selected.model.neighborCount,
      localAffineNeighborCount: selected.model.localAffineNeighborCount
    },
    noise,
    metrics: selected.metrics,
    baselines: {
      globalLinear: linearMetrics,
      linearMaeImprovementRatio: improvementRatio
    },
    queryBudget: {
      selectedRawOracleCalls: selected.queryBudget.totalRawOracleCalls,
      allAttemptRawOracleCalls: attempts.reduce((sum, attempt) => sum + attempt.queryBudget.totalRawOracleCalls, 0)
    },
    attempts: attempts.map(attempt => ({
      label: attempt.label,
      predictionMode: attempt.predictionMode,
      regionCount: attempt.model.regionCount,
      retainedSampleRatio: attempt.model.retainedSampleCount / attempt.model.sampleCount,
      selectionMetrics: attempt.selectionMetrics,
      cleanMetrics: attempt.metrics,
      queryBudget: attempt.queryBudget
    }))
  };
});

const deepOnlyRegionEstimates = results.map(result => result.recovery.deepOnlyRegionEstimate);
const sortedDeepOnly = [...deepOnlyRegionEstimates].sort((a, b) => a - b);
const medianDeepOnly = sortedDeepOnly[Math.floor(sortedDeepOnly.length / 2)];
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  profiles: [...new Set(results.map(result => result.profile))],
  selectedLocalAffineCount: results.filter(result => result.selectedPredictionMode === 'local-affine-knn').length,
  minRegionCount: Math.min(...results.map(result => result.recovery.regionCount)),
  medianDeepOnlyRegionEstimate: medianDeepOnly,
  maxDeepOnlyRegionEstimate: Math.max(...deepOnlyRegionEstimates),
  minRetainedSampleRatio: Math.min(...results.map(result => result.recovery.retainedSampleRatio)),
  maxMae: Math.max(...results.map(result => result.metrics.mae)),
  minR2: Math.min(...results.map(result => result.metrics.r2)),
  minLinearMaeImprovementRatio: Math.min(...results.map(result => result.baselines.linearMaeImprovementRatio)),
  totalSelectedRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.selectedRawOracleCalls, 0),
  totalAdaptiveSearchRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.allAttemptRawOracleCalls, 0)
};

const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy adaptive-adversarial deep black-box ReLU ND region sampling',
  blackBox: {
    type: '3D/4D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP adversarial teachers',
    visibilityToGcl: 'noisy quantized oracle calls only: the decompiler receives repeated f(vector), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'benchmark-scoped noisy output-equivalent deep-ND recovery across generated near-parallel, cancellation, low-amplitude, and mixed-oblique folded teachers with adaptive region-vote/local-affine attempt selection',
    doesNotClaim: 'unique hidden-layer recovery, exact layer attribution, topology-perfect ND region adjacency, arbitrary production API extraction, or superiority over all model-extraction algorithms'
  },
  gates: {
    pass: summary.pass,
    allTeachersPassed: summary.passedTeachers === summary.teacherCount,
    dimensionsCovered: summary.dimensions.includes(3) && summary.dimensions.includes(4),
    adversarialProfilesCovered: summary.profiles.length >= 4,
    adaptiveLocalAffineExercised: summary.selectedLocalAffineCount > 0,
    minRegionCountAtLeast5: summary.minRegionCount >= 5,
    medianDeepOnlyRegionsAbove10: summary.medianDeepOnlyRegionEstimate > 10,
    maxMaeBelow0075: summary.maxMae < 0.075,
    r2Above0975: summary.minR2 > 0.975,
    beatsLinearBy2_5x: summary.minLinearMaeImprovementRatio > 2.5,
    retainedSamplesAbove85Percent: summary.minRetainedSampleRatio > 0.85
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Noisy Adaptive-Adversarial Deep Black-Box ReLU ND Region-Sampler Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves: ${report.claimScope.proves}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Summary',
  '',
  `- Pass: ${summary.pass}`,
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Dimensions: ${summary.dimensions.join(', ')}`,
  `- Profiles: ${summary.profiles.join(', ')}`,
  `- Selected local-affine attempts: ${summary.selectedLocalAffineCount}`,
  `- Min recovered regions: ${summary.minRegionCount}`,
  `- Median deep-only region estimate: ${summary.medianDeepOnlyRegionEstimate}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Min linear MAE improvement ratio: ${summary.minLinearMaeImprovementRatio}`,
  `- Total selected raw oracle calls: ${summary.totalSelectedRawOracleCalls}`,
  `- Total adaptive-search raw oracle calls: ${summary.totalAdaptiveSearchRawOracleCalls}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Pass: ${result.pass}`,
    `- Dimension: ${result.dimension}`,
    `- Profile: ${result.profile}`,
    `- Selected attempt: ${result.selectedAttempt}`,
    `- Selected prediction mode: ${result.selectedPredictionMode}`,
    `- Recovered regions: ${result.recovery.regionCount}`,
    `- Deep-only region estimate: ${result.recovery.deepOnlyRegionEstimate}`,
    `- Retained sample ratio: ${result.recovery.retainedSampleRatio}`,
    `- GCL MAE: ${result.metrics.mae}`,
    `- GCL R2: ${result.metrics.r2}`,
    `- Linear MAE: ${result.baselines.globalLinear.mae}`,
    `- Linear improvement ratio: ${result.baselines.linearMaeImprovementRatio}`,
    `- Selected raw oracle calls: ${result.queryBudget.selectedRawOracleCalls}`,
    ''
  ].join('\n'))
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Noisy/adaptive Phase-4 deep ND folded recovery gate.',
    'Output-only lane: no hidden activation or teacher weight access.',
    'Adaptive selection uses noisy held-out oracle responses; clean teacher metrics are used only for benchmark audit.'
  ]
});

console.log(JSON.stringify({
  pass: summary.pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary
}, null, 2));

if (!summary.pass) process.exit(1);
