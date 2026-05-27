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
type BudgetLevel = 'low' | 'mid' | 'high';
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
  budgetLevel: BudgetLevel;
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
type AttemptResult = ReturnType<typeof evaluateAttempt> & {
  noisyPromoted: boolean;
  rejectionReasons: string[];
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler',
  '3d-4d-budget-search'
);

const previousReleaseReference = {
  selectedRawOracleCalls: 4274160,
  adaptiveSearchRawOracleCalls: 14132880,
  source: 'nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler/3d-4d-adaptive-search'
};

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
      label: 'low-region-vote',
      budgetLevel: 'low',
      predictionMode: 'region-vote',
      sampleCount: dimension === 3 ? 2400 : 3600,
      querySamples: 9,
      gradientStep: 0.09,
      slopeTolerance: 0.24,
      interceptTolerance: 0.48,
      minRegionSupport: 5,
      neighborCount: dimension === 3 ? 11 : 15,
      localAffineNeighborCount: dimension === 3 ? 40 : 60
    },
    {
      label: 'low-local-affine',
      budgetLevel: 'low',
      predictionMode: 'local-affine-knn',
      sampleCount: dimension === 3 ? 2400 : 3600,
      querySamples: 9,
      gradientStep: 0.09,
      slopeTolerance: 0.24,
      interceptTolerance: 0.48,
      minRegionSupport: 5,
      neighborCount: dimension === 3 ? 11 : 15,
      localAffineNeighborCount: dimension === 3 ? 40 : 60
    },
    {
      label: 'mid-region-vote',
      budgetLevel: 'mid',
      predictionMode: 'region-vote',
      sampleCount: dimension === 3 ? 3200 : 4800,
      querySamples: 11,
      gradientStep: 0.085,
      slopeTolerance: 0.22,
      interceptTolerance: 0.44,
      minRegionSupport: 6,
      neighborCount: dimension === 3 ? 13 : 17,
      localAffineNeighborCount: dimension === 3 ? 48 : 72
    },
    {
      label: 'mid-local-affine',
      budgetLevel: 'mid',
      predictionMode: 'local-affine-knn',
      sampleCount: dimension === 3 ? 3200 : 4800,
      querySamples: 11,
      gradientStep: 0.085,
      slopeTolerance: 0.22,
      interceptTolerance: 0.44,
      minRegionSupport: 6,
      neighborCount: dimension === 3 ? 13 : 17,
      localAffineNeighborCount: dimension === 3 ? 48 : 72
    },
    {
      label: 'high-region-vote',
      budgetLevel: 'high',
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
      label: 'high-local-affine',
      budgetLevel: 'high',
      predictionMode: 'local-affine-knn',
      sampleCount: dimension === 3 ? 4200 : 6200,
      querySamples: 13,
      gradientStep: 0.08,
      slopeTolerance: 0.18,
      interceptTolerance: 0.36,
      minRegionSupport: 8,
      neighborCount: dimension === 3 ? 13 : 17,
      localAffineNeighborCount: dimension === 3 ? 48 : 72
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
  const selectionInputs = makeInputs(140000 + teacher.seed + teacher.dimension, teacher.dimension, 240);
  const selectionTargets = selectionInputs.map(input => {
    selectionOracleCalls += 7;
    return aggregateNoisy(selectionNoisyOracle, input, 7);
  });
  const selectionPredicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, selectionInputs);
  const selectionMetrics = regressionMetrics(selectionTargets, selectionPredicted);

  const validationInputs = makeInputs(150000 + teacher.seed + teacher.dimension, teacher.dimension, 900);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const maxRegionFitRmse = Math.max(...model.regions.map(region => region.fitRmse));

  return {
    label: config.label,
    budgetLevel: config.budgetLevel,
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

function noisyPromotion(attempt: ReturnType<typeof evaluateAttempt>): { promoted: boolean; reasons: string[] } {
  const reasons: string[] = [];
  if (attempt.selectionMetrics.mae >= 0.06) reasons.push('selection_mae');
  if (attempt.selectionMetrics.r2 <= 0.98) reasons.push('selection_r2');
  if (attempt.model.regionCount < 5) reasons.push('region_count');
  if (attempt.model.retainedSampleCount / attempt.model.sampleCount <= 0.85) reasons.push('retained_sample_ratio');
  return { promoted: reasons.length === 0, reasons };
}

function runAdaptiveBudgetSearch(teacher: Teacher, noise: { std: number; quantization: number }): { selected: AttemptResult; attempts: AttemptResult[] } {
  const attempts: AttemptResult[] = [];
  for (const config of attemptConfigs(teacher.dimension)) {
    const attempt = evaluateAttempt(teacher, config, noise);
    const promotion = noisyPromotion(attempt);
    const result = { ...attempt, noisyPromoted: promotion.promoted, rejectionReasons: promotion.reasons };
    attempts.push(result);
    if (result.noisyPromoted) return { selected: result, attempts };
  }
  const selected = [...attempts].sort((a, b) => a.selectionMetrics.mae - b.selectionMetrics.mae)[0];
  return { selected, attempts };
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
  const { selected, attempts } = runAdaptiveBudgetSearch(teacher, noise);
  const baselineInputs = makeInputs(410000 + teacher.seed + teacher.dimension, teacher.dimension, 1600);
  const baselineTargets = baselineInputs.map(input => predictTeacher(teacher, input));
  const linear = fitLinearBaseline(baselineInputs, baselineTargets);
  const validationInputs = makeInputs(150000 + teacher.seed + teacher.dimension, teacher.dimension, 900);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const linearPredicted = validationInputs.map(input => predictLinear(linear, input));
  const linearMetrics = regressionMetrics(actual, linearPredicted);
  const improvementRatio = linearMetrics.mae / Math.max(1e-12, selected.metrics.mae);
  const deepOnlyRegionEstimate = selected.model.regionCount - teacher.l1.length - 1;
  const pass = selected.noisyPromoted
    && selected.metrics.mae < 0.075
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
    selectedBudgetLevel: selected.budgetLevel,
    selectedPredictionMode: selected.predictionMode,
    attemptedCandidateCount: attempts.length,
    rejectedCandidateCount: attempts.filter(attempt => !attempt.noisyPromoted).length,
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
    selectionMetrics: selected.selectionMetrics,
    baselines: {
      globalLinear: linearMetrics,
      linearMaeImprovementRatio: improvementRatio
    },
    queryBudget: {
      selectedRawOracleCalls: selected.queryBudget.totalRawOracleCalls,
      adaptiveSearchRawOracleCalls: attempts.reduce((sum, attempt) => sum + attempt.queryBudget.totalRawOracleCalls, 0)
    },
    attempts: attempts.map(attempt => ({
      label: attempt.label,
      budgetLevel: attempt.budgetLevel,
      predictionMode: attempt.predictionMode,
      noisyPromoted: attempt.noisyPromoted,
      rejectionReasons: attempt.rejectionReasons,
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
const totalSelectedRawOracleCalls = results.reduce((sum, result) => sum + result.queryBudget.selectedRawOracleCalls, 0);
const totalAdaptiveSearchRawOracleCalls = results.reduce((sum, result) => sum + result.queryBudget.adaptiveSearchRawOracleCalls, 0);
const selectedOracleReduction = 1 - (totalSelectedRawOracleCalls / previousReleaseReference.selectedRawOracleCalls);
const searchOracleReduction = 1 - (totalAdaptiveSearchRawOracleCalls / previousReleaseReference.adaptiveSearchRawOracleCalls);
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  profiles: [...new Set(results.map(result => result.profile))],
  selectedLocalAffineCount: results.filter(result => result.selectedPredictionMode === 'local-affine-knn').length,
  selectedLowBudgetCount: results.filter(result => result.selectedBudgetLevel === 'low').length,
  selectedMidBudgetCount: results.filter(result => result.selectedBudgetLevel === 'mid').length,
  selectedHighBudgetCount: results.filter(result => result.selectedBudgetLevel === 'high').length,
  rejectedCandidateCount: results.reduce((sum, result) => sum + result.rejectedCandidateCount, 0),
  maxAttemptedCandidateCount: Math.max(...results.map(result => result.attemptedCandidateCount)),
  minRegionCount: Math.min(...results.map(result => result.recovery.regionCount)),
  medianDeepOnlyRegionEstimate: sortedDeepOnly[Math.floor(sortedDeepOnly.length / 2)],
  minRetainedSampleRatio: Math.min(...results.map(result => result.recovery.retainedSampleRatio)),
  maxMae: Math.max(...results.map(result => result.metrics.mae)),
  minR2: Math.min(...results.map(result => result.metrics.r2)),
  minSelectionR2: Math.min(...results.map(result => result.selectionMetrics.r2)),
  minLinearMaeImprovementRatio: Math.min(...results.map(result => result.baselines.linearMaeImprovementRatio)),
  totalSelectedRawOracleCalls,
  totalAdaptiveSearchRawOracleCalls,
  previousReleaseReference,
  selectedOracleReduction,
  searchOracleReduction
};

const gates = {
  pass: summary.pass,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  dimensionsCovered: summary.dimensions.includes(3) && summary.dimensions.includes(4),
  adversarialProfilesCovered: summary.profiles.length >= 4,
  adaptiveLocalAffineExercised: summary.selectedLocalAffineCount > 0,
  lowBudgetUsuallySelected: summary.selectedLowBudgetCount >= 5,
  highBudgetEscalationExercised: summary.selectedHighBudgetCount > 0,
  rejectedCandidateLedgerPresent: summary.rejectedCandidateCount > 0,
  maxCandidateAttemptsAtMost5: summary.maxAttemptedCandidateCount <= 5,
  minRegionCountAtLeast5: summary.minRegionCount >= 5,
  medianDeepOnlyRegionsAbove10: summary.medianDeepOnlyRegionEstimate > 10,
  maxMaeBelow0075: summary.maxMae < 0.075,
  r2Above0975: summary.minR2 > 0.975,
  noisySelectionR2Above098: summary.minSelectionR2 > 0.98,
  beatsLinearBy2_5x: summary.minLinearMaeImprovementRatio > 2.5,
  retainedSamplesAbove85Percent: summary.minRetainedSampleRatio > 0.85,
  selectedOracleReductionAbove50Percent: summary.selectedOracleReduction > 0.5,
  searchOracleReductionAbove65Percent: summary.searchOracleReduction > 0.65,
  adaptiveSearchBelowPreviousSelectedCost: summary.totalAdaptiveSearchRawOracleCalls < previousReleaseReference.selectedRawOracleCalls
};

const report = {
  generatedAt: new Date().toISOString(),
  task: 'adaptive-budget noisy deep black-box ReLU ND region sampling',
  blackBox: {
    type: '3D/4D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP adversarial teachers',
    visibilityToGcl: 'noisy quantized oracle calls only: the decompiler receives repeated f(vector), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'benchmark-scoped query-budget reduction for noisy deep-ND output-equivalent recovery using sequential noisy holdout promotion and explicit rejected-candidate accounting',
    doesNotClaim: 'unique hidden-layer recovery, exact layer attribution, topology-perfect ND region adjacency, arbitrary production API extraction, or globally optimal query complexity'
  },
  gates,
  summary,
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Adaptive-Budget Noisy Deep Black-Box ReLU ND Region-Sampler Benchmark',
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
  `- Pass: ${gates.pass && Object.values(gates).every(Boolean)}`,
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Dimensions: ${summary.dimensions.join(', ')}`,
  `- Profiles: ${summary.profiles.join(', ')}`,
  `- Selected local-affine attempts: ${summary.selectedLocalAffineCount}`,
  `- Selected low/mid/high budgets: ${summary.selectedLowBudgetCount}/${summary.selectedMidBudgetCount}/${summary.selectedHighBudgetCount}`,
  `- Rejected candidates recorded: ${summary.rejectedCandidateCount}`,
  `- Max attempted candidates per teacher: ${summary.maxAttemptedCandidateCount}`,
  `- Min recovered regions: ${summary.minRegionCount}`,
  `- Median deep-only region estimate: ${summary.medianDeepOnlyRegionEstimate}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Min linear MAE improvement ratio: ${summary.minLinearMaeImprovementRatio}`,
  `- Total selected raw oracle calls: ${summary.totalSelectedRawOracleCalls}`,
  `- Total adaptive-search raw oracle calls: ${summary.totalAdaptiveSearchRawOracleCalls}`,
  `- Selected oracle reduction vs previous release: ${summary.selectedOracleReduction}`,
  `- Search oracle reduction vs previous adaptive search: ${summary.searchOracleReduction}`,
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
    `- Selected budget: ${result.selectedBudgetLevel}`,
    `- Selected prediction mode: ${result.selectedPredictionMode}`,
    `- Attempted candidates: ${result.attemptedCandidateCount}`,
    `- Rejected candidates: ${result.rejectedCandidateCount}`,
    `- Recovered regions: ${result.recovery.regionCount}`,
    `- Deep-only region estimate: ${result.recovery.deepOnlyRegionEstimate}`,
    `- GCL MAE: ${result.metrics.mae}`,
    `- GCL R2: ${result.metrics.r2}`,
    `- Selection R2: ${result.selectionMetrics.r2}`,
    `- Linear improvement ratio: ${result.baselines.linearMaeImprovementRatio}`,
    `- Selected raw oracle calls: ${result.queryBudget.selectedRawOracleCalls}`,
    `- Adaptive-search raw oracle calls: ${result.queryBudget.adaptiveSearchRawOracleCalls}`,
    ''
  ].join('\n'))
].join('\n');

const pass = Object.values(gates).every(Boolean);
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Adaptive-budget Phase-4 deep ND folded recovery gate.',
    'Sequential promotion uses noisy held-out oracle responses; clean teacher metrics are used only for benchmark audit.',
    'Rejected candidates are recorded so low-budget failures cannot be silently skipped.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
