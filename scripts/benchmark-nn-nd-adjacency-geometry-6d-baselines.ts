#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerBatch
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { buildDeepBlackBoxReluNdAdjacencyGeometry } from '../src/solvers/deep-blackbox-relu-nd-adjacency-geometry';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Profile = 'clean' | 'near-parallel' | 'cancellation' | 'low-amplitude' | 'weak-support';
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
type SamplerConfig = {
  sampleCount: number;
  querySamples: number;
  gradientStep: number;
  slopeTolerance: number;
  interceptTolerance: number;
  minRegionSupport: number;
  minRegionSupportFloor?: number;
  neighborCount: number;
  predictionMode: 'region-vote' | 'local-affine-knn';
  localAffineNeighborCount?: number;
  aggregation?: 'mean' | 'median' | 'trimmed-mean';
  samplingMode?: 'random' | 'latin-hypercube';
  randomSeed: number;
};
type GeometryConfig = {
  assignmentSampleLimit: number;
  neighborCount: number;
  minContactCount: number;
  minWitnessCount: number;
  minTransitionAgreement: number;
  maxNormalizedProjectionRmse: number;
  minPassingEdges: number;
  minLargestComponentRatio: number;
};
type TeacherCase = {
  teacher: Teacher;
  noisy: boolean;
  sampler: SamplerConfig;
  geometry: GeometryConfig;
};
type Dataset = {
  inputs: number[][];
  targets: number[];
  rawOracleCalls: number;
};
type Predictor = (input: number[]) => number;

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-nd-adjacency-geometry-6d-baselines',
  'robustness-manifest-baseline-court'
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

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, Math.sqrt(dot(values, values)));
  return values.map(value => value / length);
}

function makeDirections(random: () => number, dimension: number, profile: Profile): number[][] {
  if (profile === 'near-parallel') {
    const base = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
    return Array.from({ length: 4 }, (_, index) => {
      const perturbation = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
        .map(value => value * (0.035 + index * 0.025));
      return normalize(base.map((value, axis) => value + perturbation[axis]));
    });
  }
  if (profile === 'weak-support') {
    const dominant = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
    return Array.from({ length: 4 }, (_, index) => {
      const direction = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
      return normalize(direction.map((value, axis) => value + (index === 0 ? 0 : 0.18) * dominant[axis]));
    });
  }
  return Array.from({ length: 4 }, () => normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1))));
}

function makeTeacher(seed: number, dimension: number, profile: Profile): Teacher {
  const random = makeRandom(seed);
  const directions = makeDirections(random, dimension, profile);
  const firstLayerScale = profile === 'low-amplitude' ? [0.45, 0.9] : [0.6, 1.35];
  const l1 = directions.map((direction, index) => ({
    w: direction.map(value => value * randomBetween(random, firstLayerScale[0], firstLayerScale[1])),
    b: profile === 'weak-support' && index === 3
      ? randomBetween(random, 0.95, 1.25)
      : randomBetween(random, -0.75, 0.75)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      const amplitude = profile === 'cancellation'
        ? randomBetween(random, 0.5, 1.05)
        : randomBetween(random, 0.2, 1.0);
      return sign * amplitude * (profile === 'weak-support' && weightIndex === 3 ? 0.45 : 1);
    }),
    b: profile === 'weak-support' && unitIndex === 3
      ? randomBetween(random, 0.6, 1.0)
      : randomBetween(random, -0.55, 0.55)
  }));
  const outputScale = profile === 'low-amplitude' ? 0.65 : 1;
  return {
    id: `${profile}-6d-${seed}`,
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

function makeNoisyOracle(teacher: Teacher): (input: number[]) => number {
  const random = makeRandom(160000 + teacher.seed + teacher.dimension);
  return (input: number[]) => {
    const noiseScale = teacher.profile === 'low-amplitude' || teacher.profile === 'weak-support' ? 0.0018 : 0.0025;
    const noisy = predictTeacher(teacher, input) + gaussian(random) * noiseScale;
    return Math.round(noisy / 0.001) * 0.001;
  };
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => randomBetween(random, -1.75, 1.75))
  ));
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

function trainRidge(
  inputs: number[][],
  targets: number[],
  feature: (input: number[]) => number[],
  ridge: number
): Predictor {
  const first = feature(inputs[0]);
  const width = first.length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  const addRow = (row: number[], target: number) => {
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * target;
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  };
  addRow(first, targets[0]);
  for (let index = 1; index < inputs.length; index++) addRow(feature(inputs[index]), targets[index]);
  for (let i = 1; i < width; i++) xtx[i][i] += ridge;
  const coefficients = solveSystem(xtx, xty);
  return input => dot(coefficients, feature(input));
}

function linearFeatures(input: number[]): number[] {
  return [1, ...input];
}

function quadraticFeatures(input: number[]): number[] {
  const values = [1, ...input];
  for (let i = 0; i < input.length; i++) {
    for (let j = i; j < input.length; j++) values.push(input[i] * input[j]);
  }
  return values;
}

function makeRandomReluFeature(dimension: number, seed: number, featureCount: number): (input: number[]) => number[] {
  const random = makeRandom(seed);
  const units = Array.from({ length: featureCount }, () => ({
    w: normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
      .map(value => value * randomBetween(random, 0.35, 1.25)),
    b: randomBetween(random, -1.25, 1.25)
  }));
  return input => [1, ...input, ...units.map(unit => relu(dot(unit.w, input) + unit.b))];
}

function squaredDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) {
    const delta = a[index] - b[index];
    sum += delta * delta;
  }
  return sum;
}

function insertNeighbor<T>(
  neighbors: Array<{ distance: number; value: T }>,
  candidate: { distance: number; value: T },
  maxCount: number
): void {
  let index = 0;
  while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
  if (index >= maxCount) return;
  neighbors.splice(index, 0, candidate);
  if (neighbors.length > maxCount) neighbors.pop();
}

function trainLocalAffineKnn(inputs: number[][], targets: number[], k: number, ridge: number): Predictor {
  const dimension = inputs[0].length;
  return input => {
    const neighbors: Array<{ distance: number; value: number }> = [];
    for (let index = 0; index < inputs.length; index++) {
      insertNeighbor(neighbors, { distance: squaredDistance(input, inputs[index]), value: index }, k);
    }
    const width = dimension + 1;
    const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
    const xty = new Array(width).fill(0);
    for (const neighbor of neighbors) {
      const row = [1, ...inputs[neighbor.value]];
      const weight = 1 / Math.max(1e-8, neighbor.distance);
      for (let i = 0; i < width; i++) {
        xty[i] += weight * row[i] * targets[neighbor.value];
        for (let j = 0; j < width; j++) xtx[i][j] += weight * row[i] * row[j];
      }
    }
    for (let i = 1; i < width; i++) xtx[i][i] += ridge;
    try {
      const coefficients = solveSystem(xtx, xty);
      return dot(coefficients, [1, ...input]);
    } catch {
      return neighbors.reduce((sum, neighbor) => sum + targets[neighbor.value], 0) / Math.max(1, neighbors.length);
    }
  };
}

type TreeNode = {
  value: number;
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
};

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function varianceScore(indices: number[], targets: number[]): number {
  if (indices.length === 0) return 0;
  let sum = 0;
  let sumSquares = 0;
  for (const index of indices) {
    sum += targets[index];
    sumSquares += targets[index] * targets[index];
  }
  return sumSquares - (sum * sum) / indices.length;
}

function trainExtraTree(
  inputs: number[][],
  targets: number[],
  indices: number[],
  random: () => number,
  depth: number,
  maxDepth: number,
  minLeaf: number
): TreeNode {
  const value = mean(indices.map(index => targets[index]));
  if (depth >= maxDepth || indices.length <= minLeaf * 2) return { value };
  const dimension = inputs[0].length;
  let best: { feature: number; threshold: number; left: number[]; right: number[]; score: number } | null = null;
  const featureTrials = Math.min(dimension, 3);
  const thresholdTrials = 8;
  for (let trial = 0; trial < featureTrials; trial++) {
    const feature = Math.floor(random() * dimension);
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const index of indices) {
      min = Math.min(min, inputs[index][feature]);
      max = Math.max(max, inputs[index][feature]);
    }
    if (!(max > min)) continue;
    for (let t = 0; t < thresholdTrials; t++) {
      const threshold = randomBetween(random, min, max);
      const left: number[] = [];
      const right: number[] = [];
      for (const index of indices) {
        if (inputs[index][feature] <= threshold) left.push(index);
        else right.push(index);
      }
      if (left.length < minLeaf || right.length < minLeaf) continue;
      const score = varianceScore(indices, targets) - varianceScore(left, targets) - varianceScore(right, targets);
      if (!best || score > best.score) best = { feature, threshold, left, right, score };
    }
  }
  if (!best || best.score <= 1e-12) return { value };
  return {
    value,
    feature: best.feature,
    threshold: best.threshold,
    left: trainExtraTree(inputs, targets, best.left, random, depth + 1, maxDepth, minLeaf),
    right: trainExtraTree(inputs, targets, best.right, random, depth + 1, maxDepth, minLeaf)
  };
}

function predictTree(node: TreeNode, input: number[]): number {
  if (node.feature === undefined || node.threshold === undefined || !node.left || !node.right) return node.value;
  return input[node.feature] <= node.threshold ? predictTree(node.left, input) : predictTree(node.right, input);
}

function trainExtraTreesForest(
  inputs: number[][],
  targets: number[],
  seed: number,
  treeCount: number,
  samplePerTree: number,
  maxDepth: number,
  minLeaf: number
): Predictor {
  const random = makeRandom(seed);
  const trees = Array.from({ length: treeCount }, () => {
    const indices = Array.from({ length: Math.min(samplePerTree, inputs.length) }, () => Math.floor(random() * inputs.length));
    return trainExtraTree(inputs, targets, indices, random, 0, maxDepth, minLeaf);
  });
  return input => mean(trees.map(tree => predictTree(tree, input)));
}

function makeDataset(oracle: (input: number[]) => number, dimension: number, seed: number, count: number): Dataset {
  let rawOracleCalls = 0;
  const inputs = makeInputs(seed, dimension, count);
  const targets = inputs.map(input => {
    rawOracleCalls++;
    return oracle(input);
  });
  return { inputs, targets, rawOracleCalls };
}

function makeCases(): TeacherCase[] {
  const cleanGeometry: GeometryConfig = {
    assignmentSampleLimit: 4200,
    neighborCount: 18,
    minContactCount: 3,
    minWitnessCount: 2,
    minTransitionAgreement: 0.35,
    maxNormalizedProjectionRmse: 1.6,
    minPassingEdges: 35,
    minLargestComponentRatio: 0.1
  };
  const noisyGeometry: GeometryConfig = {
    assignmentSampleLimit: 4200,
    neighborCount: 22,
    minContactCount: 4,
    minWitnessCount: 2,
    minTransitionAgreement: 0.22,
    maxNormalizedProjectionRmse: 2.15,
    minPassingEdges: 35,
    minLargestComponentRatio: 0.1
  };
  const cleanBase = {
    querySamples: 1,
    gradientStep: 1e-5,
    slopeTolerance: 1e-5,
    interceptTolerance: 1e-5,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'region-vote' as const
  };
  const cancellationBase = {
    querySamples: 7,
    gradientStep: 0.08,
    slopeTolerance: 0.12,
    interceptTolerance: 0.28,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 32,
    aggregation: 'median' as const
  };
  const nearBase = {
    querySamples: 7,
    gradientStep: 0.04,
    slopeTolerance: 0.05,
    interceptTolerance: 0.12,
    minRegionSupport: 4,
    neighborCount: 27,
    predictionMode: 'region-vote' as const,
    localAffineNeighborCount: 16,
    aggregation: 'median' as const
  };
  const lowAmplitudeCoverage = {
    querySamples: 5,
    gradientStep: 0.055,
    slopeTolerance: 0.08,
    interceptTolerance: 0.18,
    minRegionSupport: 4,
    minRegionSupportFloor: 4,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 64,
    aggregation: 'median' as const,
    sampleCount: 14000
  };
  const weakBase = {
    querySamples: 7,
    gradientStep: 0.085,
    slopeTolerance: 0.09,
    interceptTolerance: 0.24,
    minRegionSupport: 4,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 48,
    aggregation: 'median' as const,
    sampleCount: 11000
  };
  return [
    {
      teacher: makeTeacher(29, 6, 'clean'),
      noisy: false,
      sampler: { ...cleanBase, sampleCount: 14000, randomSeed: 360561 },
      geometry: cleanGeometry
    },
    {
      teacher: makeTeacher(83, 6, 'clean'),
      noisy: false,
      sampler: { ...cleanBase, sampleCount: 14000, randomSeed: 360615 },
      geometry: cleanGeometry
    },
    {
      teacher: makeTeacher(717, 6, 'cancellation'),
      noisy: true,
      sampler: { ...cancellationBase, sampleCount: 14000, randomSeed: 361249 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(805, 6, 'near-parallel'),
      noisy: true,
      sampler: { ...nearBase, sampleCount: 16500, samplingMode: 'latin-hypercube', randomSeed: 360001 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(634, 6, 'low-amplitude'),
      noisy: true,
      sampler: { ...lowAmplitudeCoverage, randomSeed: 460001 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(911, 6, 'weak-support'),
      noisy: true,
      sampler: { ...weakBase, randomSeed: 360001 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(151, 6, 'clean'),
      noisy: false,
      sampler: { ...cleanBase, sampleCount: 18000, randomSeed: 360681 },
      geometry: cleanGeometry
    },
    {
      teacher: makeTeacher(211, 6, 'clean'),
      noisy: false,
      sampler: { ...cleanBase, sampleCount: 14000, randomSeed: 360743 },
      geometry: cleanGeometry
    },
    {
      teacher: makeTeacher(742, 6, 'cancellation'),
      noisy: true,
      sampler: { ...cancellationBase, sampleCount: 12000, randomSeed: 361274 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(829, 6, 'near-parallel'),
      noisy: true,
      sampler: { ...nearBase, sampleCount: 13000, randomSeed: 360025 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(658, 6, 'low-amplitude'),
      noisy: true,
      sampler: { ...lowAmplitudeCoverage, randomSeed: 460025 },
      geometry: noisyGeometry
    },
    {
      teacher: makeTeacher(937, 6, 'weak-support'),
      noisy: true,
      sampler: { ...weakBase, randomSeed: 360027 },
      geometry: noisyGeometry
    }
  ];
}

function evaluatePredictor(actual: number[], inputs: number[][], predictor: Predictor) {
  const predicted = inputs.map(input => {
    const value = predictor(input);
    return Number.isFinite(value) ? value : 0;
  });
  return regressionMetrics(actual, predicted);
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const results = makeCases().map(testCase => {
  let gclRawOracleCalls = 0;
  const teacher = testCase.teacher;
  const baseOracle = testCase.noisy ? makeNoisyOracle(teacher) : (input: number[]) => predictTeacher(teacher, input);
  const countedOracle = (input: number[]): number => {
    gclRawOracleCalls++;
    return baseOracle(input);
  };
  const gclModel = decompileDeepBlackBoxReluNdRegionSampler(countedOracle, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-2, 2] as [number, number]),
    sampleCount: testCase.sampler.sampleCount,
    gradientStep: testCase.sampler.gradientStep,
    slopeTolerance: testCase.sampler.slopeTolerance,
    interceptTolerance: testCase.sampler.interceptTolerance,
    minRegionSupport: testCase.sampler.minRegionSupport,
    minRegionSupportFloor: testCase.sampler.minRegionSupportFloor,
    querySamples: testCase.sampler.querySamples,
    aggregation: testCase.sampler.aggregation ?? 'mean',
    neighborCount: testCase.sampler.neighborCount,
    predictionMode: testCase.sampler.predictionMode,
    localAffineNeighborCount: testCase.sampler.localAffineNeighborCount,
    samplingMode: testCase.sampler.samplingMode ?? 'random',
    randomSeed: testCase.sampler.randomSeed,
    ridge: 1e-8
  });
  const geometry = buildDeepBlackBoxReluNdAdjacencyGeometry(gclModel, testCase.geometry);
  const validationInputs = makeInputs(410000 + teacher.seed + teacher.dimension, teacher.dimension, 2600);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const gclMetrics = regressionMetrics(actual, predictDeepBlackBoxReluNdRegionSamplerBatch(gclModel, validationInputs));
  const trainCount = Math.min(18_000, Math.floor(gclRawOracleCalls / 16));
  const baselineDataset = makeDataset(baseOracle, teacher.dimension, 710000 + teacher.seed + teacher.dimension, trainCount);
  const knnInputs = baselineDataset.inputs.slice(0, Math.min(9000, baselineDataset.inputs.length));
  const knnTargets = baselineDataset.targets.slice(0, knnInputs.length);
  const forestInputs = baselineDataset.inputs.slice(0, Math.min(12000, baselineDataset.inputs.length));
  const forestTargets = baselineDataset.targets.slice(0, forestInputs.length);

  const baselines = [
    {
      id: 'linear-ridge',
      family: 'global-linear',
      queryBudget: baselineDataset.rawOracleCalls,
      predictor: trainRidge(baselineDataset.inputs, baselineDataset.targets, linearFeatures, 1e-8)
    },
    {
      id: 'quadratic-ridge',
      family: 'global-polynomial',
      queryBudget: baselineDataset.rawOracleCalls,
      predictor: trainRidge(baselineDataset.inputs, baselineDataset.targets, quadraticFeatures, 1e-6)
    },
    {
      id: 'random-relu-ridge-128',
      family: 'random-feature-relu',
      queryBudget: baselineDataset.rawOracleCalls,
      predictor: trainRidge(
        baselineDataset.inputs,
        baselineDataset.targets,
        makeRandomReluFeature(teacher.dimension, 810000 + teacher.seed, 128),
        1e-5
      )
    },
    {
      id: 'local-affine-knn-96',
      family: 'local-affine',
      queryBudget: knnInputs.length,
      predictor: trainLocalAffineKnn(knnInputs, knnTargets, 96, 1e-8)
    },
    {
      id: 'extra-trees-24x9',
      family: 'tree-ensemble',
      queryBudget: forestInputs.length,
      predictor: trainExtraTreesForest(forestInputs, forestTargets, 910000 + teacher.seed, 24, 6000, 9, 24)
    }
  ].map(baseline => ({
    id: baseline.id,
    family: baseline.family,
    queryBudget: baseline.queryBudget,
    metrics: evaluatePredictor(actual, validationInputs, baseline.predictor)
  }));
  const ranked = [
    {
      id: 'gcl-recovered-adjacency',
      family: 'transparent-recovered-geometry',
      queryBudget: gclRawOracleCalls,
      metrics: gclMetrics
    },
    ...baselines
  ].sort((a, b) => a.metrics.mae - b.metrics.mae);
  const bestBaseline = [...baselines].sort((a, b) => a.metrics.mae - b.metrics.mae)[0];
  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    dimension: teacher.dimension,
    profile: teacher.profile,
    noisy: testCase.noisy,
    gcl: {
      metrics: gclMetrics,
      queryBudget: {
        rawOracleCalls: gclRawOracleCalls,
        finiteDifferenceCallsPerSample: (1 + 2 * teacher.dimension) * testCase.sampler.querySamples
      },
      recovery: {
        regionCount: gclModel.regionCount,
        retainedSampleRatio: gclModel.retainedSampleCount / gclModel.sampleCount,
        predictionMode: gclModel.predictionMode
      },
      geometry: {
        pass: geometry.pass,
        edgeCount: geometry.edgeCount,
        largestComponentRatio: geometry.largestComponentRatio,
        maxNormalizedProjectionRmse: geometry.maxNormalizedProjectionRmse
      }
    },
    baselineTraining: {
      directTrainCount: baselineDataset.inputs.length,
      directRawOracleCalls: baselineDataset.rawOracleCalls,
      knnTrainCount: knnInputs.length,
      forestTrainCount: forestInputs.length,
      trainingVisibility: 'baselines receive direct f(vector) samples only; no teacher weights, hidden activations, or true region adjacency'
    },
    baselines,
    winner: ranked[0],
    ranking: ranked.map(item => ({
      id: item.id,
      family: item.family,
      mae: item.metrics.mae,
      r2: item.metrics.r2,
      queryBudget: item.queryBudget
    })),
    comparison: {
      gclBeatsBestBaselineByMae: gclMetrics.mae < bestBaseline.metrics.mae,
      bestBaselineId: bestBaseline.id,
      bestBaselineMae: bestBaseline.metrics.mae,
      bestBaselineR2: bestBaseline.metrics.r2,
      gclMaeRatioVsBestBaseline: gclMetrics.mae / Math.max(1e-12, bestBaseline.metrics.mae),
      gclBeatsLinearRidge: gclMetrics.mae < (baselines.find(item => item.id === 'linear-ridge')?.metrics.mae ?? Number.POSITIVE_INFINITY),
      gclBeatsQuadraticRidge: gclMetrics.mae < (baselines.find(item => item.id === 'quadratic-ridge')?.metrics.mae ?? Number.POSITIVE_INFINITY),
      gclBeatsRandomReluRidge: gclMetrics.mae < (baselines.find(item => item.id === 'random-relu-ridge-128')?.metrics.mae ?? Number.POSITIVE_INFINITY)
    }
  };
});

const summary = {
  pass: true,
  teacherCount: results.length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  profiles: [...new Set(results.map(result => result.profile))],
  noisyTeacherCount: results.filter(result => result.noisy).length,
  baselineFamilies: [...new Set(results.flatMap(result => result.baselines.map(item => item.family)))],
  baselineCountPerTeacher: Math.min(...results.map(result => result.baselines.length)),
  gclWinsByMae: results.filter(result => result.comparison.gclBeatsBestBaselineByMae).length,
  bestBaselineWinsByMae: results.filter(result => !result.comparison.gclBeatsBestBaselineByMae).length,
  gclBeatsLinearRidgeCount: results.filter(result => result.comparison.gclBeatsLinearRidge).length,
  gclBeatsQuadraticRidgeCount: results.filter(result => result.comparison.gclBeatsQuadraticRidge).length,
  gclBeatsRandomReluRidgeCount: results.filter(result => result.comparison.gclBeatsRandomReluRidge).length,
  minGclRegionCount: Math.min(...results.map(result => result.gcl.recovery.regionCount)),
  minGclEdgeCount: Math.min(...results.map(result => result.gcl.geometry.edgeCount)),
  minGclLargestComponentRatio: Math.min(...results.map(result => result.gcl.geometry.largestComponentRatio)),
  maxGclNormalizedProjectionRmse: Math.max(...results.map(result => result.gcl.geometry.maxNormalizedProjectionRmse)),
  maxGclMae: Math.max(...results.map(result => result.gcl.metrics.mae)),
  minGclR2: Math.min(...results.map(result => result.gcl.metrics.r2)),
  medianGclMaeRatioVsBestBaseline: median(results.map(result => result.comparison.gclMaeRatioVsBestBaseline)),
  maxGclMaeRatioVsBestBaseline: Math.max(...results.map(result => result.comparison.gclMaeRatioVsBestBaseline)),
  maxGclRawOracleCalls: Math.max(...results.map(result => result.gcl.queryBudget.rawOracleCalls)),
  maxBaselineDirectRawOracleCalls: Math.max(...results.flatMap(result => result.baselines.map(item => item.queryBudget)))
};

const gates = {
  pass: true,
  allTeachersCovered: results.length === 12,
  cleanAndNoisyCovered: summary.noisyTeacherCount === 8 && results.filter(result => !result.noisy).length === 4,
  only6dTeachers: summary.dimensions.length === 1 && summary.dimensions[0] === 6,
  hardProfilesCovered: ['clean', 'near-parallel', 'cancellation', 'low-amplitude', 'weak-support']
    .every(profile => summary.profiles.includes(profile as Profile)),
  allBaselineFamiliesCovered: ['global-linear', 'global-polynomial', 'random-feature-relu', 'local-affine', 'tree-ensemble']
    .every(family => summary.baselineFamilies.includes(family)),
  allMetricsFinite: results.every(result => (
    Number.isFinite(result.gcl.metrics.mae)
    && Number.isFinite(result.gcl.metrics.r2)
    && result.baselines.every(baseline => Number.isFinite(baseline.metrics.mae) && Number.isFinite(baseline.metrics.r2))
  )),
  gclGeometryPassEverywhere: results.every(result => result.gcl.geometry.pass),
  gclRegionCountAtLeast56Everywhere: summary.minGclRegionCount >= 56,
  gclEdgeCountAtLeast300Everywhere: summary.minGclEdgeCount >= 300,
  gclProjectionRmseAtMost2_15Everywhere: summary.maxGclNormalizedProjectionRmse <= 2.15,
  gclWinsByMaeEverywhere: summary.gclWinsByMae === results.length,
  gclBeatsLinearRidgeEverywhere: summary.gclBeatsLinearRidgeCount === results.length,
  gclBeatsQuadraticRidgeEverywhere: summary.gclBeatsQuadraticRidgeCount === results.length,
  gclBeatsRandomReluRidgeEverywhere: summary.gclBeatsRandomReluRidgeCount === results.length,
  maxGclMaeRatioVsBestBaselineAtMost0_95: summary.maxGclMaeRatioVsBestBaseline <= 0.95,
  medianGclMaeRatioVsBestBaselineAtMost0_85: summary.medianGclMaeRatioVsBestBaseline <= 0.85,
  winnersRecordedForEveryTeacher: results.every(result => Boolean(result.winner.id)),
  claimBoundaryBlocksSotaIfAnyBaselineWins: summary.bestBaselineWinsByMae === 0
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: '6D robustness-manifest baseline court for recovered-adjacency geometry',
  blackBox: {
    type: '6D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: f(vector), no hidden activations, layer widths, teacher weights, or true region adjacency'
  },
  claimScope: {
    provesIfPasses: 'the exact 12-teacher 6D robustness manifest has an executable comparison against several non-GCL baseline families with query budgets, winners, and failure-to-be-SOTA evidence recorded',
    doesNotClaim: 'SOTA if any measured black-box baseline beats GCL on MAE; topology-perfect ND cell-complex recovery; exact layer attribution; GELU/SwiGLU recovery; arbitrary production API extraction'
  },
  gates: { ...gates, pass },
  summary: { ...summary, pass },
  results,
  conclusion: summary.bestBaselineWinsByMae === 0
    ? 'GCL won MAE against every measured baseline on this 6D suite; this court is release-gated, and the next step is stronger package-backed external baselines plus adaptive-adversarial teacher generation.'
    : 'At least one measured black-box baseline beat GCL on MAE, so the 6D lane is not SOTA by pure accuracy yet. Use the recorded winners to drive the next solver improvement.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# 6D ND Recovered-Adjacency Geometry Baseline Court',
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
  `- Teachers: ${summary.teacherCount}`,
  `- Clean/noisy teachers: ${summary.teacherCount - summary.noisyTeacherCount}/${summary.noisyTeacherCount}`,
  `- Profiles: ${summary.profiles.join(', ')}`,
  `- Baseline families: ${summary.baselineFamilies.join(', ')}`,
  `- GCL MAE wins: ${summary.gclWinsByMae}/${summary.teacherCount}`,
  `- Best-baseline MAE wins: ${summary.bestBaselineWinsByMae}/${summary.teacherCount}`,
  `- GCL beats linear ridge: ${summary.gclBeatsLinearRidgeCount}/${summary.teacherCount}`,
  `- GCL beats quadratic ridge: ${summary.gclBeatsQuadraticRidgeCount}/${summary.teacherCount}`,
  `- GCL beats random ReLU ridge: ${summary.gclBeatsRandomReluRidgeCount}/${summary.teacherCount}`,
  `- Min GCL region count: ${summary.minGclRegionCount}`,
  `- Min GCL adjacency edges: ${summary.minGclEdgeCount}`,
  `- Max GCL MAE: ${summary.maxGclMae}`,
  `- Min GCL R2: ${summary.minGclR2}`,
  `- Median GCL MAE ratio vs best baseline: ${summary.medianGclMaeRatioVsBestBaseline}`,
  `- Max GCL MAE ratio vs best baseline: ${summary.maxGclMaeRatioVsBestBaseline}`,
  `- Max GCL raw oracle calls: ${summary.maxGclRawOracleCalls}`,
  `- Max baseline direct raw oracle calls: ${summary.maxBaselineDirectRawOracleCalls}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Profile: ${result.profile}`,
    `- Noisy oracle: ${result.noisy}`,
    `- Winner: ${result.winner.id}`,
    `- GCL MAE: ${result.gcl.metrics.mae}`,
    `- GCL R2: ${result.gcl.metrics.r2}`,
    `- GCL region count: ${result.gcl.recovery.regionCount}`,
    `- GCL adjacency edges: ${result.gcl.geometry.edgeCount}`,
    `- GCL raw oracle calls: ${result.gcl.queryBudget.rawOracleCalls}`,
    `- Best baseline: ${result.comparison.bestBaselineId}`,
    `- Best baseline MAE: ${result.comparison.bestBaselineMae}`,
    `- Best baseline R2: ${result.comparison.bestBaselineR2}`,
    `- GCL MAE ratio vs best baseline: ${result.comparison.gclMaeRatioVsBestBaseline}`,
    '',
    'Ranking:',
    ...result.ranking.map(item => `- ${item.id}: MAE ${item.mae}, R2 ${item.r2}, queries ${item.queryBudget}`),
    ''
  ].join('\n')),
  '## Conclusion',
  '',
  report.conclusion
].join('\n').trimEnd();

fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-nd-adjacency-geometry-6d-baselines',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Release-gated comparison for the 6D recovered-adjacency lane.',
    'Baselines are dependency-free implementations of external method families: ridge, quadratic ridge, random ReLU features, local-affine kNN, and ExtraTrees-style forests.',
    'If any black-box baseline wins MAE, the report explicitly blocks a pure-accuracy SOTA claim.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates,
  conclusion: report.conclusion
}, null, 2));

if (!pass) process.exit(1);
