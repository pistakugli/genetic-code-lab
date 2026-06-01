#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileActiveSubspaceFfn,
  predictActiveSubspaceFfnBatch
} from '../src/solvers/active-subspace-ffn-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Predictor = (input: number[]) => number;
type NoisyCase = {
  id: string;
  dimension: number;
  directionSeed: number;
  randomSeed: number;
  gradientProbeCount: number;
  trainingSampleCount: number;
  validationSampleCount: number;
  neighborCount: number;
  noiseSigma: number;
  minR2: number;
  maxMae: number;
  minProjectionScore: number;
  maxRawOracleCalls: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-active-subspace-ffn-noisy',
  'rank1-gelu-noisy-128d-256d'
);

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
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function normalize(values: number[]): number[] {
  const length = Math.sqrt(Math.max(1e-12, dot(values, values)));
  return values.map(value => value / length);
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function makeDirection(seed: number, dimension: number): number[] {
  const random = makeRandom(seed);
  return normalize(Array.from({ length: dimension }, () => gaussian(random)));
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => randomBetween(random, -1.5, 1.5))
  ));
}

function project(input: number[], directions: number[][]): number[] {
  return directions.map(direction => dot(direction, input));
}

function makeRandomDirections(seed: number, dimension: number, rank: number): number[][] {
  const directions: number[][] = [];
  for (let index = 0; index < rank; index++) {
    let candidate = makeDirection(seed + index * 97, dimension);
    for (const direction of directions) {
      const projection = dot(candidate, direction);
      candidate = candidate.map((value, axis) => value - projection * direction[axis]);
    }
    directions.push(normalize(candidate));
  }
  return directions;
}

function squaredDistance(a: number[], b: number[]): number {
  let distance = 0;
  for (let index = 0; index < a.length; index++) {
    const delta = a[index] - b[index];
    distance += delta * delta;
  }
  return distance;
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
    if (Math.abs(divisor) < 1e-12) throw new Error('baseline local system is singular');
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

function quadraticFeatures(values: number[]): number[] {
  const row = [1, ...values];
  for (let i = 0; i < values.length; i++) {
    for (let j = i; j < values.length; j++) row.push(values[i] * values[j]);
  }
  return row;
}

function trainMeanBaseline(targets: number[]): Predictor {
  const mean = targets.reduce((sum, value) => sum + value, 0) / Math.max(1, targets.length);
  return () => mean;
}

function trainProjectedLocalQuadratic(
  oracle: Predictor,
  dimension: number,
  directions: number[][],
  seed: number,
  trainCount: number,
  neighborCount: number
): { predictor: Predictor; queryBudget: number } {
  const inputs = makeInputs(seed, dimension, trainCount);
  const rows = inputs.map(input => ({ coords: project(input, directions), target: oracle(input) }));
  const mean = rows.reduce((sum, row) => sum + row.target, 0) / Math.max(1, rows.length);
  return {
    queryBudget: trainCount,
    predictor: input => {
      const coords = project(input, directions);
      const neighbors: Array<{ distance: number; coords: number[]; target: number }> = [];
      for (const row of rows) {
        const candidate = { distance: squaredDistance(coords, row.coords), coords: row.coords, target: row.target };
        let index = 0;
        while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
        if (index >= neighborCount) continue;
        neighbors.splice(index, 0, candidate);
        if (neighbors.length > neighborCount) neighbors.pop();
      }
      const queryRow = quadraticFeatures(coords);
      const width = queryRow.length;
      if (neighbors.length < width) return mean;
      const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
      const xty = new Array(width).fill(0);
      for (const neighbor of neighbors) {
        const row = quadraticFeatures(neighbor.coords);
        const weight = 1 / Math.max(1e-8, neighbor.distance);
        for (let i = 0; i < width; i++) {
          xty[i] += weight * row[i] * neighbor.target;
          for (let j = 0; j < width; j++) xtx[i][j] += weight * row[i] * row[j];
        }
      }
      for (let i = 1; i < width; i++) xtx[i][i] += 1e-8;
      try {
        const value = dot(solveSystem(xtx, xty), queryRow);
        return Number.isFinite(value) ? value : mean;
      } catch {
        return neighbors.reduce((sum, neighbor) => sum + neighbor.target, 0) / Math.max(1, neighbors.length);
      }
    }
  };
}

function evaluate(actual: number[], inputs: number[][], predictor: Predictor) {
  return regressionMetrics(actual, inputs.map(input => {
    const value = predictor(input);
    return Number.isFinite(value) ? value : 0;
  }));
}

function makeCases(): NoisyCase[] {
  return [
    {
      id: 'gelu-rank1-noisy-128d',
      dimension: 128,
      directionSeed: 9911,
      randomSeed: 9913,
      gradientProbeCount: 10,
      trainingSampleCount: 1800,
      validationSampleCount: 1000,
      neighborCount: 52,
      noiseSigma: 0.01,
      minR2: 0.99,
      maxMae: 0.055,
      minProjectionScore: 0.998,
      maxRawOracleCalls: 4500
    },
    {
      id: 'gelu-rank1-noisy-256d',
      dimension: 256,
      directionSeed: 9999,
      randomSeed: 5004,
      gradientProbeCount: 10,
      trainingSampleCount: 3600,
      validationSampleCount: 2000,
      neighborCount: 100,
      noiseSigma: 0.01,
      minR2: 0.99,
      maxMae: 0.055,
      minProjectionScore: 0.998,
      maxRawOracleCalls: 9100
    }
  ];
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const results = makeCases().map(testCase => {
  const direction = makeDirection(testCase.directionSeed, testCase.dimension);
  const trueOracle = (input: number[]): number => 0.5 + 1.2 * gelu(1.8 * dot(direction, input) - 0.3);
  const makeNoisyOracle = (seed: number): Predictor => {
    const random = makeRandom(seed);
    return input => trueOracle(input) + gaussian(random) * testCase.noiseSigma;
  };
  const ranges = Array.from({ length: testCase.dimension }, () => [-1.5, 1.5] as [number, number]);
  let gclOracleCalls = 0;
  const gclNoisyOracle = makeNoisyOracle(42424242 + testCase.randomSeed);
  const model = decompileActiveSubspaceFfn(input => {
    gclOracleCalls++;
    return gclNoisyOracle(input);
  }, {
    dimension: testCase.dimension,
    ranges,
    activeRank: 1,
    gradientProbeCount: testCase.gradientProbeCount,
    trainingSampleCount: testCase.trainingSampleCount,
    finiteDifferenceStep: 1e-3,
    neighborCount: testCase.neighborCount,
    randomSeed: testCase.randomSeed,
    ridge: 1e-8
  });
  const validationInputs = makeInputs(900000 + testCase.randomSeed, testCase.dimension, testCase.validationSampleCount);
  const actual = validationInputs.map(trueOracle);
  const predicted = predictActiveSubspaceFfnBatch(model, validationInputs);
  const gclMetrics = regressionMetrics(actual, predicted);
  const trainingInputs = makeInputs(700000 + testCase.randomSeed, testCase.dimension, testCase.trainingSampleCount);
  const trainingNoisyOracle = makeNoisyOracle(52525252 + testCase.randomSeed);
  const trainingTargets = trainingInputs.map(trainingNoisyOracle);
  const meanBaseline = trainMeanBaseline(trainingTargets);
  const randomProjectionBaseline = trainProjectedLocalQuadratic(
    makeNoisyOracle(62626262 + testCase.randomSeed),
    testCase.dimension,
    makeRandomDirections(810000 + testCase.randomSeed, testCase.dimension, 1),
    820000 + testCase.randomSeed,
    testCase.trainingSampleCount,
    testCase.neighborCount
  );
  const coordinateProjectionBaseline = trainProjectedLocalQuadratic(
    makeNoisyOracle(72727272 + testCase.randomSeed),
    testCase.dimension,
    [Array.from({ length: testCase.dimension }, (_unused, axis) => axis === 0 ? 1 : 0)],
    830000 + testCase.randomSeed,
    testCase.trainingSampleCount,
    testCase.neighborCount
  );
  const baselines = [
    {
      id: 'mean-noisy-output',
      family: 'constant',
      queryBudget: testCase.trainingSampleCount,
      metrics: evaluate(actual, validationInputs, meanBaseline)
    },
    {
      id: 'random-projection-local-quadratic-rank1',
      family: 'random-subspace-local-quadratic',
      queryBudget: randomProjectionBaseline.queryBudget,
      metrics: evaluate(actual, validationInputs, randomProjectionBaseline.predictor)
    },
    {
      id: 'coordinate-head-local-quadratic-rank1',
      family: 'coordinate-subspace-local-quadratic',
      queryBudget: coordinateProjectionBaseline.queryBudget,
      metrics: evaluate(actual, validationInputs, coordinateProjectionBaseline.predictor)
    }
  ];
  const bestBaseline = [...baselines].sort((a, b) => a.metrics.mae - b.metrics.mae)[0];
  const projectionScore = Math.abs(dot(direction, model.activeDirections[0]));
  const ranking = [
    {
      id: 'gcl-active-subspace-noisy',
      family: 'active-subspace-ffn-surrogate',
      queryBudget: model.queryBudget.rawOracleCalls,
      metrics: gclMetrics
    },
    ...baselines
  ].sort((a, b) => a.metrics.mae - b.metrics.mae);
  return {
    id: testCase.id,
    dimension: testCase.dimension,
    activeRank: 1,
    noiseSigma: testCase.noiseSigma,
    thresholds: {
      minR2: testCase.minR2,
      maxMae: testCase.maxMae,
      minProjectionScore: testCase.minProjectionScore,
      maxRawOracleCalls: testCase.maxRawOracleCalls
    },
    gcl: {
      metrics: gclMetrics,
      expression: model.expression,
      queryBudget: model.queryBudget,
      countedOracleCalls: gclOracleCalls,
      directionEstimator: model.directionEstimator,
      directionCandidateScores: model.directionCandidateScores,
      finiteDifferenceStep: model.finiteDifferenceStep,
      activeSubspace: {
        selectedGradientEnergyRatio: model.selectedGradientEnergyRatio,
        meanGradientResidualRatio: model.meanGradientResidualRatio,
        projectionScore
      }
    },
    baselines,
    ranking: ranking.map(item => ({
      id: item.id,
      family: item.family,
      mae: item.metrics.mae,
      r2: item.metrics.r2,
      queryBudget: item.queryBudget
    })),
    comparison: {
      bestBaselineId: bestBaseline.id,
      bestBaselineMae: bestBaseline.metrics.mae,
      bestBaselineR2: bestBaseline.metrics.r2,
      gclBeatsBestBaselineByMae: gclMetrics.mae < bestBaseline.metrics.mae,
      gclMaeRatioVsBestBaseline: gclMetrics.mae / Math.max(1e-12, bestBaseline.metrics.mae)
    },
    pass: gclMetrics.r2 >= testCase.minR2
      && gclMetrics.mae <= testCase.maxMae
      && projectionScore >= testCase.minProjectionScore
      && model.queryBudget.rawOracleCalls <= testCase.maxRawOracleCalls
      && gclMetrics.mae < bestBaseline.metrics.mae
  };
});

const summary = {
  pass: true,
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  dimensions: results.map(result => result.dimension),
  noiseSigma: [...new Set(results.map(result => result.noiseSigma))],
  maxMae: Math.max(...results.map(result => result.gcl.metrics.mae)),
  minR2: Math.min(...results.map(result => result.gcl.metrics.r2)),
  minProjectionScore: Math.min(...results.map(result => result.gcl.activeSubspace.projectionScore)),
  maxRawOracleCalls: Math.max(...results.map(result => result.gcl.queryBudget.rawOracleCalls)),
  maxGclMaeRatioVsBestBaseline: Math.max(...results.map(result => result.comparison.gclMaeRatioVsBestBaseline)),
  gclWinsByMae: results.filter(result => result.comparison.gclBeatsBestBaselineByMae).length
};

const gates = {
  pass: true,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  dimensionsCovered: summary.dimensions.includes(128) && summary.dimensions.includes(256),
  noisyOracleCovered: summary.noiseSigma.includes(0.01),
  allR2ThresholdsPassed: results.every(result => result.gcl.metrics.r2 >= result.thresholds.minR2),
  allMaeThresholdsPassed: results.every(result => result.gcl.metrics.mae <= result.thresholds.maxMae),
  projectionScoreAtLeast0998: summary.minProjectionScore >= 0.998,
  queryBudgetWithinCaseLimits: results.every(result => result.gcl.queryBudget.rawOracleCalls <= result.thresholds.maxRawOracleCalls),
  gclWinsEveryBaselineCourt: summary.gclWinsByMae === summary.teacherCount,
  maxGclMaeRatioVsBestBaselineAtMost05: summary.maxGclMaeRatioVsBestBaseline <= 0.5
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy active-subspace FFN decompiler for rank-1 GELU teachers',
  blackBox: {
    type: '128D/256D single-channel GELU teachers with stochastic Gaussian output noise',
    visibilityToGcl: 'oracle calls only: noisy f(vector), no active direction, no weights, no hidden activations; validation is against held-out noiseless truth'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover a high-dimensional rank-1 smooth FFN active direction and faithful surrogate from noisy oracle calls under explicit query budgets',
    doesNotClaim: 'real LLM decompilation, arbitrary stochastic API extraction, exact GELU parameter recovery, mixed multi-channel FFN block recovery, or broad SOTA status'
  },
  gates: { ...gates, pass },
  summary: { ...summary, pass },
  results,
  conclusion: pass
    ? 'Noisy active-subspace FFN gate passed for controlled 128D/256D rank-1 GELU teachers; next step is rank-adaptive mixed-channel noisy FFN recovery.'
    : 'Noisy active-subspace FFN gate failed; inspect the recorded case before promoting the noisy lane.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Noisy Active-Subspace FFN Decompiler Benchmark',
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
  `- Teachers: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Dimensions: ${summary.dimensions.join(', ')}`,
  `- Noise sigma: ${summary.noiseSigma.join(', ')}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Min projection score: ${summary.minProjectionScore}`,
  `- Max raw oracle calls: ${summary.maxRawOracleCalls}`,
  `- GCL MAE wins: ${summary.gclWinsByMae}/${summary.teacherCount}`,
  `- Max GCL/best-baseline MAE ratio: ${summary.maxGclMaeRatioVsBestBaseline}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.id}`,
    '',
    `- Dimension: ${result.dimension}`,
    `- Noise sigma: ${result.noiseSigma}`,
    `- Pass: ${result.pass}`,
    `- GCL MAE: ${result.gcl.metrics.mae}`,
    `- GCL R2: ${result.gcl.metrics.r2}`,
    `- Direction projection score: ${result.gcl.activeSubspace.projectionScore}`,
    `- Direction estimator: ${result.gcl.directionEstimator}`,
    `- Finite-difference step: ${result.gcl.finiteDifferenceStep}`,
    `- Raw oracle calls: ${result.gcl.queryBudget.rawOracleCalls}`,
    `- Best baseline: ${result.comparison.bestBaselineId}`,
    `- Best baseline MAE: ${result.comparison.bestBaselineMae}`,
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

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-active-subspace-ffn-noisy',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Regression gate for the external noisy active-subspace failure mode.',
    'Validation is against noiseless held-out truth while GCL and baselines only query noisy oracles.',
    'This is not a real LLM decompilation claim; it is the noisy bridge for the Active-Subspace Neural Decompiler lane.'
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
