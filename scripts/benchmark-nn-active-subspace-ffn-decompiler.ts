#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileActiveSubspaceFfn,
  predictActiveSubspaceFfnBatch
} from '../src/solvers/active-subspace-ffn-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type TeacherKind = 'gelu-single-channel' | 'swiglu-two-direction';
type Teacher = {
  id: string;
  kind: TeacherKind;
  dimension: number;
  activeRank: number;
  valueDirection: number[];
  gateDirection?: number[];
  predict: (input: number[]) => number;
};
type CaseConfig = {
  teacher: Teacher;
  gradientProbeCount: number;
  trainingSampleCount: number;
  validationSampleCount: number;
  neighborCount: number;
  randomSeed: number;
  minR2: number;
  maxMae: number;
};
type Predictor = (input: number[]) => number;

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-active-subspace-ffn-decompiler',
  'gelu-swiglu-256d-1024d'
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

function silu(value: number): number {
  return value / (1 + Math.exp(-value));
}

function makeDirection(seed: number, dimension: number): number[] {
  const random = makeRandom(seed);
  return normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
}

function makeOrthogonalPair(seed: number, dimension: number): [number[], number[]] {
  const first = makeDirection(seed, dimension);
  const random = makeRandom(seed + 104729);
  const raw = Array.from({ length: dimension }, () => randomBetween(random, -1, 1));
  const projection = dot(raw, first);
  return [first, normalize(raw.map((value, index) => value - projection * first[index]))];
}

function makeGeluTeacher(seed: number, dimension: number): Teacher {
  const direction = makeDirection(seed, dimension);
  const predict = (input: number[]): number => 0.13 + 1.4 * gelu(1.5 * dot(direction, input) - 0.25);
  return {
    id: `gelu-${dimension}d-${seed}`,
    kind: 'gelu-single-channel',
    dimension,
    activeRank: 1,
    valueDirection: direction,
    predict
  };
}

function makeSwiGluTeacher(seed: number, dimension: number): Teacher {
  const [valueDirection, gateDirection] = makeOrthogonalPair(seed, dimension);
  const predict = (input: number[]): number => {
    const value = 0.9 * dot(valueDirection, input) - 0.12;
    const gate = 1.3 * dot(gateDirection, input) + 0.18;
    return -0.03 + 1.12 * value * silu(gate);
  };
  return {
    id: `swiglu-${dimension}d-${seed}`,
    kind: 'swiglu-two-direction',
    dimension,
    activeRank: 2,
    valueDirection,
    gateDirection,
    predict
  };
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

function trainProjectedLocalQuadratic(
  oracle: (input: number[]) => number,
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
        return dot(solveSystem(xtx, xty), queryRow);
      } catch {
        return neighbors.reduce((sum, neighbor) => sum + neighbor.target, 0) / Math.max(1, neighbors.length);
      }
    }
  };
}

function trainMeanBaseline(targets: number[]): Predictor {
  const mean = targets.reduce((sum, value) => sum + value, 0) / Math.max(1, targets.length);
  return () => mean;
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

function subspaceProjectionNorm(vector: number[], basis: number[][]): number {
  return Math.sqrt(basis.reduce((sum, direction) => sum + dot(vector, direction) ** 2, 0));
}

function evaluate(actual: number[], inputs: number[][], predictor: Predictor) {
  return regressionMetrics(actual, inputs.map(input => {
    const value = predictor(input);
    return Number.isFinite(value) ? value : 0;
  }));
}

function makeCases(): CaseConfig[] {
  return [
    {
      teacher: makeGeluTeacher(3101, 256),
      gradientProbeCount: 12,
      trainingSampleCount: 2200,
      validationSampleCount: 1400,
      neighborCount: 48,
      randomSeed: 5101,
      minR2: 0.995,
      maxMae: 0.012
    },
    {
      teacher: makeSwiGluTeacher(3201, 256),
      gradientProbeCount: 18,
      trainingSampleCount: 3600,
      validationSampleCount: 1600,
      neighborCount: 88,
      randomSeed: 5201,
      minR2: 0.99,
      maxMae: 0.035
    },
    {
      teacher: makeGeluTeacher(3301, 1024),
      gradientProbeCount: 10,
      trainingSampleCount: 2400,
      validationSampleCount: 1400,
      neighborCount: 48,
      randomSeed: 5301,
      minR2: 0.995,
      maxMae: 0.014
    },
    {
      teacher: makeSwiGluTeacher(3401, 1024),
      gradientProbeCount: 20,
      trainingSampleCount: 4400,
      validationSampleCount: 1600,
      neighborCount: 96,
      randomSeed: 5401,
      minR2: 0.99,
      maxMae: 0.04
    }
  ];
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const results = makeCases().map(testCase => {
  const teacher = testCase.teacher;
  const ranges = Array.from({ length: teacher.dimension }, () => [-1.5, 1.5] as [number, number]);
  const model = decompileActiveSubspaceFfn(teacher.predict, {
    dimension: teacher.dimension,
    ranges,
    activeRank: teacher.activeRank,
    gradientProbeCount: testCase.gradientProbeCount,
    trainingSampleCount: testCase.trainingSampleCount,
    finiteDifferenceStep: 1e-4,
    neighborCount: testCase.neighborCount,
    randomSeed: testCase.randomSeed,
    ridge: 1e-8
  });
  const validationInputs = makeInputs(610000 + testCase.randomSeed, teacher.dimension, testCase.validationSampleCount);
  const actual = validationInputs.map(teacher.predict);
  const predicted = predictActiveSubspaceFfnBatch(model, validationInputs);
  const gclMetrics = regressionMetrics(actual, predicted);
  const trainingInputs = makeInputs(710000 + testCase.randomSeed, teacher.dimension, testCase.trainingSampleCount);
  const trainingTargets = trainingInputs.map(teacher.predict);
  const randomProjectionBaseline = trainProjectedLocalQuadratic(
    teacher.predict,
    teacher.dimension,
    makeRandomDirections(810000 + testCase.randomSeed, teacher.dimension, teacher.activeRank),
    820000 + testCase.randomSeed,
    testCase.trainingSampleCount,
    testCase.neighborCount
  );
  const coordinateProjectionBaseline = trainProjectedLocalQuadratic(
    teacher.predict,
    teacher.dimension,
    Array.from({ length: teacher.activeRank }, (_, index) => (
      Array.from({ length: teacher.dimension }, (_unused, axis) => axis === index ? 1 : 0)
    )),
    830000 + testCase.randomSeed,
    testCase.trainingSampleCount,
    testCase.neighborCount
  );
  const meanBaseline = trainMeanBaseline(trainingTargets);
  const baselines = [
    {
      id: 'mean-output',
      family: 'constant',
      queryBudget: testCase.trainingSampleCount,
      metrics: evaluate(actual, validationInputs, meanBaseline)
    },
    {
      id: `random-projection-local-quadratic-rank${teacher.activeRank}`,
      family: 'random-subspace-local-quadratic',
      queryBudget: randomProjectionBaseline.queryBudget,
      metrics: evaluate(actual, validationInputs, randomProjectionBaseline.predictor)
    },
    {
      id: `coordinate-head-local-quadratic-rank${teacher.activeRank}`,
      family: 'coordinate-subspace-local-quadratic',
      queryBudget: coordinateProjectionBaseline.queryBudget,
      metrics: evaluate(actual, validationInputs, coordinateProjectionBaseline.predictor)
    }
  ];
  const bestBaseline = [...baselines].sort((a, b) => a.metrics.mae - b.metrics.mae)[0];
  const projectionScores = [
    subspaceProjectionNorm(teacher.valueDirection, model.activeDirections),
    ...(teacher.gateDirection ? [subspaceProjectionNorm(teacher.gateDirection, model.activeDirections)] : [])
  ];
  const ranking = [
    {
      id: 'gcl-active-subspace',
      family: 'active-subspace-ffn-surrogate',
      queryBudget: model.queryBudget.rawOracleCalls,
      metrics: gclMetrics
    },
    ...baselines
  ].sort((a, b) => a.metrics.mae - b.metrics.mae);
  return {
    teacherId: teacher.id,
    kind: teacher.kind,
    dimension: teacher.dimension,
    activeRank: teacher.activeRank,
    thresholds: { minR2: testCase.minR2, maxMae: testCase.maxMae },
    gcl: {
      metrics: gclMetrics,
      expression: model.expression,
      queryBudget: model.queryBudget,
      activeSubspace: {
        selectedGradientEnergyRatio: model.selectedGradientEnergyRatio,
        meanGradientResidualRatio: model.meanGradientResidualRatio,
        gradientEigenvalues: model.gradientEigenvalues,
        projectionScores,
        minProjectionScore: Math.min(...projectionScores)
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
      && model.selectedGradientEnergyRatio >= 0.99
      && Math.min(...projectionScores) >= 0.98
      && gclMetrics.mae < bestBaseline.metrics.mae
  };
});

const summary = {
  pass: true,
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  kinds: [...new Set(results.map(result => result.kind))],
  maxMae: Math.max(...results.map(result => result.gcl.metrics.mae)),
  minR2: Math.min(...results.map(result => result.gcl.metrics.r2)),
  minSelectedGradientEnergyRatio: Math.min(...results.map(result => result.gcl.activeSubspace.selectedGradientEnergyRatio)),
  maxMeanGradientResidualRatio: Math.max(...results.map(result => result.gcl.activeSubspace.meanGradientResidualRatio)),
  minProjectionScore: Math.min(...results.map(result => result.gcl.activeSubspace.minProjectionScore)),
  gclWinsByMae: results.filter(result => result.comparison.gclBeatsBestBaselineByMae).length,
  bestBaselineWinsByMae: results.filter(result => !result.comparison.gclBeatsBestBaselineByMae).length,
  maxGclMaeRatioVsBestBaseline: Math.max(...results.map(result => result.comparison.gclMaeRatioVsBestBaseline)),
  maxRawOracleCalls: Math.max(...results.map(result => result.gcl.queryBudget.rawOracleCalls)),
  maxBaselineRawOracleCalls: Math.max(...results.flatMap(result => result.baselines.map(baseline => baseline.queryBudget)))
};

const gates = {
  pass: true,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  dimensionsCovered: summary.dimensions.includes(256) && summary.dimensions.includes(1024),
  geluAndSwiGluCovered: summary.kinds.includes('gelu-single-channel') && summary.kinds.includes('swiglu-two-direction'),
  allR2ThresholdsPassed: results.every(result => result.gcl.metrics.r2 >= result.thresholds.minR2),
  allMaeThresholdsPassed: results.every(result => result.gcl.metrics.mae <= result.thresholds.maxMae),
  activeEnergyRatioAtLeast099: summary.minSelectedGradientEnergyRatio >= 0.99,
  projectionScoreAtLeast098: summary.minProjectionScore >= 0.98,
  gclWinsEveryBaselineCourt: summary.gclWinsByMae === summary.teacherCount,
  maxGclMaeRatioVsBestBaselineAtMost08: summary.maxGclMaeRatioVsBestBaseline <= 0.8,
  maxRawOracleCallsBelow50k: summary.maxRawOracleCalls < 50_000
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'active-subspace FFN decompiler for controlled high-dimensional GELU/SwiGLU teachers',
  blackBox: {
    type: '256D/1024D single-channel GELU and two-direction SwiGLU FFN teachers',
    visibilityToGcl: 'oracle calls only: f(vector), no active directions, no weights, no hidden activations'
  },
  claimScope: {
    provesIfPasses: 'GCL can discover a low-dimensional active subspace in controlled high-dimensional GELU/SwiGLU-style FFN channels and build a faithful output-equivalent surrogate with explicit query budget',
    doesNotClaim: 'real LLM weight recovery, arbitrary 4096D production model extraction, exact GELU/SwiGLU parameter recovery, tokenizer/model integration, or broad SOTA status'
  },
  gates: { ...gates, pass },
  summary: { ...summary, pass },
  results,
  conclusion: pass
    ? 'Active-subspace FFN foundation passed on controlled 256D/1024D GELU and SwiGLU teachers; next step is 4096D stress plus package-backed external baselines.'
    : 'Active-subspace FFN foundation failed at least one gate; inspect the recorded case before promoting the lane.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Active-Subspace FFN Decompiler Benchmark',
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
  `- Kinds: ${summary.kinds.join(', ')}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Min active energy ratio: ${summary.minSelectedGradientEnergyRatio}`,
  `- Max gradient residual ratio: ${summary.maxMeanGradientResidualRatio}`,
  `- Min true-direction projection score: ${summary.minProjectionScore}`,
  `- GCL MAE wins: ${summary.gclWinsByMae}/${summary.teacherCount}`,
  `- Max GCL/best-baseline MAE ratio: ${summary.maxGclMaeRatioVsBestBaseline}`,
  `- Max raw oracle calls: ${summary.maxRawOracleCalls}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Kind: ${result.kind}`,
    `- Dimension: ${result.dimension}`,
    `- Active rank: ${result.activeRank}`,
    `- Pass: ${result.pass}`,
    `- GCL MAE: ${result.gcl.metrics.mae}`,
    `- GCL R2: ${result.gcl.metrics.r2}`,
    `- Active energy ratio: ${result.gcl.activeSubspace.selectedGradientEnergyRatio}`,
    `- Min projection score: ${result.gcl.activeSubspace.minProjectionScore}`,
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
  lane: 'nn-active-subspace-ffn-decompiler',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'First controlled Active-Subspace Neural Decompiler foundation for high-dimensional FFN-style smooth activations.',
    'Covers single-channel GELU and two-direction SwiGLU-style teachers in 256D and 1024D.',
    'This is not a real LLM decompilation claim; it is the release-gated bridge from ReLU geometry to LLM-style FFN active subspaces.'
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
