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

type Unit = { w: number[]; b: number };
type Teacher = {
  id: string;
  seed: number;
  dimension: number;
  l1: Unit[];
  l2: Unit[];
  out: { w: number[]; b: number; base: number[] };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-deep-blackbox-relu-nd-region-sampler', '3d-4d-seed-sweep');

function relu(value: number): number {
  return Math.max(0, value);
}

function makeRandom(seed: number): () => number {
  let state = seed % 2147483647;
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
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function normalize(values: number[]): number[] {
  const length = Math.sqrt(dot(values, values));
  return values.map(value => value / Math.max(1e-12, length));
}

function makeTeacher(seed: number, dimension: number): Teacher {
  const random = makeRandom(seed);
  const l1 = Array.from({ length: 4 }, () => ({
    w: normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
      .map(value => value * randomBetween(random, 0.6, 1.4)),
    b: randomBetween(random, -0.7, 0.7)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      return sign * randomBetween(random, 0.2, 1.0);
    }),
    b: randomBetween(random, -0.5, 0.5)
  }));
  return {
    id: `deep-${dimension}d-${seed}`,
    seed,
    dimension,
    l1,
    l2,
    out: {
      w: Array.from({ length: 4 }, (_, index) => {
        const sign = index % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.3, 1.2);
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

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teachers = [
  makeTeacher(11, 3),
  makeTeacher(29, 3),
  makeTeacher(11, 4),
  makeTeacher(29, 4)
];

const results = teachers.map(teacher => {
  let oracleCalls = 0;
  const oracle = (input: number[]): number => {
    oracleCalls++;
    return predictTeacher(teacher, input);
  };
  const sampleCount = teacher.dimension === 3 ? 6500 : 12000;
  const neighborCount = teacher.dimension === 3 ? 11 : 15;
  const model = decompileDeepBlackBoxReluNdRegionSampler(oracle, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-2, 2] as [number, number]),
    sampleCount,
    gradientStep: 1e-5,
    slopeTolerance: 1e-5,
    interceptTolerance: 1e-5,
    minRegionSupport: 6,
    neighborCount,
    randomSeed: 260526 + teacher.seed + teacher.dimension
  });
  const validationInputs = makeInputs(310000 + teacher.seed + teacher.dimension, teacher.dimension, 2200);
  const actual = validationInputs.map(input => {
    oracleCalls++;
    return predictTeacher(teacher, input);
  });
  const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);

  const baselineInputs = makeInputs(410000 + teacher.seed + teacher.dimension, teacher.dimension, 1600);
  const baselineTargets = baselineInputs.map(input => predictTeacher(teacher, input));
  const linear = fitLinearBaseline(baselineInputs, baselineTargets);
  const linearPredicted = validationInputs.map(input => predictLinear(linear, input));
  const linearMetrics = regressionMetrics(actual, linearPredicted);
  const maxRegionFitRmse = Math.max(...model.regions.map(region => region.fitRmse));
  const improvementRatio = linearMetrics.mae / Math.max(1e-12, metrics.mae);
  const pass = model.regionCount > teacher.l1.length + 20
    && model.retainedSampleCount / model.sampleCount > 0.97
    && maxRegionFitRmse < 1e-7
    && metrics.mae < 0.018
    && metrics.r2 > 0.998
    && improvementRatio > 5;

  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    dimension: teacher.dimension,
    pass,
    recovery: {
      recoveryKind: model.recoveryKind,
      regionCount: model.regionCount,
      firstLayerWidth: teacher.l1.length,
      deepOnlyRegionEstimate: model.regionCount - teacher.l1.length - 1,
      retainedSampleCount: model.retainedSampleCount,
      droppedSampleCount: model.droppedSampleCount,
      maxRegionFitRmse,
      neighborCount: model.neighborCount
    },
    metrics,
    baselines: {
      globalLinear: linearMetrics,
      linearMaeImprovementRatio: improvementRatio
    },
    queryBudget: {
      rawOracleCalls: oracleCalls,
      samplerFitSamples: model.sampleCount,
      validationSamples: validationInputs.length,
      finiteDifferenceCallsPerSample: 1 + 2 * teacher.dimension
    }
  };
});

const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  minRegionCount: Math.min(...results.map(result => result.recovery.regionCount)),
  minDeepOnlyRegionEstimate: Math.min(...results.map(result => result.recovery.deepOnlyRegionEstimate)),
  minRetainedSampleRatio: Math.min(...results.map(result => result.recovery.retainedSampleCount / (result.recovery.retainedSampleCount + result.recovery.droppedSampleCount))),
  maxRegionFitRmse: Math.max(...results.map(result => result.recovery.maxRegionFitRmse)),
  maxMae: Math.max(...results.map(result => result.metrics.mae)),
  minR2: Math.min(...results.map(result => result.metrics.r2)),
  minLinearMaeImprovementRatio: Math.min(...results.map(result => result.baselines.linearMaeImprovementRatio)),
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0)
};

const report = {
  generatedAt: new Date().toISOString(),
  task: 'deep black-box ReLU ND output-equivalent affine-region sampling',
  blackBox: {
    type: '3D/4D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(vector), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'benchmark-scoped output-equivalent affine-region recovery in 3D/4D using random local-gradient probes and query-free kNN region assignment',
    doesNotClaim: 'unique hidden-layer recovery, exact layer attribution, topology-perfect region merging, noisy ND deep recovery, or arbitrary production API extraction'
  },
  gates: {
    pass: summary.pass,
    allTeachersPassed: summary.passedTeachers === summary.teacherCount,
    dimensionsCovered: summary.dimensions.includes(3) && summary.dimensions.includes(4),
    deepOnlyRegionsAbove20: summary.minDeepOnlyRegionEstimate > 20,
    retainedSamplesAbove97Percent: summary.minRetainedSampleRatio > 0.97,
    regionFitRmseBelow1eMinus7: summary.maxRegionFitRmse < 1e-7,
    maeBelow0018: summary.maxMae < 0.018,
    r2Above998: summary.minR2 > 0.998,
    beatsLinearBy5x: summary.minLinearMaeImprovementRatio > 5
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Deep Black-Box ReLU ND Region-Sampler Benchmark',
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
  `- Min recovered regions: ${summary.minRegionCount}`,
  `- Min deep-only region estimate: ${summary.minDeepOnlyRegionEstimate}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Min linear MAE improvement ratio: ${summary.minLinearMaeImprovementRatio}`,
  `- Total raw oracle calls: ${summary.totalRawOracleCalls}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Pass: ${result.pass}`,
    `- Dimension: ${result.dimension}`,
    `- Recovered regions: ${result.recovery.regionCount}`,
    `- Deep-only region estimate: ${result.recovery.deepOnlyRegionEstimate}`,
    `- Retained samples: ${result.recovery.retainedSampleCount}`,
    `- Dropped samples: ${result.recovery.droppedSampleCount}`,
    `- Max region fit RMSE: ${result.recovery.maxRegionFitRmse}`,
    `- GCL MAE: ${result.metrics.mae}`,
    `- GCL R2: ${result.metrics.r2}`,
    `- Linear MAE: ${result.baselines.globalLinear.mae}`,
    `- Linear improvement ratio: ${result.baselines.linearMaeImprovementRatio}`,
    `- Raw oracle calls: ${result.queryBudget.rawOracleCalls}`,
    ''
  ].join('\n'))
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-deep-blackbox-relu-nd-region-sampler',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'First Phase-4 deep ND folded recovery gate.',
    'Output-only lane: no hidden activation or teacher weight access.'
  ]
});

console.log(JSON.stringify({
  pass: summary.pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary
}, null, 2));

if (!summary.pass) process.exit(1);
