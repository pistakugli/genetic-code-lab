#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileBlackBoxReluNdLineProbe,
  predictBlackBoxReluNdLineProbeBatch
} from '../src/solvers/blackbox-relu-nd-line-probe';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type TeacherTerm = {
  normal: number[];
  offset: number;
  readout: number;
};

type Teacher = {
  id: string;
  seed: number;
  dimension: 4 | 5;
  bias: number;
  base: number[];
  hidden: TeacherTerm[];
};

type RunResult = {
  teacherId: string;
  dimension: 4 | 5;
  pass: boolean;
  teacher: Teacher;
  noise: {
    noiseStd: number;
    quantization: number;
    querySamples: number;
    aggregation: 'median';
  };
  config: {
    ranges: Array<[number, number]>;
    lineCount: number;
    samplesPerLine: number;
    ransacIterations: number;
    minSupport: number;
    minSegmentPoints: number;
    testRows: number;
  };
  baselines: {
    linearHyperplane: {
      coefficients: number[];
      metrics: ReturnType<typeof regressionMetrics>;
    };
  };
  gclLineProbe: {
    expression: string;
    baseIntercept: number;
    baseWeights: number[];
    terms: Array<{ normal: number[]; offset: number; readout: number; support: number; residual: number }>;
    kinkPointCount: number;
    metrics: ReturnType<typeof regressionMetrics>;
  };
  queryBudget: {
    rawOracleCalls: number;
    aggregatedPointEvaluations: number;
    testRows: number;
  };
  gates: {
    recoveredAllHyperplanes: boolean;
    maxPlaneErrorBelow006: boolean;
    maeBelow0025: boolean;
    r2Above0998: boolean;
    beatsLinearBy75Percent: boolean;
    supportAboveMinimum: boolean;
    queryBudgetBelowLimit: boolean;
    maxPlaneError: number;
  };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu-nd-line-probe', '4d-5d-seed-sweep');

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = norm(values);
  if (length <= 1e-12) throw new Error('cannot normalize near-zero vector');
  return values.map(value => value / length);
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

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeRanges(dimension: number): Array<[number, number]> {
  return Array.from({ length: dimension }, (_, index) => {
    const span = 2.45 - index * 0.15;
    return [-span, span] as [number, number];
  });
}

function makeNormal(random: () => number, dimension: number, existing: number[]): number[] {
  for (let attempt = 0; attempt < 1000; attempt++) {
    const normal = normalize(Array.from({ length: dimension }, (_, index) => {
      const axisBias = index === 0 ? 0.55 : 0;
      return randomBetween(random, -1, 1) + axisBias;
    }));
    if (Math.abs(normal[0]) < 0.18) continue;
    let separated = true;
    for (let i = 0; i < existing.length; i += dimension) {
      const other = existing.slice(i, i + dimension);
      if (Math.abs(dot(normal, other)) > 0.82) {
        separated = false;
        break;
      }
    }
    if (separated) return normal;
  }
  throw new Error('failed to generate separated random normal');
}

function makeTeacher(seed: number, dimension: 4 | 5): Teacher {
  const random = makeRandom(seed);
  const base = Array.from({ length: dimension }, (_, index) => {
    const sign = index % 2 === 0 ? 1 : -1;
    return sign * randomBetween(random, 0.03, 0.11);
  });
  const normalBank: number[] = [];
  const offsets = dimension === 4 ? [-0.72, 0.03, 0.78] : [-0.82, -0.08, 0.72];
  const readoutSigns = [1, -1, 1];
  const hidden = offsets.map((offset, index) => {
    const normal = makeNormal(random, dimension, normalBank);
    normalBank.push(...normal);
    return {
      normal,
      offset: offset + randomBetween(random, -0.035, 0.035),
      readout: readoutSigns[index] * randomBetween(random, 0.42, 0.76)
    };
  });
  return {
    id: `${dimension}d-noisy-seed-${seed}`,
    seed,
    dimension,
    bias: randomBetween(random, -0.16, 0.16),
    base,
    hidden
  };
}

function cleanTeacher(teacher: Teacher, input: number[]): number {
  return teacher.bias
    + dot(teacher.base, input)
    + teacher.hidden.reduce((sum, term) => sum + term.readout * Math.max(0, dot(term.normal, input) + term.offset), 0);
}

function makeNoisyCountedOracle(
  teacher: Teacher,
  noiseSeed: number,
  noiseStd: number,
  quantization: number
): { oracle: (input: number[]) => number; getCalls: () => number } {
  const random = makeRandom(noiseSeed);
  let calls = 0;
  return {
    oracle: (input: number[]) => {
      calls += 1;
      const noisy = cleanTeacher(teacher, input) + gaussian(random) * noiseStd;
      return quantization > 0 ? Math.round(noisy / quantization) * quantization : noisy;
    },
    getCalls: () => calls
  };
}

function makeTestInputs(seed: number, count: number, ranges: Array<[number, number]>): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => ranges.map(([min, max]) => randomBetween(random, min, max)));
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
    if (Math.abs(divisor) < 1e-12) throw new Error('linear baseline matrix is singular');
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[n]);
}

function fitLinearBaseline(inputs: number[][], targets: number[]): { coefficients: number[]; predicted: number[] } {
  const width = 1 + inputs[0].length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (let i = 0; i < inputs.length; i++) {
    const row = [1, ...inputs[i]];
    for (let a = 0; a < width; a++) {
      xty[a] += row[a] * targets[i];
      for (let b = 0; b < width; b++) xtx[a][b] += row[a] * row[b];
    }
  }
  const coefficients = solveSystem(xtx, xty);
  return {
    coefficients,
    predicted: inputs.map(input => coefficients[0] + dot(coefficients.slice(1), input))
  };
}

function canonicalPlane(term: { normal: number[]; offset: number }): { normal: number[]; offset: number } {
  const scale = norm(term.normal);
  let normal = term.normal.map(value => value / scale);
  let offset = term.offset / Math.max(scale, 1e-12);
  const signIndex = normal.findIndex(value => Math.abs(value) > 1e-10);
  if (signIndex >= 0 && normal[signIndex] < 0) {
    normal = normal.map(value => -value);
    offset = -offset;
  }
  return { normal, offset };
}

function greedyPlaneError(expectedTerms: TeacherTerm[], recoveredTerms: Array<{ normal: number[]; offset: number }>): number {
  const expected = expectedTerms.map(canonicalPlane);
  const recovered = recoveredTerms.map(canonicalPlane);
  const used = new Set<number>();
  let maxError = 0;
  for (const expectedPlane of expected) {
    let bestIndex = -1;
    let bestError = Infinity;
    for (let i = 0; i < recovered.length; i++) {
      if (used.has(i)) continue;
      const recoveredPlane = recovered[i];
      const directNormal = Math.sqrt(expectedPlane.normal.reduce((sum, value, index) => sum + (value - recoveredPlane.normal[index]) ** 2, 0));
      const directOffset = Math.abs(expectedPlane.offset - recoveredPlane.offset);
      const flippedNormal = Math.sqrt(expectedPlane.normal.reduce((sum, value, index) => sum + (value + recoveredPlane.normal[index]) ** 2, 0));
      const flippedOffset = Math.abs(expectedPlane.offset + recoveredPlane.offset);
      const error = Math.min(directNormal + directOffset, flippedNormal + flippedOffset);
      if (error < bestError) {
        bestError = error;
        bestIndex = i;
      }
    }
    if (bestIndex >= 0) used.add(bestIndex);
    maxError = Math.max(maxError, bestError);
  }
  return maxError;
}

function runTeacher(teacher: Teacher): RunResult {
  const ranges = makeRanges(teacher.dimension);
  const noiseStd = Number(process.env.GCL_NOISY_ND_LINE_PROBE_NOISE_STD ?? 0.0015);
  const quantization = Number(process.env.GCL_NOISY_ND_LINE_PROBE_QUANTIZATION ?? 0.0005);
  const querySamples = Math.max(3, Number(process.env.GCL_NOISY_ND_LINE_PROBE_QUERY_SAMPLES ?? 11));
  const counted = makeNoisyCountedOracle(teacher, teacher.seed + 920001, noiseStd, quantization);
  const lineCount = teacher.dimension === 4
    ? Number(process.env.GCL_NOISY_ND_LINE_PROBE_4D_LINES ?? 190)
    : Number(process.env.GCL_NOISY_ND_LINE_PROBE_5D_LINES ?? 280);
  const samplesPerLine = teacher.dimension === 4
    ? Number(process.env.GCL_NOISY_ND_LINE_PROBE_4D_SAMPLES ?? 241)
    : Number(process.env.GCL_NOISY_ND_LINE_PROBE_5D_SAMPLES ?? 281);
  const ransacIterations = teacher.dimension === 4
    ? Number(process.env.GCL_NOISY_ND_LINE_PROBE_4D_RANSAC ?? 2600)
    : Number(process.env.GCL_NOISY_ND_LINE_PROBE_5D_RANSAC ?? 4200);
  const minSupport = teacher.dimension === 4
    ? Math.max(28, Math.floor(lineCount * 0.13))
    : Math.max(32, Math.floor(lineCount * 0.11));
  const minSegmentPoints = teacher.dimension === 4 ? 14 : 16;
  const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
    dimension: teacher.dimension,
    ranges,
    termCount: teacher.hidden.length,
    lineCount,
    samplesPerLine,
    querySamples,
    aggregation: 'median',
    minSegmentPoints,
    minAbsLineDeltaSlope: Number(process.env.GCL_NOISY_ND_LINE_PROBE_MIN_DELTA ?? 0.06),
    randomSeed: teacher.seed + 930001,
    ransacIterations,
    inlierTolerance: 0.035,
    minSupport,
    ridge: 1e-8
  });
  const testInputs = makeTestInputs(teacher.seed + 940001, 3072, ranges);
  const targets = testInputs.map(input => cleanTeacher(teacher, input));
  const predicted = predictBlackBoxReluNdLineProbeBatch(model, testInputs);
  const gclMetrics = regressionMetrics(targets, predicted);
  const linear = fitLinearBaseline(testInputs, targets);
  const linearMetrics = regressionMetrics(targets, linear.predicted);
  const maxPlaneError = greedyPlaneError(teacher.hidden, model.terms);
  const rawOracleCalls = counted.getCalls();
  const gates = {
    recoveredAllHyperplanes: model.terms.length === teacher.hidden.length,
    maxPlaneErrorBelow006: maxPlaneError < 0.06,
    maeBelow0025: gclMetrics.mae < 0.025,
    r2Above0998: gclMetrics.r2 > 0.998,
    beatsLinearBy75Percent: gclMetrics.mae < linearMetrics.mae * 0.25,
    supportAboveMinimum: model.terms.every(term => term.support >= minSupport),
    queryBudgetBelowLimit: rawOracleCalls <= (teacher.dimension === 4 ? 720000 : 1050000),
    maxPlaneError
  };
  return {
    teacherId: teacher.id,
    dimension: teacher.dimension,
    pass: Object.values(gates).every(value => value === true || typeof value === 'number'),
    teacher,
    noise: { noiseStd, quantization, querySamples, aggregation: 'median' },
    config: {
      ranges,
      lineCount,
      samplesPerLine,
      ransacIterations,
      minSupport,
      minSegmentPoints,
      testRows: testInputs.length
    },
    baselines: {
      linearHyperplane: {
        coefficients: linear.coefficients,
        metrics: linearMetrics
      }
    },
    gclLineProbe: {
      expression: model.expression,
      baseIntercept: model.baseIntercept,
      baseWeights: model.baseWeights,
      terms: model.terms,
      kinkPointCount: model.kinkPointCount,
      metrics: gclMetrics
    },
    queryBudget: {
      rawOracleCalls,
      aggregatedPointEvaluations: Math.floor(rawOracleCalls / querySamples),
      testRows: testInputs.length
    },
    gates
  };
}

function renderMarkdown(report: { generatedAt: string; summary: Record<string, unknown>; results: RunResult[] }): string {
  return [
    '# Noisy 4D/5D Black-Box ReLU Line-Probe Sweep',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    `Pass: ${report.summary.pass}`,
    '',
    '## Summary',
    '',
    `- Teachers: ${report.summary.passedTeachers}/${report.summary.teacherCount}`,
    `- Dimensions: ${(report.summary.dimensions as number[]).join(', ')}`,
    `- Max MAE: ${report.summary.maxMae}`,
    `- Min R2: ${report.summary.minR2}`,
    `- Max plane error: ${report.summary.maxPlaneError}`,
    `- Total raw oracle calls: ${report.summary.totalRawOracleCalls}`,
    '',
    '## Results',
    '',
    '| Teacher | Dim | Pass | MAE | R2 | Max plane error | Raw oracle calls |',
    '|---|---:|---|---:|---:|---:|---:|',
    ...report.results.map(result => `| ${result.teacherId} | ${result.dimension} | ${result.pass ? 'yes' : 'no'} | ${result.gclLineProbe.metrics.mae.toExponential(6)} | ${result.gclLineProbe.metrics.r2.toFixed(12)} | ${result.gates.maxPlaneError.toExponential(6)} | ${result.queryBudget.rawOracleCalls} |`),
    ''
  ].join('\n');
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teachers = [
  makeTeacher(101, 4),
  makeTeacher(2026, 4),
  makeTeacher(7777, 5)
];
const results = teachers.map(runTeacher);
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  maxMae: Math.max(...results.map(result => result.gclLineProbe.metrics.mae)),
  minR2: Math.min(...results.map(result => result.gclLineProbe.metrics.r2)),
  maxLinearMae: Math.max(...results.map(result => result.baselines.linearHyperplane.metrics.mae)),
  maxPlaneError: Math.max(...results.map(result => result.gates.maxPlaneError)),
  minKinkPointCount: Math.min(...results.map(result => result.gclLineProbe.kinkPointCount)),
  minTermSupport: Math.min(...results.flatMap(result => result.gclLineProbe.terms.map(term => term.support))),
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0),
  maxRawOracleCallsPerTeacher: Math.max(...results.map(result => result.queryBudget.rawOracleCalls))
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy randomized 4D/5D black-box ReLU random-direction line-probe hyperplane stitching',
  blackBox: {
    type: 'deterministic randomized 4D/5D shallow ReLU teachers with stochastic output noise and quantization',
    visibilityToGcl: 'oracle calls only: the solver receives repeated noisy f(vector), not hidden activations or teacher weights'
  },
  claimScope: {
    proves: 'noise-tolerant oracle-only random-direction line-probe hyperplane stitching across a deterministic 4D/5D seed sweep',
    doesNotClaim: 'deep ReLU composition recovery, adversarial weak-support ND recovery, or arbitrary production model extraction'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, renderMarkdown(report));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-noisy-blackbox-relu-nd-line-probe',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark extends the ND random-direction line-probe lane to repeated noisy and quantized oracle calls.',
    'The solver receives only f(vector) samples; no teacher weights or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
