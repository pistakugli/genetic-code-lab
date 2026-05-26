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

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-blackbox-relu-nd-line-probe-randomized', '4d-5d-seed-sweep');

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

function makeRanges(dimension: number): Array<[number, number]> {
  return Array.from({ length: dimension }, (_, index) => {
    const span = 2.55 - index * 0.16;
    return [-span, span] as [number, number];
  });
}

function makeNormal(random: () => number, dimension: number, existing: number[]): number[] {
  for (let attempt = 0; attempt < 1000; attempt++) {
    const normal = normalize(Array.from({ length: dimension }, (_, index) => {
      const axisBias = index === 0 ? 0.35 : 0;
      return randomBetween(random, -1, 1) + axisBias;
    }));
    if (Math.abs(normal[0]) < 0.08) continue;
    let separated = true;
    for (let i = 0; i < existing.length; i += dimension) {
      const other = existing.slice(i, i + dimension);
      if (Math.abs(dot(normal, other)) > 0.92) {
        separated = false;
        break;
      }
    }
    if (separated) return normal;
  }
  throw new Error('failed to generate separated random normal');
}

function makeTeacher(seed: number, dimension: 4 | 5, termCount: number): Teacher {
  const random = makeRandom(seed);
  const base = Array.from({ length: dimension }, (_, index) => {
    const sign = index % 2 === 0 ? 1 : -1;
    return sign * randomBetween(random, 0.025, 0.13);
  });
  const normalBank: number[] = [];
  const hidden: TeacherTerm[] = [];
  const offsets = dimension === 4
    ? [-0.86, -0.27, 0.31, 0.92, 1.28]
    : [-0.92, -0.44, 0.08, 0.57, 1.06];
  for (let i = 0; i < termCount; i++) {
    const normal = makeNormal(random, dimension, normalBank);
    normalBank.push(...normal);
    const readoutMagnitude = randomBetween(random, 0.28, 0.78);
    hidden.push({
      normal,
      offset: offsets[i] + randomBetween(random, -0.045, 0.045),
      readout: (i % 2 === 0 ? 1 : -1) * readoutMagnitude
    });
  }
  return {
    id: `${dimension}d-seed-${seed}`,
    seed,
    dimension,
    bias: randomBetween(random, -0.2, 0.2),
    base,
    hidden
  };
}

function cleanTeacher(teacher: Teacher, input: number[]): number {
  return teacher.bias
    + dot(teacher.base, input)
    + teacher.hidden.reduce((sum, term) => sum + term.readout * Math.max(0, dot(term.normal, input) + term.offset), 0);
}

function makeCountedOracle(teacher: Teacher): { oracle: (input: number[]) => number; getCalls: () => number } {
  let calls = 0;
  return {
    oracle: (input: number[]) => {
      calls += 1;
      return cleanTeacher(teacher, input);
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

function runTeacher(teacher: Teacher): unknown {
  const ranges = makeRanges(teacher.dimension);
  const counted = makeCountedOracle(teacher);
  const lineCount = teacher.dimension === 4
    ? Number(process.env.GCL_ND_LINE_PROBE_RANDOMIZED_4D_LINES ?? 240)
    : Number(process.env.GCL_ND_LINE_PROBE_RANDOMIZED_5D_LINES ?? 380);
  const samplesPerLine = teacher.dimension === 4
    ? Number(process.env.GCL_ND_LINE_PROBE_RANDOMIZED_4D_SAMPLES ?? 801)
    : Number(process.env.GCL_ND_LINE_PROBE_RANDOMIZED_5D_SAMPLES ?? 901);
  const ransacIterations = teacher.dimension === 4
    ? Number(process.env.GCL_ND_LINE_PROBE_RANDOMIZED_4D_RANSAC ?? 3000)
    : Number(process.env.GCL_ND_LINE_PROBE_RANDOMIZED_5D_RANSAC ?? 6200);
  const minSupport = teacher.dimension === 4
    ? Math.max(38, Math.floor(lineCount * 0.16))
    : Math.max(46, Math.floor(lineCount * 0.12));
  const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
    dimension: teacher.dimension,
    ranges,
    termCount: teacher.hidden.length,
    lineCount,
    samplesPerLine,
    randomSeed: teacher.seed + 700001,
    ransacIterations,
    inlierTolerance: 3e-5,
    minSupport,
    slopeTolerance: 1e-8,
    kinkMergeTolerance: 1e-5
  });
  const testInputs = makeTestInputs(teacher.seed + 800001, 4096, ranges);
  const targets = testInputs.map(input => cleanTeacher(teacher, input));
  const predicted = predictBlackBoxReluNdLineProbeBatch(model, testInputs);
  const gclMetrics = regressionMetrics(targets, predicted);
  const linear = fitLinearBaseline(testInputs, targets);
  const linearMetrics = regressionMetrics(targets, linear.predicted);
  const maxPlaneError = greedyPlaneError(teacher.hidden, model.terms);
  const gates = {
    recoveredAllHyperplanes: model.terms.length === teacher.hidden.length,
    maxPlaneErrorBelow1eMinus3: maxPlaneError < 1e-3,
    maeBelow1eMinus8: gclMetrics.mae < 1e-8,
    r2Above999999999: gclMetrics.r2 > 0.999999999,
    beatsLinearBy1000x: gclMetrics.mae * 1000 < linearMetrics.mae,
    supportAboveMinimum: model.terms.every(term => term.support >= minSupport),
    queryBudgetBelowLimit: counted.getCalls() <= (teacher.dimension === 4 ? 260000 : 360000)
  };
  const pass = Object.values(gates).every(value => value);
  return {
    teacherId: teacher.id,
    dimension: teacher.dimension,
    pass,
    teacher,
    config: {
      ranges,
      lineCount,
      samplesPerLine,
      ransacIterations,
      minSupport,
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
      oracleCalls: counted.getCalls(),
      lineProbeCallsUpperBound: model.lineCount * model.samplesPerLine,
      fitAndValidationCalls: counted.getCalls() - model.lineCount * model.samplesPerLine,
      testRows: testInputs.length
    },
    gates: {
      ...gates,
      maxPlaneError
    }
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teachers = [
  makeTeacher(101, 4, 4),
  makeTeacher(2026, 4, 4),
  makeTeacher(4242, 4, 4),
  makeTeacher(7777, 5, 4),
  makeTeacher(31415, 5, 4)
];
const results = teachers.map(runTeacher) as Array<{
  teacherId: string;
  dimension: 4 | 5;
  pass: boolean;
  gclLineProbe: { metrics: { mae: number; r2: number }; kinkPointCount: number; terms: Array<{ support: number }> };
  baselines: { linearHyperplane: { metrics: { mae: number } } };
  queryBudget: { oracleCalls: number };
  gates: { maxPlaneError: number };
}>;
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
  totalOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.oracleCalls, 0),
  maxOracleCallsPerTeacher: Math.max(...results.map(result => result.queryBudget.oracleCalls))
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'randomized 4D/5D black-box ReLU random-direction line-probe hyperplane stitching',
  blackBox: {
    type: 'deterministic randomized 4D/5D shallow ReLU teachers',
    visibilityToGcl: 'oracle calls only: the solver receives f(vector), not hidden activations or teacher weights'
  },
  claimScope: {
    proves: 'clean oracle-only random-direction line-probe hyperplane stitching across a deterministic randomized 4D/5D seed sweep',
    doesNotClaim: 'noisy ND recovery, deep ReLU composition recovery, or arbitrary production model extraction'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Randomized 4D/5D Black-Box ReLU Line-Probe Sweep',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${summary.pass}`,
  '',
  '## Summary',
  '',
  `- Teachers: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Dimensions: ${summary.dimensions.join(', ')}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Max plane error: ${summary.maxPlaneError}`,
  `- Min kink points: ${summary.minKinkPointCount}`,
  `- Min term support: ${summary.minTermSupport}`,
  `- Total oracle calls: ${summary.totalOracleCalls}`,
  `- Max oracle calls per teacher: ${summary.maxOracleCallsPerTeacher}`,
  '',
  '## Results',
  '',
  '| Teacher | Dim | Pass | MAE | R2 | Max plane error | Oracle calls |',
  '|---|---:|---|---:|---:|---:|---:|',
  ...results.map(result => `| ${result.teacherId} | ${result.dimension} | ${result.pass ? 'yes' : 'no'} | ${result.gclLineProbe.metrics.mae.toExponential(6)} | ${result.gclLineProbe.metrics.r2.toFixed(12)} | ${result.gates.maxPlaneError.toExponential(6)} | ${result.queryBudget.oracleCalls} |`),
  ''
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-blackbox-relu-nd-line-probe-randomized',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark broadens the clean ND random-direction line-probe lane from one 4D teacher to a deterministic 4D/5D seed sweep.',
    'The solver receives only f(vector) oracle calls; no teacher weights or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
