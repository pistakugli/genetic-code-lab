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
  dimension: number;
  bias: number;
  base: number[];
  hidden: TeacherTerm[];
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-blackbox-relu-nd-line-probe', 'four-dimensional');

const teacher: Teacher = {
  id: '4d-random-direction-hyperplane-stitching',
  dimension: 4,
  bias: 0.17,
  base: [0.12, -0.08, 0.05, -0.03],
  hidden: [
    { normal: [0.92, -0.28, 0.18, 0.12], offset: 0.58, readout: 0.72 },
    { normal: [0.36, 0.81, -0.31, 0.18], offset: -0.24, readout: -0.53 },
    { normal: [-0.22, 0.16, 0.88, -0.38], offset: -0.92, readout: 0.41 },
    { normal: [0.14, -0.42, 0.25, 0.86], offset: 0.18, readout: -0.34 }
  ]
};

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

function cleanTeacher(input: number[]): number {
  return teacher.bias
    + dot(teacher.base, input)
    + teacher.hidden.reduce(
      (sum, unit) => sum + unit.readout * Math.max(0, dot(unit.normal, input) + unit.offset),
      0
    );
}

function makeCountedOracle(): { oracle: (input: number[]) => number; getCalls: () => number } {
  let calls = 0;
  return {
    oracle: (input: number[]) => {
      calls += 1;
      return cleanTeacher(input);
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
  const normalized = normalize(term.normal);
  const scale = norm(term.normal);
  let normal = [...normalized];
  let offset = term.offset / Math.max(scale, 1e-12);
  const signIndex = normal.findIndex(value => Math.abs(value) > 1e-10);
  if (signIndex >= 0 && normal[signIndex] < 0) {
    normal = normal.map(value => -value);
    offset = -offset;
  }
  return { normal, offset };
}

function maxPlaneError(): number {
  const expected = teacher.hidden.map(canonicalPlane);
  const recovered = model.terms.map(canonicalPlane);
  let maxError = 0;
  for (const expectedPlane of expected) {
    let best = Infinity;
    for (const recoveredPlane of recovered) {
      const directNormal = Math.sqrt(expectedPlane.normal.reduce((sum, value, index) => sum + (value - recoveredPlane.normal[index]) ** 2, 0));
      const directOffset = Math.abs(expectedPlane.offset - recoveredPlane.offset);
      const flippedNormal = Math.sqrt(expectedPlane.normal.reduce((sum, value, index) => sum + (value + recoveredPlane.normal[index]) ** 2, 0));
      const flippedOffset = Math.abs(expectedPlane.offset + recoveredPlane.offset);
      best = Math.min(best, directNormal + directOffset, flippedNormal + flippedOffset);
    }
    maxError = Math.max(maxError, best);
  }
  return maxError;
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const ranges: Array<[number, number]> = [
  [-2.6, 2.6],
  [-2.3, 2.3],
  [-2.1, 2.1],
  [-1.9, 1.9]
];
const counted = makeCountedOracle();
const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
  dimension: 4,
  ranges,
  termCount: teacher.hidden.length,
  lineCount: Number(process.env.GCL_ND_LINE_PROBE_LINES ?? 220),
  samplesPerLine: Number(process.env.GCL_ND_LINE_PROBE_SAMPLES ?? 901),
  randomSeed: 260529,
  ransacIterations: Number(process.env.GCL_ND_LINE_PROBE_RANSAC ?? 2600),
  inlierTolerance: 2e-5,
  minSupport: 42,
  slopeTolerance: 1e-8,
  kinkMergeTolerance: 1e-5
});

const testInputs = makeTestInputs(918273, 4096, ranges);
const targets = testInputs.map(cleanTeacher);
const predicted = predictBlackBoxReluNdLineProbeBatch(model, testInputs);
const gclMetrics = regressionMetrics(targets, predicted);
const linear = fitLinearBaseline(testInputs, targets);
const linearMetrics = regressionMetrics(targets, linear.predicted);
const planeError = maxPlaneError();
const runtimeMs = Date.now() - started;
const gates = {
  pass: false,
  recoveredAllHyperplanes: model.terms.length === teacher.hidden.length,
  maxPlaneErrorBelow1eMinus3: planeError < 1e-3,
  maeBelow1eMinus8: gclMetrics.mae < 1e-8,
  r2Above999999999: gclMetrics.r2 > 0.999999999,
  beatsLinearBy1000x: gclMetrics.mae * 1000 < linearMetrics.mae,
  supportAboveMinimum: model.terms.every(term => term.support >= 42)
};
gates.pass = Object.entries(gates)
  .filter(([key]) => key !== 'pass')
  .every(([, value]) => value === true);

const report = {
  generatedAt: new Date().toISOString(),
  task: '4D black-box ReLU random-direction line-probe hyperplane stitching',
  blackBox: {
    type: '4-input/4-hidden shallow ReLU teacher',
    visibilityToGcl: 'oracle calls only: the solver receives f(vector), not hidden activations or teacher weights'
  },
  config: {
    ranges,
    lineCount: model.lineCount,
    samplesPerLine: model.samplesPerLine,
    ransac: {
      iterations: Number(process.env.GCL_ND_LINE_PROBE_RANSAC ?? 2600),
      inlierTolerance: 2e-5,
      minSupport: 42
    },
    testRows: testInputs.length
  },
  auditTeacherCircuit: teacher,
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
    maxPlaneError: planeError
  },
  runtimeMs
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# 4D Black-Box ReLU Random-Direction Line Probe',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${gates.pass}`,
  '',
  '## Metrics',
  '',
  `- GCL MAE: ${gclMetrics.mae}`,
  `- GCL R2: ${gclMetrics.r2}`,
  `- Linear MAE: ${linearMetrics.mae}`,
  `- Max plane error: ${planeError}`,
  `- Kink points: ${model.kinkPointCount}`,
  `- Oracle calls: ${counted.getCalls()}`,
  '',
  '## Gate',
  '',
  ...Object.entries(report.gates).map(([key, value]) => `- ${key}: ${value}`),
  ''
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-blackbox-relu-nd-line-probe',
  status: gates.pass ? 'pass' : 'fail',
  summary: {
    pass: gates.pass,
    dimension: teacher.dimension,
    termCount: teacher.hidden.length,
    recoveredTerms: model.terms.length,
    mae: gclMetrics.mae,
    r2: gclMetrics.r2,
    linearMae: linearMetrics.mae,
    maxPlaneError: planeError,
    kinkPointCount: model.kinkPointCount,
    oracleCalls: counted.getCalls()
  },
  notes: [
    'This benchmark proves the first general-ND random-direction probing lane on a 4D shallow ReLU oracle.',
    'The solver receives only f(vector) oracle calls; no teacher weights or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!gates.pass) process.exit(1);
