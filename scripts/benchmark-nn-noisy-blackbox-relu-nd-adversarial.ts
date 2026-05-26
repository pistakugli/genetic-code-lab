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
  profile: string;
  seed: number;
  dimension: 4 | 5;
  bias: number;
  base: number[];
  hidden: TeacherTerm[];
  config: {
    lineCount: number;
    samplesPerLine: number;
    querySamples: number;
    minAbsLineDeltaSlope: number;
    minSegmentPoints: number;
    ransacIterations: number;
    inlierTolerance: number;
    minSupport: number;
  };
};

type RunResult = {
  teacherId: string;
  profile: string;
  dimension: 4 | 5;
  pass: boolean;
  teacher: Teacher;
  noise: {
    noiseStd: number;
    quantization: number;
    querySamples: number;
    aggregation: 'median';
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
    maxPlaneErrorBelow012: boolean;
    maeBelow0035: boolean;
    r2Above0995: boolean;
    beatsLinearBy65Percent: boolean;
    supportAboveMinimum: boolean;
    queryBudgetBelowLimit: boolean;
    maxPlaneError: number;
  };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu-nd-adversarial', 'generated-sweep');

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

function add(a: number[], b: number[], scale = 1): number[] {
  return a.map((value, index) => value + b[index] * scale);
}

function orthogonal(base: number[], raw: number[]): number[] {
  const projection = dot(base, raw);
  const residual = raw.map((value, index) => value - projection * base[index]);
  if (norm(residual) <= 1e-10) {
    residual[0] += 0.37;
    residual[residual.length - 1] -= 0.19;
  }
  return normalize(residual);
}

function makeRanges(dimension: number): Array<[number, number]> {
  return Array.from({ length: dimension }, (_, index) => {
    const span = 2.55 - index * 0.14;
    return [-span, span] as [number, number];
  });
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
  let normal = term.normal.map(value => value / Math.max(scale, 1e-12));
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
  let maxError = recovered.length === expected.length ? 0 : Infinity;
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

function baseConfig(dimension: 4 | 5): Teacher['config'] {
  return dimension === 4
    ? {
      lineCount: 230,
      samplesPerLine: 241,
      querySamples: 11,
      minAbsLineDeltaSlope: 0.045,
      minSegmentPoints: 14,
      ransacIterations: 3200,
      inlierTolerance: 0.045,
      minSupport: 32
    }
    : {
      lineCount: 360,
      samplesPerLine: 281,
      querySamples: 11,
      minAbsLineDeltaSlope: 0.040,
      minSegmentPoints: 16,
      ransacIterations: 5800,
      inlierTolerance: 0.050,
      minSupport: 38
    };
}

function makeNearParallel4D(seed: number): Teacher {
  const base = normalize([0.68, -0.31, 0.23, 0.14]);
  const directions = [
    orthogonal(base, [0.17, 0.54, -0.18, 0.29]),
    orthogonal(base, [-0.22, 0.18, 0.58, -0.31]),
    orthogonal(base, [0.31, -0.11, 0.24, 0.66]),
    orthogonal(base, [-0.44, -0.08, 0.12, 0.37])
  ];
  const eps = [0, 0.105, -0.115, 0.15];
  return {
    id: '4d-generated-near-parallel',
    profile: 'near-parallel',
    seed,
    dimension: 4,
    bias: 0.07,
    base: [0.08, -0.045, 0.055, -0.035],
    hidden: eps.map((value, index) => ({
      normal: normalize(add(base, directions[index], value)),
      offset: [-0.86, -0.22, 0.34, 0.92][index],
      readout: [0.62, -0.47, 0.38, -0.30][index]
    })),
    config: baseConfig(4)
  };
}

function makeOffsetCollision4D(seed: number): Teacher {
  const normals = [
    normalize([0.58, -0.45, 0.28, 0.19]),
    normalize([0.46, 0.38, -0.41, 0.25]),
    normalize([0.52, 0.12, 0.49, -0.33]),
    normalize([0.37, -0.28, -0.22, 0.65])
  ];
  return {
    id: '4d-generated-offset-collision',
    profile: 'offset-collision',
    seed,
    dimension: 4,
    bias: -0.05,
    base: [0.06, 0.04, -0.035, 0.025],
    hidden: normals.map((normal, index) => ({
      normal,
      offset: [-0.44, -0.405, 0.43, 0.88][index],
      readout: [0.68, -0.42, 0.35, -0.27][index]
    })),
    config: { ...baseConfig(4), lineCount: 250, ransacIterations: 3600 }
  };
}

function makeLowAmplitude5D(seed: number): Teacher {
  const normals = [
    normalize([0.50, -0.22, 0.31, -0.14, 0.37]),
    normalize([0.42, 0.36, -0.33, 0.25, -0.21]),
    normalize([0.34, -0.44, -0.15, 0.41, 0.28]),
    normalize([0.28, 0.18, 0.47, 0.36, -0.31])
  ];
  return {
    id: '5d-generated-low-amplitude',
    profile: 'low-amplitude',
    seed,
    dimension: 5,
    bias: 0.04,
    base: [0.045, -0.035, 0.03, -0.025, 0.02],
    hidden: normals.map((normal, index) => ({
      normal,
      offset: [-0.72, -0.12, 0.52, 0.98][index],
      readout: [0.44, -0.24, 0.19, -0.17][index]
    })),
    config: { ...baseConfig(5), lineCount: 390, minAbsLineDeltaSlope: 0.028, minSupport: 36 }
  };
}

function makeObliqueWeakSupport5D(seed: number): Teacher {
  const random = makeRandom(seed);
  const hidden: TeacherTerm[] = [];
  const raw = [
    [0.21, 0.62, -0.38, 0.32, -0.18],
    [0.24, -0.54, -0.30, 0.43, 0.26],
    [0.18, 0.27, 0.56, -0.42, 0.29],
    [0.26, -0.24, 0.35, 0.52, -0.41]
  ];
  for (let i = 0; i < raw.length; i++) {
    hidden.push({
      normal: normalize(raw[i].map(value => value + randomBetween(random, -0.025, 0.025))),
      offset: [-1.02, -0.58, 0.74, 1.08][i],
      readout: [0.58, -0.36, 0.29, -0.21][i]
    });
  }
  return {
    id: '5d-generated-oblique-weak-support',
    profile: 'oblique-weak-support',
    seed,
    dimension: 5,
    bias: -0.035,
    base: [0.035, 0.025, -0.03, 0.02, -0.018],
    hidden,
    config: { ...baseConfig(5), lineCount: 410, ransacIterations: 6800, minAbsLineDeltaSlope: 0.032, minSupport: 36 }
  };
}

function runTeacher(teacher: Teacher): RunResult {
  const ranges = makeRanges(teacher.dimension);
  const noiseStd = Number(process.env.GCL_NOISY_ND_ADVERSARIAL_NOISE_STD ?? 0.0015);
  const quantization = Number(process.env.GCL_NOISY_ND_ADVERSARIAL_QUANTIZATION ?? 0.0005);
  const counted = makeNoisyCountedOracle(teacher, teacher.seed + 19001, noiseStd, quantization);
  const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
    dimension: teacher.dimension,
    ranges,
    termCount: teacher.hidden.length,
    lineCount: teacher.config.lineCount,
    samplesPerLine: teacher.config.samplesPerLine,
    querySamples: teacher.config.querySamples,
    aggregation: 'median',
    minSegmentPoints: teacher.config.minSegmentPoints,
    minAbsLineDeltaSlope: teacher.config.minAbsLineDeltaSlope,
    randomSeed: teacher.seed + 29001,
    ransacIterations: teacher.config.ransacIterations,
    inlierTolerance: teacher.config.inlierTolerance,
    minSupport: teacher.config.minSupport,
    ridge: 1e-8
  });
  const testInputs = makeTestInputs(teacher.seed + 39001, 4096, ranges);
  const targets = testInputs.map(input => cleanTeacher(teacher, input));
  const predicted = predictBlackBoxReluNdLineProbeBatch(model, testInputs);
  const gclMetrics = regressionMetrics(targets, predicted);
  const linear = fitLinearBaseline(testInputs, targets);
  const linearMetrics = regressionMetrics(targets, linear.predicted);
  const maxPlaneError = greedyPlaneError(teacher.hidden, model.terms);
  const rawOracleCalls = counted.getCalls();
  const gates = {
    recoveredAllHyperplanes: model.terms.length === teacher.hidden.length,
    maxPlaneErrorBelow012: maxPlaneError < 0.12,
    maeBelow0035: gclMetrics.mae < 0.035,
    r2Above0995: gclMetrics.r2 > 0.995,
    beatsLinearBy65Percent: gclMetrics.mae < linearMetrics.mae * 0.35,
    supportAboveMinimum: model.terms.every(term => term.support >= teacher.config.minSupport),
    queryBudgetBelowLimit: rawOracleCalls <= (teacher.dimension === 4 ? 850000 : 1450000),
    maxPlaneError
  };
  const pass = gates.recoveredAllHyperplanes
    && gates.maxPlaneErrorBelow012
    && gates.maeBelow0035
    && gates.r2Above0995
    && gates.beatsLinearBy65Percent
    && gates.supportAboveMinimum
    && gates.queryBudgetBelowLimit;
  return {
    teacherId: teacher.id,
    profile: teacher.profile,
    dimension: teacher.dimension,
    pass,
    teacher,
    noise: { noiseStd, quantization, querySamples: teacher.config.querySamples, aggregation: 'median' },
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
      aggregatedPointEvaluations: Math.floor(rawOracleCalls / teacher.config.querySamples),
      testRows: testInputs.length
    },
    gates
  };
}

function renderMarkdown(report: { generatedAt: string; summary: Record<string, unknown>; results: RunResult[] }): string {
  return [
    '# Generated Noisy 4D/5D Adversarial ND ReLU Line-Probe Sweep',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    `Pass: ${report.summary.pass}`,
    '',
    '## Summary',
    '',
    `- Teachers: ${report.summary.passedTeachers}/${report.summary.teacherCount}`,
    `- Profiles: ${(report.summary.profiles as string[]).join(', ')}`,
    `- Dimensions: ${(report.summary.dimensions as number[]).join(', ')}`,
    `- Max MAE: ${report.summary.maxMae}`,
    `- Min R2: ${report.summary.minR2}`,
    `- Max plane error: ${report.summary.maxPlaneError}`,
    `- Total raw oracle calls: ${report.summary.totalRawOracleCalls}`,
    '',
    '## Results',
    '',
    '| Teacher | Profile | Dim | Pass | MAE | R2 | Linear MAE | Max plane error | Raw oracle calls |',
    '|---|---|---:|---|---:|---:|---:|---:|---:|',
    ...report.results.map(result => `| ${result.teacherId} | ${result.profile} | ${result.dimension} | ${result.pass ? 'yes' : 'no'} | ${result.gclLineProbe.metrics.mae.toExponential(6)} | ${result.gclLineProbe.metrics.r2.toFixed(12)} | ${result.baselines.linearHyperplane.metrics.mae.toExponential(6)} | ${result.gates.maxPlaneError.toExponential(6)} | ${result.queryBudget.rawOracleCalls} |`),
    ''
  ].join('\n');
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teachers = [
  makeNearParallel4D(260601),
  makeOffsetCollision4D(260602),
  makeLowAmplitude5D(260603),
  makeObliqueWeakSupport5D(260604)
];
const results = teachers.map(runTeacher);
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  profiles: results.map(result => result.profile),
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
  task: 'generated adversarial noisy 4D/5D black-box ReLU ND line-probe hyperplane stitching',
  blackBox: {
    type: 'generated near-parallel, offset-collision, low-amplitude, and oblique weak-support 4D/5D shallow ReLU teachers with stochastic output noise and quantization',
    visibilityToGcl: 'oracle calls only: the solver receives repeated noisy f(vector), not hidden activations or teacher weights'
  },
  claimScope: {
    proves: 'noise-tolerant oracle-only ND line-probe recovery across generated adversarial shallow 4D/5D profiles',
    doesNotClaim: 'deep ReLU composition recovery, unique recovery under exact hidden-unit cancellation, or arbitrary production model extraction'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, renderMarkdown(report));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-noisy-blackbox-relu-nd-adversarial',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark broadens noisy ND line-probe proof from a fixed seed sweep to generated adversarial shallow-ReLU profiles.',
    'The solver receives only repeated noisy f(vector) samples; no teacher weights or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
