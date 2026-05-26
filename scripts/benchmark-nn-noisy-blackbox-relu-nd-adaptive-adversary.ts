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

type DifficultyScore = {
  score: number;
  minNormalDistance: number;
  minOffsetDistance: number;
  minExpectedLineDeltaSlope: number;
  minSupportFraction: number;
  maxBoundaryPressure: number;
  cancellationRisk: number;
  reasons: string[];
};

type RunResult = {
  teacherId: string;
  profile: string;
  dimension: 4 | 5;
  pass: boolean;
  teacher: Teacher;
  difficulty: DifficultyScore;
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
  gclLineProbe: null | {
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
  error?: string;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu-nd-adaptive-adversary', 'adaptive-search');

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

function randomNormal(random: () => number, dimension: number): number[] {
  return normalize(Array.from({ length: dimension }, () => gaussian(random)));
}

function jitteredNormal(random: () => number, base: number[], severity: number): number[] {
  const raw = Array.from({ length: base.length }, () => gaussian(random));
  const direction = orthogonal(base, raw);
  return normalize(add(base, direction, severity));
}

function estimateSupport(term: TeacherTerm, ranges: Array<[number, number]>, seed: number): number {
  const random = makeRandom(seed);
  const count = 512;
  let active = 0;
  for (let i = 0; i < count; i++) {
    const input = ranges.map(([min, max]) => randomBetween(random, min, max));
    if (dot(term.normal, input) + term.offset > 0) active += 1;
  }
  return active / count;
}

function scoreTeacherDifficulty(teacher: Teacher): DifficultyScore {
  const ranges = makeRanges(teacher.dimension);
  const supports = teacher.hidden.map((term, index) => estimateSupport(term, ranges, teacher.seed + index * 101));
  let minNormalDistance = Infinity;
  let minOffsetDistance = Infinity;
  let cancellationRisk = 0;
  for (let i = 0; i < teacher.hidden.length; i++) {
    for (let j = i + 1; j < teacher.hidden.length; j++) {
      const normalDistance = Math.min(
        norm(teacher.hidden[i].normal.map((value, index) => value - teacher.hidden[j].normal[index])),
        norm(teacher.hidden[i].normal.map((value, index) => value + teacher.hidden[j].normal[index]))
      );
      minNormalDistance = Math.min(minNormalDistance, normalDistance);
      minOffsetDistance = Math.min(minOffsetDistance, Math.abs(teacher.hidden[i].offset - teacher.hidden[j].offset));
      if (normalDistance < 0.28 && Math.sign(teacher.hidden[i].readout) !== Math.sign(teacher.hidden[j].readout)) {
        cancellationRisk += (0.28 - normalDistance) * Math.min(Math.abs(teacher.hidden[i].readout), Math.abs(teacher.hidden[j].readout));
      }
    }
  }
  const minExpectedLineDeltaSlope = Math.min(...teacher.hidden.map(term => Math.abs(term.readout) / Math.sqrt(teacher.dimension)));
  const minSupportFraction = Math.min(...supports);
  const maxBoundaryPressure = Math.max(...supports.map(value => Math.max(0, 0.32 - value, value - 0.78)));
  const reasons: string[] = [];
  if (minNormalDistance < 0.22) reasons.push('near-parallel normals');
  if (minOffsetDistance < 0.08) reasons.push('offset collision');
  if (minExpectedLineDeltaSlope < 0.13) reasons.push('weak expected line kink');
  if (minSupportFraction < 0.22) reasons.push('weak domain support');
  if (cancellationRisk > 0.01) reasons.push('opposite-sign cancellation risk');
  const score =
    2.2 / Math.max(0.08, minNormalDistance)
    + 0.9 / Math.max(0.04, minOffsetDistance)
    + 0.7 / Math.max(0.08, minExpectedLineDeltaSlope)
    + 1.1 / Math.max(0.12, minSupportFraction)
    + 3.5 * maxBoundaryPressure
    + 5.0 * cancellationRisk;
  return {
    score,
    minNormalDistance,
    minOffsetDistance,
    minExpectedLineDeltaSlope,
    minSupportFraction,
    maxBoundaryPressure,
    cancellationRisk,
    reasons: reasons.length > 0 ? reasons : ['mixed generated stress']
  };
}

function makeAdaptiveCandidate(seed: number, dimension: 4 | 5, profile: string): Teacher {
  const random = makeRandom(seed);
  const termCount = 4;
  const baseNormal = randomNormal(random, dimension);
  const hidden: TeacherTerm[] = [];
  const offsetCenter = randomBetween(random, -0.42, 0.42);
  const config = baseConfig(dimension);
  for (let i = 0; i < termCount; i++) {
    let normal = randomNormal(random, dimension);
    let offset = randomBetween(random, -0.95, 0.95);
    let readout = randomBetween(random, 0.24, 0.68) * (i % 2 === 0 ? 1 : -1);

    if (profile === 'near-parallel') {
      normal = jitteredNormal(random, baseNormal, randomBetween(random, 0.07, 0.18));
      offset = -0.78 + i * randomBetween(random, 0.30, 0.48);
    } else if (profile === 'offset-collision') {
      offset = offsetCenter + (i < 2 ? randomBetween(random, -0.035, 0.035) : randomBetween(random, 0.36, 0.86));
    } else if (profile === 'low-amplitude') {
      readout = randomBetween(random, 0.17, 0.34) * (i % 2 === 0 ? 1 : -1);
      config.minAbsLineDeltaSlope = dimension === 4 ? 0.025 : 0.023;
      if (i === 0) config.lineCount += dimension === 4 ? 35 : 45;
    } else if (profile === 'edge-support') {
      offset = (i % 2 === 0 ? 1 : -1) * randomBetween(random, 0.88, 1.22);
      if (i === 0) config.lineCount += dimension === 4 ? 45 : 60;
      config.minSupport = Math.max(dimension + 4, config.minSupport - 4);
    } else {
      normal = i < 2 ? jitteredNormal(random, baseNormal, randomBetween(random, 0.10, 0.24)) : normal;
      offset = i < 2 ? offsetCenter + randomBetween(random, -0.055, 0.055) : offset;
      readout = randomBetween(random, 0.21, 0.55) * (i % 2 === 0 ? 1 : -1);
      config.minAbsLineDeltaSlope = dimension === 4 ? 0.030 : 0.028;
      if (i === 0) config.lineCount += dimension === 4 ? 30 : 50;
    }
    hidden.push({ normal, offset, readout });
  }
  if (dimension === 5) {
    config.ransacIterations += 700;
    config.inlierTolerance = Math.max(config.inlierTolerance, 0.055);
  }
  return {
    id: `${dimension}d-adaptive-${profile}-${seed}`,
    profile,
    seed,
    dimension,
    bias: randomBetween(random, -0.08, 0.08),
    base: Array.from({ length: dimension }, () => randomBetween(random, -0.055, 0.055)),
    hidden,
    config
  };
}

function buildAdaptivePool(): Array<{ teacher: Teacher; difficulty: DifficultyScore }> {
  const poolSize = Math.max(10, Math.floor(Number(process.env.GCL_ADAPTIVE_ND_POOL_SIZE ?? 24)));
  const profiles = ['near-parallel', 'offset-collision', 'low-amplitude', 'edge-support', 'mixed'];
  const candidates: Array<{ teacher: Teacher; difficulty: DifficultyScore }> = [];
  for (let i = 0; i < poolSize; i++) {
    const dimension = (i % 3 === 2 ? 5 : 4) as 4 | 5;
    const profile = profiles[i % profiles.length];
    const teacher = makeAdaptiveCandidate(270000 + i * 37, dimension, profile);
    const difficulty = scoreTeacherDifficulty(teacher);
    candidates.push({ teacher, difficulty });
  }
  return candidates.sort((a, b) => b.difficulty.score - a.difficulty.score);
}

function runTeacher(teacher: Teacher, difficulty: DifficultyScore): RunResult {
  const ranges = makeRanges(teacher.dimension);
  const noiseStd = Number(process.env.GCL_NOISY_ND_ADAPTIVE_NOISE_STD ?? 0.0015);
  const quantization = Number(process.env.GCL_NOISY_ND_ADAPTIVE_QUANTIZATION ?? 0.0005);
  const counted = makeNoisyCountedOracle(teacher, teacher.seed + 19001, noiseStd, quantization);
  const testInputs = makeTestInputs(teacher.seed + 39001, 4096, ranges);
  const targets = testInputs.map(input => cleanTeacher(teacher, input));
  const linear = fitLinearBaseline(testInputs, targets);
  const linearMetrics = regressionMetrics(targets, linear.predicted);
  let model: ReturnType<typeof decompileBlackBoxReluNdLineProbe>;
  try {
    model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
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
  } catch (error) {
    const rawOracleCalls = counted.getCalls();
    return {
      teacherId: teacher.id,
      profile: teacher.profile,
      dimension: teacher.dimension,
      pass: false,
      teacher,
      difficulty,
      noise: { noiseStd, quantization, querySamples: teacher.config.querySamples, aggregation: 'median' },
      baselines: {
        linearHyperplane: {
          coefficients: linear.coefficients,
          metrics: linearMetrics
        }
      },
      gclLineProbe: null,
      queryBudget: {
        rawOracleCalls,
        aggregatedPointEvaluations: Math.floor(rawOracleCalls / teacher.config.querySamples),
        testRows: testInputs.length
      },
      gates: {
        recoveredAllHyperplanes: false,
        maxPlaneErrorBelow012: false,
        maeBelow0035: false,
        r2Above0995: false,
        beatsLinearBy65Percent: false,
        supportAboveMinimum: false,
        queryBudgetBelowLimit: rawOracleCalls <= (teacher.dimension === 4 ? 950000 : 1650000),
        maxPlaneError: Infinity
      },
      error: error instanceof Error ? error.message : String(error)
    };
  }
  const predicted = predictBlackBoxReluNdLineProbeBatch(model, testInputs);
  const gclMetrics = regressionMetrics(targets, predicted);
  const maxPlaneError = greedyPlaneError(teacher.hidden, model.terms);
  const rawOracleCalls = counted.getCalls();
  const gates = {
    recoveredAllHyperplanes: model.terms.length === teacher.hidden.length,
    maxPlaneErrorBelow012: maxPlaneError < 0.12,
    maeBelow0035: gclMetrics.mae < 0.035,
    r2Above0995: gclMetrics.r2 > 0.995,
    beatsLinearBy65Percent: gclMetrics.mae < linearMetrics.mae * 0.35,
    supportAboveMinimum: model.terms.every(term => term.support >= teacher.config.minSupport),
    queryBudgetBelowLimit: rawOracleCalls <= (teacher.dimension === 4 ? 950000 : 1650000),
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
    difficulty,
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

function renderMarkdown(report: { generatedAt: string; summary: Record<string, unknown>; adaptiveSearch: Record<string, unknown>; results: RunResult[] }): string {
  return [
    '# Adaptive Noisy 4D/5D Adversarial ND ReLU Line-Probe Sweep',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    `Pass: ${report.summary.pass}`,
    '',
    '## Summary',
    '',
    `- Teachers: ${report.summary.passedTeachers}/${report.summary.teacherCount}`,
    `- Candidate pool size: ${report.adaptiveSearch.candidatePoolSize}`,
    `- Selected hard teachers: ${report.adaptiveSearch.selectedTeacherCount}`,
    `- Profiles: ${(report.summary.profiles as string[]).join(', ')}`,
    `- Dimensions: ${(report.summary.dimensions as number[]).join(', ')}`,
    `- Max MAE: ${report.summary.maxMae}`,
    `- Min R2: ${report.summary.minR2}`,
    `- Max plane error: ${report.summary.maxPlaneError}`,
    `- Total raw oracle calls: ${report.summary.totalRawOracleCalls}`,
    '',
    '## Results',
    '',
    '| Teacher | Profile | Dim | Difficulty | Reasons | Pass | MAE | R2 | Linear MAE | Max plane error | Raw oracle calls | Error |',
    '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|',
    ...report.results.map(result => {
      const mae = result.gclLineProbe ? result.gclLineProbe.metrics.mae.toExponential(6) : '-';
      const r2 = result.gclLineProbe ? result.gclLineProbe.metrics.r2.toFixed(12) : '-';
      const planeError = Number.isFinite(result.gates.maxPlaneError) ? result.gates.maxPlaneError.toExponential(6) : 'inf';
      return `| ${result.teacherId} | ${result.profile} | ${result.dimension} | ${result.difficulty.score.toFixed(3)} | ${result.difficulty.reasons.join(', ')} | ${result.pass ? 'yes' : 'no'} | ${mae} | ${r2} | ${result.baselines.linearHyperplane.metrics.mae.toExponential(6)} | ${planeError} | ${result.queryBudget.rawOracleCalls} | ${result.error ?? ''} |`;
    }),
    ''
  ].join('\n');
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const candidatePool = buildAdaptivePool();
const selectedTeacherCount = Math.max(3, Math.floor(Number(process.env.GCL_ADAPTIVE_ND_SELECTED_COUNT ?? 4)));
const selected = candidatePool.slice(0, selectedTeacherCount);
const results = selected.map(candidate => runTeacher(candidate.teacher, candidate.difficulty));
const successfulResults = results.filter(result => result.gclLineProbe);
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  profiles: results.map(result => result.profile),
  dimensions: [...new Set(results.map(result => result.dimension))],
  maxDifficultyScore: Math.max(...results.map(result => result.difficulty.score)),
  minDifficultyScore: Math.min(...results.map(result => result.difficulty.score)),
  maxMae: successfulResults.length > 0 ? Math.max(...successfulResults.map(result => result.gclLineProbe!.metrics.mae)) : Infinity,
  minR2: successfulResults.length > 0 ? Math.min(...successfulResults.map(result => result.gclLineProbe!.metrics.r2)) : -Infinity,
  maxLinearMae: Math.max(...results.map(result => result.baselines.linearHyperplane.metrics.mae)),
  maxPlaneError: Math.max(...results.map(result => result.gates.maxPlaneError)),
  minKinkPointCount: successfulResults.length > 0 ? Math.min(...successfulResults.map(result => result.gclLineProbe!.kinkPointCount)) : 0,
  minTermSupport: successfulResults.length > 0 ? Math.min(...successfulResults.flatMap(result => result.gclLineProbe!.terms.map(term => term.support))) : 0,
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0),
  maxRawOracleCallsPerTeacher: Math.max(...results.map(result => result.queryBudget.rawOracleCalls))
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'adaptive adversarial noisy 4D/5D black-box ReLU ND line-probe hyperplane stitching',
  adaptiveSearch: {
    candidatePoolSize: candidatePool.length,
    selectedTeacherCount: selected.length,
    scoring: 'select hardest generated teachers by normal closeness, offset collision, weak expected line delta slope, weak domain support, boundary pressure, and cancellation risk',
    topCandidates: candidatePool.slice(0, Math.min(10, candidatePool.length)).map(candidate => ({
      teacherId: candidate.teacher.id,
      profile: candidate.teacher.profile,
      dimension: candidate.teacher.dimension,
      difficulty: candidate.difficulty
    }))
  },
  blackBox: {
    type: 'adaptive-search-selected near-parallel, offset-collision, low-amplitude, edge-support, and mixed 4D/5D shallow ReLU teachers with stochastic output noise and quantization',
    visibilityToGcl: 'oracle calls only: the solver receives repeated noisy f(vector), not hidden activations or teacher weights'
  },
  claimScope: {
    proves: 'noise-tolerant oracle-only ND line-probe recovery across adaptively selected adversarial shallow 4D/5D profiles',
    doesNotClaim: 'deep ReLU composition recovery, unique recovery under exact hidden-unit cancellation, or arbitrary production model extraction'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, renderMarkdown(report));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-noisy-blackbox-relu-nd-adaptive-adversary',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark selects hard noisy ND shallow-ReLU teachers from an adaptive adversarial candidate pool before running the solver.',
    'The solver receives only repeated noisy f(vector) samples; no teacher weights or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
