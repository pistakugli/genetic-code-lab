#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxRelu2DRegionGraph,
  predictDeepBlackBoxRelu2DRegionGraphBatch
} from '../src/solvers/deep-blackbox-relu2d-region-graph';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type Layer1Unit = { w: [number, number]; b: number };
type LayerUnit = { w: number[]; b: number };
type AdversarialProfile =
  | 'near-parallel-fold'
  | 'offset-collision-fold'
  | 'cancellation-fold'
  | 'low-amplitude-fold'
  | 'boundary-grazing-fold'
  | 'mixed-oblique-fold';

type Teacher = {
  id: string;
  seed: number;
  profile: AdversarialProfile;
  l1: Layer1Unit[];
  l2: LayerUnit[];
  out: { w: number[]; b: number; base: [number, number] };
  adversarialKnobs: Record<string, number | string>;
};

type ProbeStage = 'screening' | 'refined';

type ProbeConfig = {
  stage: ProbeStage;
  gridWidth: number;
  gridHeight: number;
  querySamples: number;
  gradientStep: number;
  slopeTolerance: number;
  interceptTolerance: number;
  rawCallBudget: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph',
  'adaptive-search'
);

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

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function unitFromAngle(angle: number, scale: number): [number, number] {
  return [Math.cos(angle) * scale, Math.sin(angle) * scale];
}

function profileFor(index: number): AdversarialProfile {
  const profiles: AdversarialProfile[] = [
    'near-parallel-fold',
    'offset-collision-fold',
    'cancellation-fold',
    'low-amplitude-fold',
    'boundary-grazing-fold',
    'mixed-oblique-fold'
  ];
  return profiles[index % profiles.length];
}

function makeLayer1(random: () => number, profile: AdversarialProfile): { units: Layer1Unit[]; spread: number } {
  const baseAngle = randomBetween(random, -Math.PI, Math.PI);
  const scale = () => randomBetween(random, 0.8, 1.55);
  const profileOffsets: Record<AdversarialProfile, number[]> = {
    'near-parallel-fold': [-0.16, -0.055, 0.06, 0.17],
    'offset-collision-fold': [-0.7, -0.08, 0.56, 1.15],
    'cancellation-fold': [-0.95, -0.18, 0.38, 1.03],
    'low-amplitude-fold': [-1.2, -0.32, 0.44, 1.22],
    'boundary-grazing-fold': [-1.05, -0.45, 0.35, 0.92],
    'mixed-oblique-fold': [-1.28, -0.37, 0.28, 1.17]
  };
  const profileBiases: Record<AdversarialProfile, number[]> = {
    'near-parallel-fold': [-0.35, -0.12, 0.11, 0.34],
    'offset-collision-fold': [-0.08, -0.045, 0.035, 0.07],
    'cancellation-fold': [-0.62, -0.02, 0.04, 0.66],
    'low-amplitude-fold': [-0.74, -0.16, 0.2, 0.78],
    'boundary-grazing-fold': [-1.18, -0.72, 0.72, 1.18],
    'mixed-oblique-fold': [-0.5, -0.07, 0.32, 0.83]
  };
  const offsets = profileOffsets[profile];
  return {
    units: offsets.map((offset, index) => ({
      w: unitFromAngle(baseAngle + offset, scale()),
      b: profileBiases[profile][index] + randomBetween(random, -0.035, 0.035)
    })),
    spread: Math.max(...offsets) - Math.min(...offsets)
  };
}

function makeLayer2(random: () => number, profile: AdversarialProfile): LayerUnit[] {
  const rows = [
    [0.95, -0.66, 0.58, -0.52],
    [-0.78, 0.72, -0.47, 0.64],
    [0.44, -0.36, 0.92, -0.74],
    [-0.31, 0.56, -0.82, 0.88]
  ];
  const biasByProfile: Record<AdversarialProfile, number[]> = {
    'near-parallel-fold': [-0.18, -0.11, 0.08, 0.16],
    'offset-collision-fold': [-0.035, -0.015, 0.018, 0.042],
    'cancellation-fold': [-0.12, -0.105, 0.1, 0.12],
    'low-amplitude-fold': [-0.42, -0.08, 0.06, 0.38],
    'boundary-grazing-fold': [-0.58, -0.28, 0.32, 0.58],
    'mixed-oblique-fold': [-0.32, -0.03, 0.18, 0.51]
  };
  const amplitude = profile === 'low-amplitude-fold' ? 1.05 : profile === 'near-parallel-fold' ? 1.18 : profile === 'cancellation-fold' ? 0.82 : 1;
  return rows.map((row, rowIndex) => ({
    w: row.map(weight => (weight * amplitude) + randomBetween(random, -0.08, 0.08)),
    b: biasByProfile[profile][rowIndex] + randomBetween(random, -0.035, 0.035)
  }));
}

function makeTeacher(seed: number, profile: AdversarialProfile): Teacher {
  const random = makeRandom(seed);
  const layer1 = makeLayer1(random, profile);
  const l2 = makeLayer2(random, profile);
  const cancellation = profile === 'cancellation-fold' ? 0.78 : 1;
  const lowAmplitude = profile === 'low-amplitude-fold' ? 1.15 : 1;
  const outTemplates: Record<AdversarialProfile, number[]> = {
    'near-parallel-fold': [1.05, -0.86, 0.88, -0.77],
    'offset-collision-fold': [0.92, -0.8, 0.68, -0.57],
    'cancellation-fold': [0.86, -0.84 * cancellation, 0.48, -0.46 * cancellation],
    'low-amplitude-fold': [0.58 * lowAmplitude, -0.51 * lowAmplitude, 0.43 * lowAmplitude, -0.38 * lowAmplitude],
    'boundary-grazing-fold': [0.82, -0.62, 0.52, -0.46],
    'mixed-oblique-fold': [0.78, -0.66, 0.74, -0.59]
  };
  return {
    id: `adaptive-deep-2d-${profile}-${seed}`,
    seed,
    profile,
    l1: layer1.units,
    l2,
    out: {
      w: outTemplates[profile].map(weight => weight + randomBetween(random, -0.045, 0.045)),
      b: randomBetween(random, -0.12, 0.12),
      base: [randomBetween(random, -0.08, 0.08), randomBetween(random, -0.08, 0.08)]
    },
    adversarialKnobs: {
      profile,
      layer1AngleSpread: layer1.spread,
      outputScale: profile === 'low-amplitude-fold' ? lowAmplitude : 1,
      cancellationRatio: cancellation
    }
  };
}

function cleanTeacher(teacher: Teacher, x: number, y: number): number {
  const h1 = teacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b));
  const h2 = teacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
  return teacher.out.b
    + teacher.out.base[0] * x
    + teacher.out.base[1] * y
    + teacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
}

function makeNoisyOracle(
  teacher: Teacher,
  noiseSeed: number,
  noiseStd: number,
  quantization: number
): { oracle: (x: number, y: number) => number; getCalls: () => number } {
  const random = makeRandom(noiseSeed);
  let calls = 0;
  return {
    oracle: (x, y) => {
      calls++;
      const noisy = cleanTeacher(teacher, x, y) + gaussian(random) * noiseStd;
      return Math.round(noisy / quantization) * quantization;
    },
    getCalls: () => calls
  };
}

function makeGrid(width: number, height: number, xOffset: number, yOffset: number): Array<[number, number]> {
  const inputs: Array<[number, number]> = [];
  for (let xIndex = 0; xIndex < width; xIndex++) {
    for (let yIndex = 0; yIndex < height; yIndex++) {
      inputs.push([
        -1.95 + (3.9 * (xIndex + xOffset)) / width,
        -1.95 + (3.9 * (yIndex + yOffset)) / height
      ]);
    }
  }
  return inputs;
}

function readNumberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function makeProbeConfigs(): { screening: ProbeConfig; refined: ProbeConfig } {
  return {
    screening: {
      stage: 'screening',
      gridWidth: Math.max(48, Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_GRID_WIDTH', 65))),
      gridHeight: Math.max(48, Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_GRID_HEIGHT', 65))),
      querySamples: Math.max(3, Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_QUERY_SAMPLES', 15))),
      gradientStep: readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_GRADIENT_STEP', 0.085),
      slopeTolerance: readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_SLOPE_TOL', 0.16),
      interceptTolerance: readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_INTERCEPT_TOL', 0.24),
      rawCallBudget: Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_SCREENING_RAW_CALL_BUDGET', 450000))
    },
    refined: {
      stage: 'refined',
      gridWidth: Math.max(48, Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_GRID_WIDTH', 81))),
      gridHeight: Math.max(48, Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_GRID_HEIGHT', 81))),
      querySamples: Math.max(3, Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_QUERY_SAMPLES', 21))),
      gradientStep: readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_GRADIENT_STEP', 0.075),
      slopeTolerance: readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_SLOPE_TOL', 0.11),
      interceptTolerance: readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_INTERCEPT_TOL', 0.18),
      rawCallBudget: Math.floor(readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_REFINED_RAW_CALL_BUDGET', 750000))
    }
  };
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
    if (Math.abs(divisor) < 1e-12) throw new Error('baseline matrix is singular');
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

function fitBasis(
  inputs: Array<[number, number]>,
  targets: number[],
  rowFor: (input: [number, number]) => number[]
): { coefficients: number[]; predicted: number[] } {
  const width = rowFor(inputs[0]).length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  const rows = inputs.map(rowFor);
  for (let rowIndex = 0; rowIndex < inputs.length; rowIndex++) {
    const row = rows[rowIndex];
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * targets[rowIndex];
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += 1e-9;
  const coefficients = solveSystem(xtx, xty);
  return {
    coefficients,
    predicted: rows.map(row => row.reduce((sum, value, index) => sum + value * coefficients[index], 0))
  };
}

function difficultyScore(result: {
  foldingEvidence: { deepOnlyRegionCount: number; adjacencyEdgeCount: number; kinkCellBoundaryCount: number };
  gcl: { metrics: ReturnType<typeof regressionMetrics> };
  baselines: { plane: { metrics: ReturnType<typeof regressionMetrics> }; firstLayerOnly: { metrics: ReturnType<typeof regressionMetrics> } };
}): number {
  const gclMae = Math.max(1e-6, result.gcl.metrics.mae);
  const deepComplexity = Math.log1p(result.foldingEvidence.deepOnlyRegionCount) * 1.4
    + Math.log1p(result.foldingEvidence.adjacencyEdgeCount) * 0.7
    + Math.log1p(result.foldingEvidence.kinkCellBoundaryCount / 100) * 0.8;
  const baselineGap = Math.min(30, result.baselines.firstLayerOnly.metrics.mae / gclMae) * 0.15
    + Math.min(60, result.baselines.plane.metrics.mae / gclMae) * 0.05;
  const residualHardness = Math.min(1.5, result.gcl.metrics.mae * 120);
  return deepComplexity + baselineGap + residualHardness;
}

function runCandidateAttempt(teacher: Teacher, probe: ProbeConfig, noiseStd: number, quantization: number) {
  const xRange: [number, number] = [-2, 2];
  const yRange: [number, number] = [-2, 2];
  const noisy = makeNoisyOracle(teacher, teacher.seed + 260601 + (probe.stage === 'refined' ? 9109 : 0), noiseStd, quantization);
  const model = decompileDeepBlackBoxRelu2DRegionGraph(noisy.oracle, {
    xRange,
    yRange,
    gridWidth: probe.gridWidth,
    gridHeight: probe.gridHeight,
    gradientStep: probe.gradientStep,
    slopeTolerance: probe.slopeTolerance,
    interceptTolerance: probe.interceptTolerance,
    querySamples: probe.querySamples,
    aggregation: 'median'
  });
  const inputs = makeGrid(61, 57, 0.37, 0.23);
  const actual = inputs.map(([x, y]) => cleanTeacher(teacher, x, y));
  const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const plane = fitBasis(inputs, actual, ([x, y]) => [1, x, y]);
  const planeMetrics = regressionMetrics(actual, plane.predicted);
  const firstLayerOnly = fitBasis(inputs, actual, ([x, y]) => [
    1,
    x,
    y,
    ...teacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b))
  ]);
  const firstLayerOnlyMetrics = regressionMetrics(actual, firstLayerOnly.predicted);
  const deepOnlyRegionCount = Math.max(0, model.regionCount - teacher.l1.length - 1);
  const firstLayerMaeRatio = metrics.mae / Math.max(1e-12, firstLayerOnlyMetrics.mae);
  const planeMaeRatio = metrics.mae / Math.max(1e-12, planeMetrics.mae);
  const gates = {
    recoveredAtLeast8Regions: model.regionCount >= 8,
    recoveredAtLeast3DeepOnlyRegions: deepOnlyRegionCount >= 3,
    adjacencyAtLeast10: model.adjacency.length >= 10,
    kinkCellBoundaryAtLeast180: model.kinkCellBoundaryCount >= 180,
    maeBelow004: metrics.mae < 0.04,
    r2Above99: metrics.r2 > 0.99,
    beatsFirstLayerOnlyBy15Percent: firstLayerMaeRatio < 0.85,
    beatsPlaneBy10x: metrics.mae * 10 < planeMetrics.mae,
    queryBudgetBelowStageLimit: noisy.getCalls() <= probe.rawCallBudget
  };
  const result = {
    teacherId: teacher.id,
    seed: teacher.seed,
    profile: teacher.profile,
    stage: probe.stage,
    pass: Object.values(gates).every(value => value),
    difficultyScore: 0,
    teacher,
    noise: {
      noiseStd,
      quantization,
      querySamples: probe.querySamples,
      aggregation: 'median'
    },
    config: {
      xRange,
      yRange,
      gridWidth: probe.gridWidth,
      gridHeight: probe.gridHeight,
      gradientStep: model.gradientStep,
      slopeTolerance: probe.slopeTolerance,
      interceptTolerance: probe.interceptTolerance,
      testRows: inputs.length
    },
    foldingEvidence: {
      firstLayerUnitCount: teacher.l1.length,
      secondLayerUnitCount: teacher.l2.length,
      recoveredRegionCount: model.regionCount,
      deepOnlyRegionCount,
      adjacencyEdgeCount: model.adjacency.length,
      kinkCellBoundaryCount: model.kinkCellBoundaryCount
    },
    baselines: {
      plane: {
        coefficients: plane.coefficients,
        metrics: planeMetrics
      },
      firstLayerOnly: {
        coefficients: firstLayerOnly.coefficients,
        metrics: firstLayerOnlyMetrics
      }
    },
    gcl: {
      recoveryKind: model.recoveryKind,
      expression: model.expression,
      metrics
    },
    ratios: {
      firstLayerMaeRatio,
      planeMaeRatio
    },
    queryBudget: {
      rawOracleCalls: noisy.getCalls(),
      aggregatedProbePoints: noisy.getCalls() / probe.querySamples,
      gridCells: probe.gridWidth * probe.gridHeight,
      testRows: inputs.length
    },
    gates
  };
  result.difficultyScore = difficultyScore(result);
  return result;
}

type CandidateAttempt = ReturnType<typeof runCandidateAttempt>;

function combineCandidateAttempts(teacher: Teacher, attempts: CandidateAttempt[]) {
  const finalAttempt = attempts.find(attempt => attempt.stage === 'refined') ?? attempts[0];
  const screeningAttempt = attempts.find(attempt => attempt.stage === 'screening');
  const refinedAttempt = attempts.find(attempt => attempt.stage === 'refined');
  const totalRawOracleCalls = attempts.reduce((sum, attempt) => sum + attempt.queryBudget.rawOracleCalls, 0);
  const gates = {
    ...finalAttempt.gates,
    totalQueryBudgetBelow1100000: totalRawOracleCalls <= 1100000
  };
  const result = {
    ...finalAttempt,
    finalStage: finalAttempt.stage,
    refinedUsed: Boolean(refinedAttempt),
    attemptCount: attempts.length,
    selectedHardCase: false,
    attempts,
    teacher,
    pass: Object.values(gates).every(value => value),
    queryBudget: {
      ...finalAttempt.queryBudget,
      rawOracleCalls: totalRawOracleCalls,
      finalRawOracleCalls: finalAttempt.queryBudget.rawOracleCalls,
      screeningRawOracleCalls: screeningAttempt?.queryBudget.rawOracleCalls ?? 0,
      refinedRawOracleCalls: refinedAttempt?.queryBudget.rawOracleCalls ?? 0,
      totalRawOracleCalls
    },
    gates
  };
  result.difficultyScore = difficultyScore(result);
  return result;
}

type CandidateResult = ReturnType<typeof combineCandidateAttempts>;

function summarizeProfiles(results: CandidateResult[]): Record<string, number> {
  return results.reduce<Record<string, number>>((counts, result) => {
    counts[result.profile] = (counts[result.profile] ?? 0) + 1;
    return counts;
  }, {});
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const poolSize = Math.max(6, Math.floor(Number(process.env.GCL_ADAPTIVE_DEEP_RELU2D_POOL_SIZE ?? 12)));
const selectedCount = Math.max(2, Math.floor(Number(process.env.GCL_ADAPTIVE_DEEP_RELU2D_SELECTED_COUNT ?? 4)));
const seedBase = Math.floor(Number(process.env.GCL_ADAPTIVE_DEEP_RELU2D_SEED_BASE ?? 260601));
const noiseStd = readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_NOISE_STD', 0.003);
const quantization = readNumberEnv('GCL_ADAPTIVE_DEEP_RELU2D_QUANTIZATION', 0.001);
const probeConfigs = makeProbeConfigs();
const teachers = Array.from({ length: poolSize }, (_, index) => makeTeacher(seedBase + index * 97, profileFor(index)));
const attemptEntries = teachers.map(teacher => ({
  teacher,
  attempts: [runCandidateAttempt(teacher, probeConfigs.screening, noiseStd, quantization)]
}));
const preliminaryResults = attemptEntries.map(entry => combineCandidateAttempts(entry.teacher, entry.attempts));
const preliminarySelectedIds = new Set([...preliminaryResults]
  .sort((a, b) => b.difficultyScore - a.difficultyScore)
  .slice(0, Math.min(selectedCount, preliminaryResults.length))
  .map(result => result.teacherId));
for (const entry of attemptEntries) {
  const screening = entry.attempts[0];
  if (preliminarySelectedIds.has(entry.teacher.id) || !screening.pass) {
    entry.attempts.push(runCandidateAttempt(entry.teacher, probeConfigs.refined, noiseStd, quantization));
  }
}
let results = attemptEntries.map(entry => combineCandidateAttempts(entry.teacher, entry.attempts));
let selected = [...results]
  .sort((a, b) => b.difficultyScore - a.difficultyScore)
  .slice(0, Math.min(selectedCount, results.length));
const finalSelectedIdsNeedingRefinement = new Set(selected
  .filter(result => !result.refinedUsed)
  .map(result => result.teacherId));
if (finalSelectedIdsNeedingRefinement.size > 0) {
  for (const entry of attemptEntries) {
    if (finalSelectedIdsNeedingRefinement.has(entry.teacher.id)) {
      entry.attempts.push(runCandidateAttempt(entry.teacher, probeConfigs.refined, noiseStd, quantization));
    }
  }
  results = attemptEntries.map(entry => combineCandidateAttempts(entry.teacher, entry.attempts));
  selected = [...results]
    .sort((a, b) => b.difficultyScore - a.difficultyScore)
    .slice(0, Math.min(selectedCount, results.length));
}
selected = [...results]
  .sort((a, b) => b.difficultyScore - a.difficultyScore)
  .slice(0, Math.min(selectedCount, results.length));
const selectedIds = new Set(selected.map(result => result.teacherId));
for (const result of results) result.selectedHardCase = selectedIds.has(result.teacherId);
const selectedProfiles = new Set(selected.map(result => result.profile));
const poolFailures = results.filter(result => !result.pass);
const selectedFailures = selected.filter(result => !result.pass);
const summary = {
  pass: poolFailures.length === 0
    && selectedFailures.length === 0
    && selected.length >= selectedCount
    && selectedProfiles.size >= 3
    && Math.min(...selected.map(result => result.foldingEvidence.deepOnlyRegionCount)) >= 8
    && Math.min(...selected.map(result => result.foldingEvidence.adjacencyEdgeCount)) >= 18
    && Math.max(...selected.map(result => result.gcl.metrics.mae)) < 0.03
    && Math.min(...selected.map(result => result.gcl.metrics.r2)) > 0.992
    && Math.max(...selected.map(result => result.ratios.firstLayerMaeRatio)) < 0.75
    && Math.max(...selected.map(result => result.queryBudget.rawOracleCalls)) <= 1100000
    && results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0) <= 12500000,
  poolSize: results.length,
  selectedCount: selected.length,
  passedCandidates: results.filter(result => result.pass).length,
  selectedPassed: selected.filter(result => result.pass).length,
  profiles: summarizeProfiles(results),
  selectedProfiles: Array.from(selectedProfiles).sort(),
  selectedTeacherIds: selected.map(result => result.teacherId),
  poolFailures: poolFailures.map(result => result.teacherId),
  selectedFailures: selectedFailures.map(result => result.teacherId),
  minSelectedRegionCount: Math.min(...selected.map(result => result.foldingEvidence.recoveredRegionCount)),
  minSelectedDeepOnlyRegionCount: Math.min(...selected.map(result => result.foldingEvidence.deepOnlyRegionCount)),
  minSelectedAdjacencyEdges: Math.min(...selected.map(result => result.foldingEvidence.adjacencyEdgeCount)),
  minSelectedKinkCellBoundaryCount: Math.min(...selected.map(result => result.foldingEvidence.kinkCellBoundaryCount)),
  maxSelectedMae: Math.max(...selected.map(result => result.gcl.metrics.mae)),
  minSelectedR2: Math.min(...selected.map(result => result.gcl.metrics.r2)),
  maxSelectedFirstLayerOnlyMae: Math.max(...selected.map(result => result.baselines.firstLayerOnly.metrics.mae)),
  maxSelectedPlaneMae: Math.max(...selected.map(result => result.baselines.plane.metrics.mae)),
  maxSelectedFirstLayerMaeRatio: Math.max(...selected.map(result => result.ratios.firstLayerMaeRatio)),
  maxSelectedPlaneMaeRatio: Math.max(...selected.map(result => result.ratios.planeMaeRatio)),
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0),
  maxRawOracleCallsPerCandidate: Math.max(...results.map(result => result.queryBudget.rawOracleCalls)),
  refinedCandidates: results.filter(result => result.refinedUsed).length,
  maxSelectedDifficultyScore: Math.max(...selected.map(result => result.difficultyScore)),
  minSelectedDifficultyScore: Math.min(...selected.map(result => result.difficultyScore))
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'adaptive-adversarial noisy deep black-box ReLU 2D folded region-graph decompilation',
  blackBox: {
    type: 'adaptive generated 2-input / 4-hidden / 4-hidden / 1-output two-layer ReLU MLP teachers with stochastic output noise and quantization',
    visibilityToGcl: 'oracle calls only: repeated noisy f(x,y), no hidden activations, layer widths, teacher weights, or profile labels are passed to the solver'
  },
  adaptiveSearch: {
    poolSize,
    selectedCount,
    probing: 'two-pass adaptive probing: every generated candidate gets a cheaper screening pass; screening failures and selected hard cases get a refined noisy median probe pass; both stage and total query budgets are recorded',
    selection: 'score each generated adversarial teacher by recovered deep region complexity, baseline gap, and residual hardness; fail the whole benchmark if any final generated candidate fails recovery gates',
    profiles: [
      'near-parallel first-layer folds',
      'second-layer offset collisions',
      'opposite-sign cancellation folds',
      'low-amplitude deep folds',
      'boundary-grazing folds',
      'mixed oblique folds'
    ]
  },
  claimScope: {
    proves: 'adaptive search-generated noisy deep folded 2D ReLU region-graph recovery across a deterministic adversarial candidate pool',
    doesNotClaim: 'unique original hidden-layer weight recovery, exact layer attribution, arbitrary deep ND network extraction, or production API extraction beyond this benchmark noise model'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Adaptive-Adversarial Noisy Deep Black-Box ReLU 2D Region-Graph Sweep',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${summary.pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- Teachers: adaptive generated 2-input / 4-hidden / 4-hidden / 1-output ReLU MLPs',
  '- GCL access: repeated noisy/quantized oracle calls, `f(x,y)` only',
  '- Probing: screening pass for every candidate; refined pass for screening failures and selected hard cases',
  '- Selection: generate adversarial pool, score difficulty, select hardest cases after final probing',
  '- Hard rule: any final candidate failure fails the benchmark; failures are not filtered out',
  '- Not claimed: original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, or production API extraction beyond this noise model',
  '',
  '## Summary',
  '',
  `- Pool candidates passed: ${summary.passedCandidates}/${summary.poolSize}`,
  `- Selected hard cases passed: ${summary.selectedPassed}/${summary.selectedCount}`,
  `- Selected profiles: ${summary.selectedProfiles.join(', ')}`,
  `- Selected teacher ids: ${summary.selectedTeacherIds.join(', ')}`,
  `- Min selected regions: ${summary.minSelectedRegionCount}`,
  `- Min selected deep-only region estimate: ${summary.minSelectedDeepOnlyRegionCount}`,
  `- Min selected adjacency edges: ${summary.minSelectedAdjacencyEdges}`,
  `- Min selected kink-cell boundaries: ${summary.minSelectedKinkCellBoundaryCount}`,
  `- Max selected MAE: ${summary.maxSelectedMae}`,
  `- Min selected R2: ${summary.minSelectedR2}`,
  `- Max selected first-layer MAE ratio: ${summary.maxSelectedFirstLayerMaeRatio}`,
  `- Refined candidates: ${summary.refinedCandidates}/${summary.poolSize}`,
  `- Total raw oracle calls: ${summary.totalRawOracleCalls}`,
  `- Max raw oracle calls per candidate: ${summary.maxRawOracleCallsPerCandidate}`,
  '',
  '## Results',
  '',
  '| Teacher | Profile | Stage | Selected | Pass | Score | Regions | Deep-only | Edges | Kink boundaries | MAE | R2 | First-layer ratio | First-layer MAE | Plane MAE | Final calls | Total calls |',
  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
  ...results.map(result => [
    `| ${result.teacherId}`,
    result.profile,
    result.finalStage,
    result.selectedHardCase ? 'yes' : 'no',
    result.pass ? 'yes' : 'no',
    result.difficultyScore.toFixed(6),
    result.foldingEvidence.recoveredRegionCount,
    result.foldingEvidence.deepOnlyRegionCount,
    result.foldingEvidence.adjacencyEdgeCount,
    result.foldingEvidence.kinkCellBoundaryCount,
    result.gcl.metrics.mae.toExponential(6),
    result.gcl.metrics.r2.toFixed(12),
    result.ratios.firstLayerMaeRatio.toFixed(6),
    result.baselines.firstLayerOnly.metrics.mae.toExponential(6),
    result.baselines.plane.metrics.mae.toExponential(6),
    result.queryBudget.finalRawOracleCalls,
    result.queryBudget.rawOracleCalls
  ].join(' | ') + ' |'),
  ''
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark makes hard-case selection adaptive: the generated candidate pool is screened, hard/failing cases receive refined probing, the hardest cases are selected, and any final candidate failure fails the benchmark.',
    'The solver receives only repeated noisy f(x,y) oracle calls; no teacher weights, layer widths, hidden activations, or adversarial profile labels are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
