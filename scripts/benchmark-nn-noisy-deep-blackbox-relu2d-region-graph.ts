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
type Teacher = {
  id: string;
  seed: number;
  l1: Layer1Unit[];
  l2: LayerUnit[];
  out: { w: number[]; b: number; base: [number, number] };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-deep-blackbox-relu2d-region-graph', 'seed-sweep');

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

function norm2(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}

function makeLayer1(random: () => number, count: number): Layer1Unit[] {
  const units: Layer1Unit[] = [];
  for (let index = 0; index < count; index++) {
    for (let attempt = 0; attempt < 1000; attempt++) {
      const angle = randomBetween(random, -Math.PI, Math.PI);
      const scale = randomBetween(random, 0.65, 1.55);
      const normal: [number, number] = [Math.cos(angle) * scale, Math.sin(angle) * scale];
      const separated = units.every(unit => {
        const dot = (unit.w[0] * normal[0] + unit.w[1] * normal[1]) / Math.max(1e-12, norm2(unit.w[0], unit.w[1]) * norm2(normal[0], normal[1]));
        return Math.abs(dot) < 0.94;
      });
      if (!separated) continue;
      units.push({ w: normal, b: randomBetween(random, -0.85, 0.85) });
      break;
    }
    if (units.length !== index + 1) throw new Error('failed to generate separated first-layer unit');
  }
  return units;
}

function makeTeacher(seed: number): Teacher {
  const random = makeRandom(seed);
  const l1 = makeLayer1(random, 4);
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: l1.length }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      return sign * randomBetween(random, 0.18, 1.25);
    }),
    b: randomBetween(random, -0.75, 0.75)
  }));
  return {
    id: `noisy-deep-2d-seed-${seed}`,
    seed,
    l1,
    l2,
    out: {
      w: Array.from({ length: l2.length }, (_, index) => {
        const sign = index % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.32, 1.2);
      }),
      b: randomBetween(random, -0.18, 0.18),
      base: [randomBetween(random, -0.09, 0.09), randomBetween(random, -0.09, 0.09)]
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

function runTeacher(teacher: Teacher) {
  const xRange: [number, number] = [-2, 2];
  const yRange: [number, number] = [-2, 2];
  const gridWidth = Math.max(48, Math.floor(Number(process.env.GCL_NOISY_DEEP_RELU2D_GRID_WIDTH ?? 81)));
  const gridHeight = Math.max(48, Math.floor(Number(process.env.GCL_NOISY_DEEP_RELU2D_GRID_HEIGHT ?? 81)));
  const querySamples = Math.max(3, Math.floor(Number(process.env.GCL_NOISY_DEEP_RELU2D_QUERY_SAMPLES ?? 21)));
  const noiseStd = Number(process.env.GCL_NOISY_DEEP_RELU2D_NOISE_STD ?? 0.003);
  const quantization = Number(process.env.GCL_NOISY_DEEP_RELU2D_QUANTIZATION ?? 0.001);
  const noisy = makeNoisyOracle(teacher, teacher.seed + 260525, noiseStd, quantization);
  const model = decompileDeepBlackBoxRelu2DRegionGraph(noisy.oracle, {
    xRange,
    yRange,
    gridWidth,
    gridHeight,
    gradientStep: Number(process.env.GCL_NOISY_DEEP_RELU2D_GRADIENT_STEP ?? 0.075),
    slopeTolerance: Number(process.env.GCL_NOISY_DEEP_RELU2D_SLOPE_TOL ?? 0.11),
    interceptTolerance: Number(process.env.GCL_NOISY_DEEP_RELU2D_INTERCEPT_TOL ?? 0.18),
    querySamples,
    aggregation: 'median'
  });
  const inputs = makeGrid(73, 67, 0.41, 0.29);
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
  const gates = {
    recoveredAtLeast10Regions: model.regionCount >= 10,
    recoveredAtLeast4DeepOnlyRegions: deepOnlyRegionCount >= 4,
    adjacencyAtLeast14: model.adjacency.length >= 14,
    kinkCellBoundaryAtLeast250: model.kinkCellBoundaryCount >= 250,
    maeBelow003: metrics.mae < 0.03,
    r2Above995: metrics.r2 > 0.995,
    beatsFirstLayerOnlyBy4x: metrics.mae * 4 < firstLayerOnlyMetrics.mae,
    beatsPlaneBy20x: metrics.mae * 20 < planeMetrics.mae,
    queryBudgetBelow800000: noisy.getCalls() <= 800000
  };
  const pass = Object.values(gates).every(value => value);
  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    pass,
    teacher,
    noise: {
      noiseStd,
      quantization,
      querySamples,
      aggregation: 'median'
    },
    config: {
      xRange,
      yRange,
      gridWidth,
      gridHeight,
      gradientStep: model.gradientStep,
      slopeTolerance: Number(process.env.GCL_NOISY_DEEP_RELU2D_SLOPE_TOL ?? 0.11),
      interceptTolerance: Number(process.env.GCL_NOISY_DEEP_RELU2D_INTERCEPT_TOL ?? 0.18),
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
    queryBudget: {
      rawOracleCalls: noisy.getCalls(),
      aggregatedProbePoints: noisy.getCalls() / querySamples,
      gridCells: gridWidth * gridHeight,
      testRows: inputs.length
    },
    gates
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const seeds = (process.env.GCL_NOISY_DEEP_RELU2D_SEEDS ?? '29,83')
  .split(',')
  .map(seed => Number(seed.trim()))
  .filter(seed => Number.isFinite(seed));
if (seeds.length === 0) throw new Error('GCL_NOISY_DEEP_RELU2D_SEEDS produced no valid seeds');
const results = seeds.map(seed => runTeacher(makeTeacher(seed)));
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  seeds,
  minRegionCount: Math.min(...results.map(result => result.foldingEvidence.recoveredRegionCount)),
  minDeepOnlyRegionCount: Math.min(...results.map(result => result.foldingEvidence.deepOnlyRegionCount)),
  minAdjacencyEdges: Math.min(...results.map(result => result.foldingEvidence.adjacencyEdgeCount)),
  minKinkCellBoundaryCount: Math.min(...results.map(result => result.foldingEvidence.kinkCellBoundaryCount)),
  maxMae: Math.max(...results.map(result => result.gcl.metrics.mae)),
  minR2: Math.min(...results.map(result => result.gcl.metrics.r2)),
  maxPlaneMae: Math.max(...results.map(result => result.baselines.plane.metrics.mae)),
  maxFirstLayerOnlyMae: Math.max(...results.map(result => result.baselines.firstLayerOnly.metrics.mae)),
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0),
  maxRawOracleCallsPerTeacher: Math.max(...results.map(result => result.queryBudget.rawOracleCalls))
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'noisy randomized deep black-box ReLU 2D folded region-graph decompilation',
  blackBox: {
    type: 'deterministic randomized 2-input / 4-hidden / 4-hidden / 1-output two-layer ReLU MLP teachers with stochastic output noise and quantization',
    visibilityToGcl: 'oracle calls only: repeated noisy f(x,y), no hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'noise-tolerant output-equivalent affine-region graph recovery across a deterministic deep folded 2D ReLU seed sweep',
    doesNotClaim: 'unique original hidden-layer weight recovery, exact layer attribution, arbitrary deep ND network extraction, or production API extraction beyond the benchmark noise model'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Noisy Deep Black-Box ReLU 2D Region-Graph Sweep',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${summary.pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- Teachers: deterministic randomized 2-input / 4-hidden / 4-hidden / 1-output ReLU MLPs',
  '- GCL access: repeated noisy/quantized oracle calls, `f(x,y)` only',
  '- Claim: noise-tolerant output-equivalent affine-region graph recovery across a seed sweep',
  '- Not claimed: original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, or production API extraction beyond this noise model',
  '',
  '## Summary',
  '',
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Seeds: ${summary.seeds.join(', ')}`,
  `- Min recovered regions: ${summary.minRegionCount}`,
  `- Min deep-only region estimate: ${summary.minDeepOnlyRegionCount}`,
  `- Min adjacency edges: ${summary.minAdjacencyEdges}`,
  `- Min kink-cell boundaries: ${summary.minKinkCellBoundaryCount}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Total raw oracle calls: ${summary.totalRawOracleCalls}`,
  `- Max raw oracle calls per teacher: ${summary.maxRawOracleCallsPerTeacher}`,
  '',
  '## Results',
  '',
  '| Teacher | Pass | Regions | Deep-only | Edges | Kink boundaries | MAE | R2 | First-layer MAE | Plane MAE | Raw calls |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
  ...results.map(result => [
    `| ${result.teacherId}`,
    result.pass ? 'yes' : 'no',
    result.foldingEvidence.recoveredRegionCount,
    result.foldingEvidence.deepOnlyRegionCount,
    result.foldingEvidence.adjacencyEdgeCount,
    result.foldingEvidence.kinkCellBoundaryCount,
    result.gcl.metrics.mae.toExponential(6),
    result.gcl.metrics.r2.toFixed(12),
    result.baselines.firstLayerOnly.metrics.mae.toExponential(6),
    result.baselines.plane.metrics.mae.toExponential(6),
    result.queryBudget.rawOracleCalls
  ].join(' | ') + ' |'),
  ''
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-noisy-deep-blackbox-relu2d-region-graph',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark is the first noisy/quantized deep folded 2D region-graph recovery lane.',
    'The solver receives repeated noisy f(x,y) oracle calls only; no teacher weights, layer widths, or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
