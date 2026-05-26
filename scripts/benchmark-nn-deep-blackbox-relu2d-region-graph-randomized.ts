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
  out: {
    w: number[];
    b: number;
    base: [number, number];
  };
};

type TeacherResult = {
  teacherId: string;
  seed: number;
  pass: boolean;
  teacher: Teacher;
  config: {
    xRange: [number, number];
    yRange: [number, number];
    gridWidth: number;
    gridHeight: number;
    testRows: number;
  };
  foldingEvidence: {
    firstLayerUnitCount: number;
    secondLayerUnitCount: number;
    recoveredRegionCount: number;
    deepOnlyRegionCount: number;
    adjacencyEdgeCount: number;
    kinkCellBoundaryCount: number;
  };
  baselines: {
    plane: { coefficients: number[]; metrics: ReturnType<typeof regressionMetrics> };
    firstLayerOnly: { coefficients: number[]; metrics: ReturnType<typeof regressionMetrics> };
  };
  gcl: {
    recoveryKind: string;
    expression: string;
    metrics: ReturnType<typeof regressionMetrics>;
  };
  queryBudget: {
    rawOracleCalls: number;
    gridCells: number;
  };
  gates: Record<string, boolean | number>;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-deep-blackbox-relu2d-region-graph-randomized',
  'seed-sweep'
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
      units.push({
        w: normal,
        b: randomBetween(random, -0.85, 0.85)
      });
      break;
    }
    if (units.length !== index + 1) throw new Error('failed to generate separated first-layer unit');
  }
  return units;
}

function makeLayer2(random: () => number, inputWidth: number, count: number): LayerUnit[] {
  return Array.from({ length: count }, (_, unitIndex) => {
    const w = Array.from({ length: inputWidth }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      return sign * randomBetween(random, 0.18, 1.25);
    });
    return {
      w,
      b: randomBetween(random, -0.75, 0.75)
    };
  });
}

function makeTeacher(seed: number): Teacher {
  const random = makeRandom(seed);
  const l1 = makeLayer1(random, 4);
  const l2 = makeLayer2(random, l1.length, 4);
  const outWeights = Array.from({ length: l2.length }, (_, index) => {
    const sign = index % 2 === 0 ? 1 : -1;
    return sign * randomBetween(random, 0.32, 1.2);
  });
  return {
    id: `deep-2d-seed-${seed}`,
    seed,
    l1,
    l2,
    out: {
      w: outWeights,
      b: randomBetween(random, -0.18, 0.18),
      base: [randomBetween(random, -0.09, 0.09), randomBetween(random, -0.09, 0.09)]
    }
  };
}

function evaluateTeacher(teacher: Teacher, x: number, y: number): number {
  const h1 = teacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b));
  const h2 = teacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
  return teacher.out.b
    + teacher.out.base[0] * x
    + teacher.out.base[1] * y
    + teacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
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

function runTeacher(teacher: Teacher): TeacherResult {
  const xRange: [number, number] = [-2, 2];
  const yRange: [number, number] = [-2, 2];
  const gridWidth = Math.max(64, Math.floor(Number(process.env.GCL_DEEP_RELU2D_RANDOM_GRID_WIDTH ?? 101)));
  const gridHeight = Math.max(64, Math.floor(Number(process.env.GCL_DEEP_RELU2D_RANDOM_GRID_HEIGHT ?? 101)));
  let oracleCalls = 0;
  const countedOracle = (x: number, y: number): number => {
    oracleCalls++;
    return evaluateTeacher(teacher, x, y);
  };
  const model = decompileDeepBlackBoxRelu2DRegionGraph(countedOracle, {
    xRange,
    yRange,
    gridWidth,
    gridHeight,
    gradientStep: Number(process.env.GCL_DEEP_RELU2D_RANDOM_GRADIENT_STEP ?? 1e-5),
    slopeTolerance: Number(process.env.GCL_DEEP_RELU2D_RANDOM_SLOPE_TOL ?? 1e-5),
    interceptTolerance: Number(process.env.GCL_DEEP_RELU2D_RANDOM_INTERCEPT_TOL ?? 1e-5)
  });
  const testInputs = makeGrid(91, 83, 0.41, 0.29);
  const targets = testInputs.map(([x, y]) => evaluateTeacher(teacher, x, y));
  const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, testInputs);
  const metrics = regressionMetrics(targets, predicted);
  const plane = fitBasis(testInputs, targets, ([x, y]) => [1, x, y]);
  const planeMetrics = regressionMetrics(targets, plane.predicted);
  const firstLayerOnly = fitBasis(testInputs, targets, ([x, y]) => [
    1,
    x,
    y,
    ...teacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b))
  ]);
  const firstLayerOnlyMetrics = regressionMetrics(targets, firstLayerOnly.predicted);
  const deepOnlyRegionCount = Math.max(0, model.regionCount - teacher.l1.length - 1);
  const gates = {
    recoveredAtLeast12Regions: model.regionCount >= 12,
    recoveredAtLeast6DeepOnlyRegions: deepOnlyRegionCount >= 6,
    adjacencyAtLeast18: model.adjacency.length >= 18,
    kinkCellBoundaryAtLeast350: model.kinkCellBoundaryCount >= 350,
    maeBelow0015: metrics.mae < 0.015,
    r2Above999: metrics.r2 > 0.999,
    beatsFirstLayerOnlyBy20x: metrics.mae * 20 < firstLayerOnlyMetrics.mae,
    beatsPlaneBy100x: metrics.mae * 100 < planeMetrics.mae,
    queryBudgetBelow60000: oracleCalls <= 60000
  };
  const pass = Object.values(gates).every(value => value);
  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    pass,
    teacher,
    config: {
      xRange,
      yRange,
      gridWidth,
      gridHeight,
      testRows: testInputs.length
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
      rawOracleCalls: oracleCalls,
      gridCells: gridWidth * gridHeight
    },
    gates
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const seeds = (process.env.GCL_DEEP_RELU2D_RANDOM_SEEDS ?? '11,29,47,83')
  .split(',')
  .map(seed => Number(seed.trim()))
  .filter(seed => Number.isFinite(seed));
if (seeds.length === 0) throw new Error('GCL_DEEP_RELU2D_RANDOM_SEEDS produced no valid seeds');
const teachers = seeds.map(makeTeacher);
const results = teachers.map(runTeacher);
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
  task: 'randomized deep black-box ReLU 2D folded region-graph decompilation',
  blackBox: {
    type: 'deterministic randomized 2-input / 4-hidden / 4-hidden / 1-output two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'output-equivalent affine-region graph recovery across a deterministic randomized deep folded 2D ReLU seed sweep',
    doesNotClaim: 'unique original hidden-layer weight recovery, exact layer attribution, arbitrary deep ND network extraction, or stochastic API decompilation'
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Randomized Deep Black-Box ReLU 2D Region-Graph Sweep',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${summary.pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- Teachers: deterministic randomized 2-input / 4-hidden / 4-hidden / 1-output ReLU MLPs',
  '- GCL access: oracle calls only, `f(x,y)`',
  '- Claim: output-equivalent affine-region graph recovery across a seed sweep',
  '- Not claimed: original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, or noisy API recovery',
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
  '| Teacher | Pass | Regions | Deep-only | Edges | Kink boundaries | MAE | R2 | First-layer MAE | Plane MAE | Oracle calls |',
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
  lane: 'nn-deep-blackbox-relu2d-region-graph-randomized',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark broadens the deep/folded 2D region-graph lane from one fixed teacher to a deterministic randomized seed sweep.',
    'The solver receives only f(x,y) oracle calls; no teacher weights, layer widths, or hidden activations are passed to GCL.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
