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

type ReLULayer1Unit = { w: [number, number]; b: number };
type ReLULayerUnit = { w: number[]; b: number };

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-deep-blackbox-relu2d-region-graph', 'folded-two-layer');

const teacher = {
  seed: 1,
  l1: [
    { w: [-1.4999765222891948, -1.105386636783729], b: 0.6134527729949475 },
    { w: [-1.358866152687805, 0.5365941501563363], b: 0.43031137364945504 },
    { w: [1.3040786877312498, -0.34949376839165924], b: 0.04659929242599681 },
    { w: [0.992896038100958, -1.3962836697663028], b: -1.0716920769509786 }
  ] satisfies ReLULayer1Unit[],
  l2: [
    { w: [0.08316054072525403, 0.479218274987506, -1.3784450799026013, -0.3264361786902288], b: -0.7796839732487537 },
    { w: [-0.23103927227765264, 0.5229635942009869, 0.24913459946320815, 1.2052221851472016], b: 0.6231004027864898 },
    { w: [0.07540057662446098, -1.142498307062777, 0.43097309398555494, -0.2352018015787023], b: 0.36214306974983124 },
    { w: [1.1488983252541147, 0.7341545108092524, -0.6651316187019753, -1.2670993628605263], b: 0.4249473920324326 }
  ] satisfies ReLULayerUnit[],
  out: {
    w: [-0.4122378580386171, 0.3183325625195472, 0.615385166011178, 1.1784897252716957],
    b: -0.053864531827964385,
    base: [-0.05059222295004151, 0.0965100572411996] as [number, number]
  }
};

function relu(value: number): number {
  return Math.max(0, value);
}

function deepTeacher(x: number, y: number): number {
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

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const xRange: [number, number] = [-2, 2];
const yRange: [number, number] = [-2, 2];
const gridWidth = Math.max(64, Math.floor(Number(process.env.GCL_DEEP_RELU2D_REGION_GRID_WIDTH ?? 101)));
const gridHeight = Math.max(64, Math.floor(Number(process.env.GCL_DEEP_RELU2D_REGION_GRID_HEIGHT ?? 101)));
let oracleCalls = 0;
const countedOracle = (x: number, y: number): number => {
  oracleCalls++;
  return deepTeacher(x, y);
};
const model = decompileDeepBlackBoxRelu2DRegionGraph(countedOracle, {
  xRange,
  yRange,
  gridWidth,
  gridHeight,
  gradientStep: Number(process.env.GCL_DEEP_RELU2D_GRADIENT_STEP ?? 1e-5),
  slopeTolerance: Number(process.env.GCL_DEEP_RELU2D_SLOPE_TOL ?? 1e-5),
  interceptTolerance: Number(process.env.GCL_DEEP_RELU2D_INTERCEPT_TOL ?? 1e-5)
});
const inputs = makeGrid(121, 111, 0.37, 0.23);
const actual = inputs.map(([x, y]) => deepTeacher(x, y));
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
const pass = model.regionCount >= 18
  && deepOnlyRegionCount >= 12
  && model.adjacency.length >= 35
  && model.kinkCellBoundaryCount >= 850
  && metrics.mae < 0.001
  && metrics.r2 > 0.99999
  && metrics.mae < firstLayerOnlyMetrics.mae * 0.01
  && metrics.mae < planeMetrics.mae * 0.001;

const report = {
  generatedAt: new Date().toISOString(),
  task: 'deep black-box ReLU 2D folded region-graph decompilation',
  blackBox: {
    type: '2-input / 4-hidden / 4-hidden / 1-output two-layer ReLU MLP teacher',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'output-equivalent affine-region graph recovery for a deep folded 2D ReLU black-box teacher',
    doesNotClaim: 'unique original hidden-layer weight recovery, exact layer attribution, arbitrary deep ND network extraction, or stochastic API decompilation'
  },
  config: {
    xRange,
    yRange,
    gridWidth,
    gridHeight,
    testRows: inputs.length
  },
  auditTeacherCircuit: teacher,
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
    regions: model.regions,
    adjacency: model.adjacency,
    metrics
  },
  queryBudget: {
    rawOracleCalls: oracleCalls,
    gridCells: gridWidth * gridHeight
  },
  gates: {
    pass,
    recoveredAtLeast18Regions: model.regionCount >= 18,
    recoveredDeepOnlyRegions: deepOnlyRegionCount >= 12,
    adjacencyAtLeast35: model.adjacency.length >= 35,
    kinkCellBoundaryAtLeast850: model.kinkCellBoundaryCount >= 850,
    maeBelow001: metrics.mae < 0.001,
    r2Above99999: metrics.r2 > 0.99999,
    beatsFirstLayerOnlyBy100x: metrics.mae < firstLayerOnlyMetrics.mae * 0.01,
    beatsPlaneBy1000x: metrics.mae < planeMetrics.mae * 0.001
  },
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Deep Black-Box ReLU 2D Region-Graph Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- Teacher: 2-input / 4-hidden / 4-hidden / 1-output ReLU MLP',
  '- GCL access: oracle calls only, `f(x,y)`',
  '- Claim: output-equivalent affine-region graph recovery',
  '- Not claimed: original hidden-layer weights or exact layer attribution',
  '',
  '## Metrics',
  '',
  '| Model | MAE | RMSE | R2 |',
  '|---|---:|---:|---:|',
  `| Plane | ${planeMetrics.mae.toExponential(6)} | ${planeMetrics.rmse.toExponential(6)} | ${planeMetrics.r2.toFixed(12)} |`,
  `| First-layer-only ReLU basis | ${firstLayerOnlyMetrics.mae.toExponential(6)} | ${firstLayerOnlyMetrics.rmse.toExponential(6)} | ${firstLayerOnlyMetrics.r2.toFixed(12)} |`,
  `| GCL region graph | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(12)} |`,
  '',
  '## Folding Evidence',
  '',
  `- First-layer unit count: ${teacher.l1.length}`,
  `- Recovered affine region count: ${model.regionCount}`,
  `- Deep-only region count estimate: ${deepOnlyRegionCount}`,
  `- Region adjacency edges: ${model.adjacency.length}`,
  `- Kink cell boundaries: ${model.kinkCellBoundaryCount}`,
  `- Raw oracle calls: ${oracleCalls}`,
  '',
  '## Recovered Structure',
  '',
  '```',
  model.expression,
  '```',
  ''
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, markdown);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-deep-blackbox-relu2d-region-graph',
  status: pass ? 'pass' : 'fail',
  summary: {
    pass,
    regionCount: model.regionCount,
    adjacencyEdgeCount: model.adjacency.length,
    kinkCellBoundaryCount: model.kinkCellBoundaryCount,
    mae: metrics.mae,
    r2: metrics.r2,
    firstLayerOnlyMae: firstLayerOnlyMetrics.mae,
    rawOracleCalls: oracleCalls
  },
  notes: [
    'This benchmark is the first 2D deep/folded ReLU output-equivalent region-graph recovery lane.',
    'The solver receives only f(x,y), not hidden activations, layer widths, or teacher weights.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!pass) process.exit(1);
