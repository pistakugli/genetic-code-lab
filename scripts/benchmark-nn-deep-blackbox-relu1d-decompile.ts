#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxRelu1D,
  predictDeepBlackBoxRelu1DBatch
} from '../src/solvers/deep-blackbox-relu1d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type ReLULayer1Unit = { w: number; b: number };
type ReLULayerUnit = { w: number[]; b: number };

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-deep-blackbox-relu1d-decompile', 'folded-two-layer');

const teacher = {
  l1: [
    { w: -1.9999060854314883, b: -0.526933179262032 },
    { w: -0.932736135025263, b: -0.6202480226012395 },
    { w: 0.3932068481978095, b: 0.7843877941224608 },
    { w: -1.4354646070259294, b: -2.3170292492183195 },
    { w: -1.8484431317527248, b: 1.5203934386562494 }
  ] satisfies ReLULayer1Unit[],
  l2: [
    { w: [-1.3979750717039918, 0.23299646399262963, -0.028415847596168442, -1.5851346753399211, -1.3584603810296025], b: 0.2673017394424433 },
    { w: [-1.9462073929106884, -1.907621767285859, -1.3990121925240495, -1.1978931512664008, -0.9901683078987229], b: -1.319045590068256 },
    { w: [1.0677197138478234, 1.1652379363451506, 0.15400268524326632, 0.3231453302531926, -0.8964213122580398], b: 1.3852706606362677 },
    { w: [-1.0080077191889358, -1.5857128683344581, 0.9238499653747771, -0.8536235251032036, 1.149435923573967], b: -1.0728193778291526 },
    { w: [-1.1670173864504485, 1.9388107135322046, 1.5916628144603808, -0.9230741699441096, 1.8924486263584797], b: -1.2119519647322148 }
  ] satisfies ReLULayerUnit[],
  out: {
    w: [0.7233499700421002, 1.3429525758539818, -0.996056394182198, -0.7197974861783882, 1.3636671727184835],
    b: -0.16916541091088708,
    base: 0.15796166170198628
  }
};

function relu(value: number): number {
  return Math.max(0, value);
}

function deepTeacher(x: number): number {
  const h1 = teacher.l1.map(unit => relu(unit.w * x + unit.b));
  const h2 = teacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
  return teacher.out.b + teacher.out.base * x + teacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
}

function makeGrid(count: number, min: number, max: number, offset = 0): number[] {
  return Array.from({ length: count }, (_, index) => min + offset + ((max - min - 2 * offset) * index) / (count - 1));
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

function fitReluBasis(xs: number[], targets: number[], kinks: number[]): { coefficients: number[]; predicted: number[] } {
  const width = 2 + kinks.length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (let rowIndex = 0; rowIndex < xs.length; rowIndex++) {
    const row = [1, xs[rowIndex], ...kinks.map(kink => relu(xs[rowIndex] - kink))];
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * targets[rowIndex];
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += 1e-10;
  const coefficients = solveSystem(xtx, xty);
  return {
    coefficients,
    predicted: xs.map(x => coefficients[0] + coefficients[1] * x + kinks.reduce((sum, kink, index) => sum + coefficients[2 + index] * relu(x - kink), 0))
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const min = Number(process.env.GCL_DEEP_RELU1D_MIN ?? -3);
const max = Number(process.env.GCL_DEEP_RELU1D_MAX ?? 3);
const samples = Math.max(2048, Math.floor(Number(process.env.GCL_DEEP_RELU1D_SAMPLES ?? 12001)));
const testXs = makeGrid(2001, min, max, 0.0017);
const targets = testXs.map(deepTeacher);
const firstLayerKinks = teacher.l1
  .map(unit => -unit.b / unit.w)
  .filter(kink => kink > min && kink < max)
  .sort((a, b) => a - b);

const model = decompileDeepBlackBoxRelu1D(deepTeacher, {
  min,
  max,
  samples,
  slopeTolerance: Number(process.env.GCL_DEEP_RELU1D_SLOPE_TOL ?? 1e-7),
  minPlateauIntervals: Number(process.env.GCL_DEEP_RELU1D_MIN_PLATEAU ?? 4),
  kinkMergeTolerance: Number(process.env.GCL_DEEP_RELU1D_KINK_MERGE_TOL ?? 0.002)
});
const predicted = predictDeepBlackBoxRelu1DBatch(model, testXs);
const metrics = regressionMetrics(targets, predicted);
const firstLayerOnly = fitReluBasis(testXs, targets, firstLayerKinks);
const firstLayerOnlyMetrics = regressionMetrics(targets, firstLayerOnly.predicted);
const linear = fitReluBasis(testXs, targets, []);
const linearMetrics = regressionMetrics(targets, linear.predicted);
const deepOnlyKinkCount = Math.max(0, model.kinkCount - firstLayerKinks.length);
const pass = model.kinkCount >= 10
  && deepOnlyKinkCount >= 5
  && metrics.mae < 1e-9
  && metrics.r2 > 0.999999999
  && metrics.mae < firstLayerOnlyMetrics.mae * 1e-6
  && metrics.mae < linearMetrics.mae * 1e-8;

const report = {
  generatedAt: new Date().toISOString(),
  task: 'deep black-box ReLU 1D folded composition decompilation',
  blackBox: {
    type: '1-input / 5-hidden / 5-hidden / 1-output two-layer ReLU MLP teacher',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(x), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'output-equivalent canonical spline recovery for a deep folded 1D ReLU black-box teacher',
    doesNotClaim: 'unique original hidden-layer weight recovery, arbitrary deep ND network extraction, or stochastic API decompilation'
  },
  config: {
    min,
    max,
    samples,
    testRows: testXs.length
  },
  auditTeacherCircuit: teacher,
  foldingEvidence: {
    firstLayerKinks,
    firstLayerKinkCount: firstLayerKinks.length,
    recoveredCanonicalKinks: model.terms.map(term => term.kink),
    recoveredCanonicalKinkCount: model.kinkCount,
    recoveredRegionCount: model.regionCount,
    deepOnlyKinkCount
  },
  baselines: {
    linear: {
      coefficients: linear.coefficients,
      metrics: linearMetrics
    },
    firstLayerKinkOnly: {
      coefficients: firstLayerOnly.coefficients,
      kinks: firstLayerKinks,
      metrics: firstLayerOnlyMetrics
    }
  },
  gcl: {
    recoveryKind: model.recoveryKind,
    expression: model.expression,
    baseIntercept: model.baseIntercept,
    baseSlope: model.baseSlope,
    terms: model.terms,
    plateaus: model.plateaus.length,
    metrics
  },
  gates: {
    pass,
    recoveredAtLeast10Kinks: model.kinkCount >= 10,
    recoveredDeepOnlyKinks: deepOnlyKinkCount >= 5,
    maeBelow1eMinus9: metrics.mae < 1e-9,
    r2Above999999999: metrics.r2 > 0.999999999,
    beatsFirstLayerOnlyBy1e6: metrics.mae < firstLayerOnlyMetrics.mae * 1e-6,
    beatsLinearBy1e8: metrics.mae < linearMetrics.mae * 1e-8
  },
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Deep Black-Box ReLU 1D Decompile Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- Teacher: 1-input / 5-hidden / 5-hidden / 1-output ReLU MLP',
  '- GCL access: oracle calls only, `f(x)`',
  '- Claim: output-equivalent canonical spline recovery',
  '- Not claimed: unique original hidden-layer weights',
  '',
  '## Metrics',
  '',
  '| Model | MAE | RMSE | R2 |',
  '|---|---:|---:|---:|',
  `| Linear | ${linearMetrics.mae.toExponential(6)} | ${linearMetrics.rmse.toExponential(6)} | ${linearMetrics.r2.toFixed(12)} |`,
  `| First-layer-kink-only | ${firstLayerOnlyMetrics.mae.toExponential(6)} | ${firstLayerOnlyMetrics.rmse.toExponential(6)} | ${firstLayerOnlyMetrics.r2.toFixed(12)} |`,
  `| GCL canonical deep spline | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(12)} |`,
  '',
  '## Folding Evidence',
  '',
  `- First-layer kink count: ${firstLayerKinks.length}`,
  `- Recovered canonical kink count: ${model.kinkCount}`,
  `- Deep-only kink count: ${deepOnlyKinkCount}`,
  `- Recovered region count: ${model.regionCount}`,
  '',
  '## Recovered Formula',
  '',
  '```',
  model.expression,
  '```',
  ''
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, markdown);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-deep-blackbox-relu1d-decompile',
  status: pass ? 'pass' : 'fail',
  summary: {
    pass,
    kinkCount: model.kinkCount,
    firstLayerKinkCount: firstLayerKinks.length,
    deepOnlyKinkCount,
    mae: metrics.mae,
    r2: metrics.r2,
    firstLayerOnlyMae: firstLayerOnlyMetrics.mae
  },
  notes: [
    'This benchmark starts the deep ReLU lane with output-equivalent recovery for a folded 1D two-layer ReLU teacher.',
    'The solver receives only f(x), not hidden activations, layer widths, or teacher weights.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!pass) process.exit(1);
