#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileBlackBoxRelu2D,
  predictBlackBoxRelu2DBatch
} from '../src/solvers/blackbox-relu2d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-blackbox-relu2d-decompile', 'smoke');

const teacher = {
  bias: 0.2,
  base: [0.3, -0.15],
  hidden: [
    { alphaY: -0.4, beta: -1.2, readout: 0.9 },
    { alphaY: 0.7, beta: 0.15, readout: -0.55 },
    { alphaY: -0.2, beta: 1.1, readout: 0.35 }
  ]
};

function blackBoxTeacher(x: number, y: number): number {
  return teacher.bias
    + teacher.base[0] * x
    + teacher.base[1] * y
    + teacher.hidden.reduce(
      (sum, unit) => sum + unit.readout * Math.max(0, x - unit.alphaY * y - unit.beta),
      0
    );
}

function fitPlane(inputs: Array<[number, number]>, targets: number[]): { coefficients: [number, number, number]; predicted: number[] } {
  const xtx = Array.from({ length: 3 }, () => new Array(3).fill(0));
  const xty = new Array(3).fill(0);
  for (let i = 0; i < inputs.length; i++) {
    const row = [1, inputs[i][0], inputs[i][1]];
    for (let a = 0; a < 3; a++) {
      xty[a] += row[a] * targets[i];
      for (let b = 0; b < 3; b++) xtx[a][b] += row[a] * row[b];
    }
  }
  const coefficients = solve3x3(xtx, xty);
  return {
    coefficients: coefficients as [number, number, number],
    predicted: inputs.map(([x, y]) => coefficients[0] + coefficients[1] * x + coefficients[2] * y)
  };
}

function solve3x3(matrix: number[][], rhs: number[]): number[] {
  const a = matrix.map((row, index) => [...row, rhs[index]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= 3; j++) a[col][j] /= divisor;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= 3; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[3]);
}

function makeGrid(): Array<[number, number]> {
  const inputs: Array<[number, number]> = [];
  for (let i = 0; i < 61; i++) {
    for (let j = 0; j < 45; j++) {
      inputs.push([
        -2.85 + (5.7 * i) / 60,
        -1.425 + (2.85 * j) / 44
      ]);
    }
  }
  return inputs;
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const xRange: [number, number] = [-3, 3];
  const yRange: [number, number] = [-1.5, 1.5];
  const samplesPerSlice = Math.max(64, Number(process.env.GCL_BLACKBOX_RELU2D_SAMPLES_PER_SLICE ?? 1201));
  const model = decompileBlackBoxRelu2D(blackBoxTeacher, {
    xRange,
    yRange,
    samplesPerSlice,
    slopeTolerance: Number(process.env.GCL_BLACKBOX_RELU2D_SLOPE_TOL ?? 1e-8),
    clusterSlopeTolerance: Number(process.env.GCL_BLACKBOX_RELU2D_CLUSTER_TOL ?? 1e-7)
  });
  const inputs = makeGrid();
  const actual = inputs.map(([x, y]) => blackBoxTeacher(x, y));
  const predicted = predictBlackBoxRelu2DBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const plane = fitPlane(inputs, actual);
  const planeMetrics = regressionMetrics(actual, plane.predicted);
  const recoveredDeltas = model.terms.map(term => term.deltaSlopeX).sort((a, b) => a - b);
  const expectedDeltas = teacher.hidden.map(term => term.readout).sort((a, b) => a - b);
  const maxDeltaError = expectedDeltas.reduce((maxError, expected, index) =>
    Math.max(maxError, Math.abs(expected - (recoveredDeltas[index] ?? Infinity))), 0);
  const pass = model.terms.length === teacher.hidden.length
    && maxDeltaError < 1e-8
    && metrics.mae < 1e-9
    && metrics.r2 > 0.999999999
    && metrics.mae < planeMetrics.mae * 0.001;

  const report = {
    generatedAt: new Date().toISOString(),
    task: '2D black-box ReLU circuit decompilation',
    blackBox: {
      type: '2-input/3-hidden non-vertical ReLU MLP teacher',
      visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y), not hidden activations or teacher weights'
    },
    config: {
      xRange,
      yRange,
      samplesPerSlice,
      testRows: inputs.length
    },
    auditTeacherCircuit: teacher,
    baselines: {
      plane: {
        expression: `${plane.coefficients[0]} + ${plane.coefficients[1]} * x + ${plane.coefficients[2]} * y`,
        metrics: planeMetrics
      }
    },
    gcl: {
      expression: model.expression,
      baseIntercept: model.baseIntercept,
      baseSlopeX: model.baseSlopeX,
      baseSlopeY: model.baseSlopeY,
      terms: model.terms,
      metrics
    },
    gates: {
      pass,
      expectedDeltas,
      recoveredDeltas,
      maxDeltaError,
      allKinkLinesRecovered: model.terms.length === teacher.hidden.length && maxDeltaError < 1e-8,
      maeBelow1eMinus9: metrics.mae < 1e-9,
      r2Above999999999: metrics.r2 > 0.999999999,
      beatsPlaneBy1000x: metrics.mae < planeMetrics.mae * 0.001
    },
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN 2D Black-Box ReLU Decompile Benchmark',
    '',
    '- Teacher: fixed 2-input/3-hidden non-vertical ReLU MLP',
    '- GCL access: oracle calls only, `f(x,y)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Samples per horizontal slice: ${samplesPerSlice}`,
    `- Test rows: ${inputs.length}`,
    '',
    '| Model | Test MAE | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Plane baseline | ${planeMetrics.mae.toExponential(6)} | ${planeMetrics.rmse.toExponential(6)} | ${planeMetrics.r2.toFixed(9)} |`,
    `| GCL 2D black-box ReLU decompiler | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(9)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- All kink lines recovered: ${report.gates.allKinkLinesRecovered ? 'yes' : 'no'}`,
    `- Max delta-slope error: ${maxDeltaError.toExponential(6)}`,
    `- MAE < 1e-9: ${report.gates.maeBelow1eMinus9 ? 'yes' : 'no'} (${metrics.mae.toExponential(6)})`,
    `- R2 > 0.999999999: ${report.gates.r2Above999999999 ? 'yes' : 'no'} (${metrics.r2.toFixed(12)})`,
    '',
    '## Recovered Formula',
    '',
    '```',
    model.expression,
    '```',
    ''
  ].join('\n');

  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, md);
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-blackbox-relu2d-decompile',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      terms: model.terms.length,
      mae: metrics.mae,
      r2: metrics.r2,
      maxDeltaError
    },
    notes: [
      'This benchmark tests 2D black-box ReLU circuit decompilation from oracle calls only.',
      'It recovers non-vertical kink lines visible from horizontal slices, not arbitrary high-dimensional networks.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
