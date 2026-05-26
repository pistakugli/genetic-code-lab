#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileBlackBoxRelu3D,
  predictBlackBoxRelu3DBatch
} from '../src/solvers/blackbox-relu3d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type TeacherTerm = {
  alphaY: number;
  alphaZ: number;
  beta: number;
  readout: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-blackbox-relu3d-decompile', 'smoke');

const teacher = {
  bias: 0.18,
  base: [0.22, -0.11, 0.07],
  hidden: [
    { alphaY: -0.25, alphaZ: 0.20, beta: -1.10, readout: 0.82 },
    { alphaY: 0.35, alphaZ: -0.30, beta: 0.15, readout: -0.48 },
    { alphaY: -0.10, alphaZ: -0.22, beta: 1.20, readout: 0.31 }
  ] satisfies TeacherTerm[]
};

function blackBoxTeacher(x: number, y: number, z: number): number {
  return teacher.bias
    + teacher.base[0] * x
    + teacher.base[1] * y
    + teacher.base[2] * z
    + teacher.hidden.reduce(
      (sum, unit) => sum + unit.readout * Math.max(0, x - unit.alphaY * y - unit.alphaZ * z - unit.beta),
      0
    );
}

function makeGrid(): Array<[number, number, number]> {
  const inputs: Array<[number, number, number]> = [];
  for (let i = 0; i < 31; i++) {
    for (let j = 0; j < 21; j++) {
      for (let k = 0; k < 17; k++) {
        inputs.push([
          -2.85 + (5.7 * i) / 30,
          -1.33 + (2.66 * j) / 20,
          -1.14 + (2.28 * k) / 16
        ]);
      }
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

function fitHyperplane(
  inputs: Array<[number, number, number]>,
  targets: number[]
): { coefficients: [number, number, number, number]; predicted: number[] } {
  const xtx = Array.from({ length: 4 }, () => new Array(4).fill(0));
  const xty = new Array(4).fill(0);
  for (let i = 0; i < inputs.length; i++) {
    const row = [1, inputs[i][0], inputs[i][1], inputs[i][2]];
    for (let a = 0; a < row.length; a++) {
      xty[a] += row[a] * targets[i];
      for (let b = 0; b < row.length; b++) xtx[a][b] += row[a] * row[b];
    }
  }
  const coefficients = solveSystem(xtx, xty) as [number, number, number, number];
  return {
    coefficients,
    predicted: inputs.map(([x, y, z]) => coefficients[0] + coefficients[1] * x + coefficients[2] * y + coefficients[3] * z)
  };
}

function maxGeometryError(
  recoveredTerms: Array<{ alphaY: number; alphaZ: number; beta: number; deltaSlopeX: number }>
): { maxDeltaError: number; maxAlphaYError: number; maxAlphaZError: number; maxBetaError: number } {
  const expected = [...teacher.hidden].sort((a, b) => a.readout - b.readout);
  const recovered = [...recoveredTerms].sort((a, b) => a.deltaSlopeX - b.deltaSlopeX);
  return expected.reduce((acc, item, index) => {
    const term = recovered[index];
    return {
      maxDeltaError: Math.max(acc.maxDeltaError, Math.abs(item.readout - (term?.deltaSlopeX ?? Infinity))),
      maxAlphaYError: Math.max(acc.maxAlphaYError, Math.abs(item.alphaY - (term?.alphaY ?? Infinity))),
      maxAlphaZError: Math.max(acc.maxAlphaZError, Math.abs(item.alphaZ - (term?.alphaZ ?? Infinity))),
      maxBetaError: Math.max(acc.maxBetaError, Math.abs(item.beta - (term?.beta ?? Infinity)))
    };
  }, { maxDeltaError: 0, maxAlphaYError: 0, maxAlphaZError: 0, maxBetaError: 0 });
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const xRange: [number, number] = [-3, 3];
  const yRange: [number, number] = [-1.4, 1.4];
  const zRange: [number, number] = [-1.2, 1.2];
  const sliceCount = Math.max(3, Number(process.env.GCL_BLACKBOX_RELU3D_SLICE_COUNT ?? 7));
  const samplesPerSlice = Math.max(64, Number(process.env.GCL_BLACKBOX_RELU3D_SAMPLES_PER_SLICE ?? 801));
  const model = decompileBlackBoxRelu3D(blackBoxTeacher, {
    xRange,
    yRange,
    zRange,
    sliceCount,
    samplesPerSlice,
    slopeTolerance: Number(process.env.GCL_BLACKBOX_RELU3D_SLOPE_TOL ?? 1e-8),
    clusterTolerance: Number(process.env.GCL_BLACKBOX_RELU3D_CLUSTER_TOL ?? 1e-7)
  });
  const inputs = makeGrid();
  const actual = inputs.map(([x, y, z]) => blackBoxTeacher(x, y, z));
  const predicted = predictBlackBoxRelu3DBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const hyperplane = fitHyperplane(inputs, actual);
  const hyperplaneMetrics = regressionMetrics(actual, hyperplane.predicted);
  const geometry = maxGeometryError(model.terms);
  const pass = model.terms.length === teacher.hidden.length
    && geometry.maxDeltaError < 1e-8
    && geometry.maxAlphaYError < 1e-8
    && geometry.maxAlphaZError < 1e-8
    && geometry.maxBetaError < 1e-8
    && metrics.mae < 1e-9
    && metrics.r2 > 0.999999999
    && metrics.mae < hyperplaneMetrics.mae * 0.001;

  const report = {
    generatedAt: new Date().toISOString(),
    task: '3D black-box ReLU hyperplane decompilation',
    blackBox: {
      type: '3-input/3-hidden non-vertical ReLU MLP teacher',
      visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y,z), not hidden activations or teacher weights'
    },
    config: {
      xRange,
      yRange,
      zRange,
      sliceCount,
      samplesPerSlice,
      testRows: inputs.length
    },
    auditTeacherCircuit: teacher,
    baselines: {
      hyperplane: {
        expression: `${hyperplane.coefficients[0]} + ${hyperplane.coefficients[1]} * x + ${hyperplane.coefficients[2]} * y + ${hyperplane.coefficients[3]} * z`,
        metrics: hyperplaneMetrics
      }
    },
    gcl: {
      expression: model.expression,
      baseIntercept: model.baseIntercept,
      baseSlopeX: model.baseSlopeX,
      baseSlopeY: model.baseSlopeY,
      baseSlopeZ: model.baseSlopeZ,
      terms: model.terms,
      metrics
    },
    gates: {
      pass,
      ...geometry,
      allHyperplanesRecovered: model.terms.length === teacher.hidden.length
        && geometry.maxDeltaError < 1e-8
        && geometry.maxAlphaYError < 1e-8
        && geometry.maxAlphaZError < 1e-8
        && geometry.maxBetaError < 1e-8,
      maeBelow1eMinus9: metrics.mae < 1e-9,
      r2Above999999999: metrics.r2 > 0.999999999,
      beatsLinearHyperplaneBy1000x: metrics.mae < hyperplaneMetrics.mae * 0.001
    },
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN 3D Black-Box ReLU Decompile Benchmark',
    '',
    '- Teacher: fixed 3-input/3-hidden non-vertical ReLU MLP',
    '- GCL access: oracle calls only, `f(x,y,z)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Z slices: ${sliceCount}`,
    `- Samples per 2D slice line: ${samplesPerSlice}`,
    `- Test rows: ${inputs.length}`,
    '',
    '| Model | Test MAE | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Linear hyperplane baseline | ${hyperplaneMetrics.mae.toExponential(6)} | ${hyperplaneMetrics.rmse.toExponential(6)} | ${hyperplaneMetrics.r2.toFixed(9)} |`,
    `| GCL 3D black-box ReLU decompiler | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(9)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- All hyperplanes recovered: ${report.gates.allHyperplanesRecovered ? 'yes' : 'no'}`,
    `- Max delta-slope error: ${geometry.maxDeltaError.toExponential(6)}`,
    `- Max alphaY error: ${geometry.maxAlphaYError.toExponential(6)}`,
    `- Max alphaZ error: ${geometry.maxAlphaZError.toExponential(6)}`,
    `- Max beta error: ${geometry.maxBetaError.toExponential(6)}`,
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
    lane: 'nn-blackbox-relu3d-decompile',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      terms: model.terms.length,
      mae: metrics.mae,
      r2: metrics.r2,
      ...geometry
    },
    notes: [
      'This benchmark tests 3D black-box ReLU hyperplane stitching from oracle calls only.',
      'It recovers non-vertical hyperplanes visible through z-indexed 2D slices, not arbitrary high-dimensional or deep networks.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
