#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileNoisyBlackBoxRelu3D,
  predictNoisyBlackBoxRelu3DBatch
} from '../src/solvers/noisy-blackbox-relu3d-decompiler';
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
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu3d-decompile', 'smoke');

const teacher = {
  bias: 0.18,
  base: [0.22, -0.11, 0.07],
  hidden: [
    { alphaY: -0.25, alphaZ: 0.20, beta: -1.10, readout: 0.82 },
    { alphaY: 0.35, alphaZ: -0.30, beta: 0.15, readout: -0.48 },
    { alphaY: -0.10, alphaZ: -0.22, beta: 1.20, readout: 0.31 }
  ] satisfies TeacherTerm[]
};

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

function cleanTeacher(x: number, y: number, z: number): number {
  return teacher.bias
    + teacher.base[0] * x
    + teacher.base[1] * y
    + teacher.base[2] * z
    + teacher.hidden.reduce(
      (sum, unit) => sum + unit.readout * Math.max(0, x - unit.alphaY * y - unit.alphaZ * z - unit.beta),
      0
    );
}

function makeNoisyOracle(seed: number, noiseStd: number, quantization: number): (x: number, y: number, z: number) => number {
  const random = makeRandom(seed);
  return (x: number, y: number, z: number) => {
    const noisy = cleanTeacher(x, y, z) + gaussian(random) * noiseStd;
    return quantization > 0 ? Math.round(noisy / quantization) * quantization : noisy;
  };
}

function makeGrid(): Array<[number, number, number]> {
  const inputs: Array<[number, number, number]> = [];
  for (let i = 0; i < 23; i++) {
    for (let j = 0; j < 15; j++) {
      for (let k = 0; k < 13; k++) {
        inputs.push([
          -2.8 + (5.6 * i) / 22,
          -1.3 + (2.6 * j) / 14,
          -1.1 + (2.2 * k) / 12
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
  const seed = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_SEED ?? 260527);
  const noiseStd = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_NOISE_STD ?? 0.006);
  const quantization = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_QUANTIZATION ?? 0.002);
  const sliceCount = Math.max(5, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_SLICE_COUNT ?? 7));
  const ySliceCount = Math.max(5, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_Y_SLICE_COUNT ?? 9));
  const samplesPerSlice = Math.max(80, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_SAMPLES_PER_SLICE ?? 261));
  const querySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_QUERY_SAMPLES ?? 17));
  const model = decompileNoisyBlackBoxRelu3D(
    makeNoisyOracle(seed, noiseStd, quantization),
    {
      xRange,
      yRange,
      zRange,
      maxTerms: teacher.hidden.length,
      sliceCount,
      ySliceCount,
      samplesPerSlice,
      querySamples,
      minSegmentPoints: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_MIN_SEGMENT ?? 16),
      clusterSlopeTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_CLUSTER_TOL ?? 0.10),
      kinkLineTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_KINK_LINE_TOL ?? 0.11),
      hyperplaneTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_HYPERPLANE_TOL ?? 0.12)
    }
  );
  const inputs = makeGrid();
  const actual = inputs.map(([x, y, z]) => cleanTeacher(x, y, z));
  const predicted = predictNoisyBlackBoxRelu3DBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const hyperplane = fitHyperplane(inputs, actual);
  const hyperplaneMetrics = regressionMetrics(actual, hyperplane.predicted);
  const geometry = maxGeometryError(model.terms);
  const pass = model.terms.length === teacher.hidden.length
    && geometry.maxDeltaError < 0.12
    && geometry.maxAlphaYError < 0.12
    && geometry.maxAlphaZError < 0.12
    && geometry.maxBetaError < 0.12
    && metrics.mae < 0.08
    && metrics.r2 > 0.995
    && metrics.mae < hyperplaneMetrics.mae * 0.35;

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'noise-tolerant 3D black-box ReLU hyperplane decompilation',
    blackBox: {
      type: '3-input/3-hidden non-vertical ReLU MLP teacher with noisy quantized oracle',
      visibilityToGcl: 'oracle calls only: repeated noisy f(x,y,z), no hidden activations and no teacher weights'
    },
    noise: { seed, noiseStd, quantization, querySamples, aggregation: 'median' },
    config: { xRange, yRange, zRange, sliceCount, ySliceCount, samplesPerSlice, testRows: inputs.length, maxTerms: teacher.hidden.length },
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
      allHyperplanesRecoveredWithin012: model.terms.length === teacher.hidden.length
        && geometry.maxDeltaError < 0.12
        && geometry.maxAlphaYError < 0.12
        && geometry.maxAlphaZError < 0.12
        && geometry.maxBetaError < 0.12,
      maeBelow008: metrics.mae < 0.08,
      r2Above0995: metrics.r2 > 0.995,
      beatsLinearHyperplaneBy65Percent: metrics.mae < hyperplaneMetrics.mae * 0.35
    },
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Noisy 3D Black-Box ReLU Decompile Benchmark',
    '',
    '- Teacher: fixed 3-input/3-hidden non-vertical ReLU MLP',
    '- GCL access: repeated noisy oracle calls, `f(x,y,z)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Noise std: ${noiseStd}`,
    `- Quantization step: ${quantization}`,
    `- Query samples per x,y,z: ${querySamples}`,
    `- Z slices: ${sliceCount}`,
    `- Y slices per z-slice: ${ySliceCount}`,
    `- Samples per horizontal line: ${samplesPerSlice}`,
    `- Test rows: ${inputs.length}`,
    '',
    '| Model | Test MAE vs clean teacher | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Linear hyperplane baseline | ${hyperplaneMetrics.mae.toExponential(6)} | ${hyperplaneMetrics.rmse.toExponential(6)} | ${hyperplaneMetrics.r2.toFixed(9)} |`,
    `| GCL noisy 3D black-box ReLU decompiler | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(9)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- All hyperplanes recovered within 0.12: ${report.gates.allHyperplanesRecoveredWithin012 ? 'yes' : 'no'}`,
    `- Max delta-slope error: ${geometry.maxDeltaError.toExponential(6)}`,
    `- Max alphaY error: ${geometry.maxAlphaYError.toExponential(6)}`,
    `- Max alphaZ error: ${geometry.maxAlphaZError.toExponential(6)}`,
    `- Max beta error: ${geometry.maxBetaError.toExponential(6)}`,
    `- MAE < 0.08: ${report.gates.maeBelow008 ? 'yes' : 'no'} (${metrics.mae.toExponential(6)})`,
    `- R2 > 0.995: ${report.gates.r2Above0995 ? 'yes' : 'no'} (${metrics.r2.toFixed(12)})`,
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
    lane: 'nn-noisy-blackbox-relu3d-decompile',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      terms: model.terms.length,
      mae: metrics.mae,
      r2: metrics.r2,
      ...geometry
    },
    notes: [
      'This benchmark tests 3D black-box ReLU hyperplane stitching from repeated noisy and quantized oracle calls only.',
      'It recovers non-vertical hyperplanes visible through z-indexed noisy 2D slices, not arbitrary high-dimensional, deep, or production API decompilation.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
