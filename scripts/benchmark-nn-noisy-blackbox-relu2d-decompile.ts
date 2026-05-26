#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileNoisyBlackBoxRelu2D,
  predictNoisyBlackBoxRelu2DBatch
} from '../src/solvers/noisy-blackbox-relu2d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu2d-decompile', 'smoke');

const profile = process.env.GCL_NOISY_BLACKBOX_RELU2D_PROFILE === 'crossing'
  ? 'crossing'
  : process.env.GCL_NOISY_BLACKBOX_RELU2D_PROFILE === 'near-parallel'
    ? 'near-parallel'
    : process.env.GCL_NOISY_BLACKBOX_RELU2D_PROFILE === 'near-collision'
      ? 'near-collision'
    : 'non-crossing';
const teacher = profile === 'crossing'
  ? {
    bias: 0.2,
    base: [0.3, -0.15],
    hidden: [
      { alphaY: -0.4, beta: -1.2, readout: 0.9 },
      { alphaY: 0.7, beta: 0.15, readout: -0.55 },
      { alphaY: -0.2, beta: 1.1, readout: 0.35 }
    ]
  }
  : profile === 'near-parallel'
    ? {
      bias: 0.15,
      base: [0.28, -0.12],
      hidden: [
        { alphaY: -0.28, beta: -1.15, readout: 0.85 },
        { alphaY: -0.24, beta: 0.05, readout: -0.50 },
        { alphaY: -0.20, beta: 1.28, readout: 0.32 }
      ]
    }
  : profile === 'near-collision'
    ? {
      bias: 0.12,
      base: [0.25, -0.10],
      hidden: [
        { alphaY: -0.35, beta: -0.2, readout: 0.75 },
        { alphaY: 0.35, beta: -0.2, readout: -0.45 },
        { alphaY: -0.10, beta: 1.25, readout: 0.28 }
      ]
    }
  : {
    bias: 0.2,
    base: [0.3, -0.15],
    hidden: [
      { alphaY: -0.35, beta: -1.25, readout: 0.9 },
      { alphaY: 0.25, beta: 0.15, readout: -0.55 },
      { alphaY: -0.15, beta: 1.25, readout: 0.35 }
    ]
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

function cleanTeacher(x: number, y: number): number {
  return teacher.bias
    + teacher.base[0] * x
    + teacher.base[1] * y
    + teacher.hidden.reduce(
      (sum, unit) => sum + unit.readout * Math.max(0, x - unit.alphaY * y - unit.beta),
      0
    );
}

function makeNoisyOracle(seed: number, noiseStd: number, quantization: number): (x: number, y: number) => number {
  const random = makeRandom(seed);
  return (x: number, y: number) => {
    const noisy = cleanTeacher(x, y) + gaussian(random) * noiseStd;
    return quantization > 0 ? Math.round(noisy / quantization) * quantization : noisy;
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

function maxGeometryError(
  recoveredTerms: Array<{ alphaY: number; beta: number; deltaSlopeX: number }>
): { maxDeltaError: number; maxAlphaError: number; maxBetaError: number } {
  const expected = [...teacher.hidden].sort((a, b) => a.readout - b.readout);
  const recovered = [...recoveredTerms].sort((a, b) => a.deltaSlopeX - b.deltaSlopeX);
  return expected.reduce((acc, item, index) => {
    const term = recovered[index];
    return {
      maxDeltaError: Math.max(acc.maxDeltaError, Math.abs(item.readout - (term?.deltaSlopeX ?? Infinity))),
      maxAlphaError: Math.max(acc.maxAlphaError, Math.abs(item.alphaY - (term?.alphaY ?? Infinity))),
      maxBetaError: Math.max(acc.maxBetaError, Math.abs(item.beta - (term?.beta ?? Infinity)))
    };
  }, { maxDeltaError: 0, maxAlphaError: 0, maxBetaError: 0 });
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const xRange: [number, number] = [-3, 3];
  const yRange: [number, number] = [-1.5, 1.5];
  const seed = Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_SEED ?? 260524);
  const noiseStd = Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_NOISE_STD ?? 0.01);
  const quantization = Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_QUANTIZATION ?? 0.0025);
  const samplesPerSlice = Math.max(80, Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_SAMPLES_PER_SLICE ?? 321));
  const querySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_QUERY_SAMPLES ?? 25));
  const model = decompileNoisyBlackBoxRelu2D(
    makeNoisyOracle(seed, noiseStd, quantization),
    {
      xRange,
      yRange,
      maxTerms: teacher.hidden.length,
      samplesPerSlice,
      querySamples,
      minSegmentPoints: Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_MIN_SEGMENT ?? 18),
      clusterSlopeTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU2D_CLUSTER_TOL ?? (profile === 'near-collision' ? 0.09 : 0.08))
    }
  );
  const inputs = makeGrid();
  const actual = inputs.map(([x, y]) => cleanTeacher(x, y));
  const predicted = predictNoisyBlackBoxRelu2DBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const plane = fitPlane(inputs, actual);
  const planeMetrics = regressionMetrics(actual, plane.predicted);
  const { maxDeltaError, maxAlphaError, maxBetaError } = maxGeometryError(model.terms);
  const pass = model.terms.length === teacher.hidden.length
    && maxDeltaError < 0.08
    && maxAlphaError < 0.08
    && maxBetaError < 0.08
    && metrics.mae < 0.04
    && metrics.r2 > 0.998
    && metrics.mae < planeMetrics.mae * 0.2;

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'noise-tolerant 2D black-box ReLU circuit decompilation',
    blackBox: {
      type: `2-input/3-hidden ${profile} non-vertical ReLU MLP teacher with noisy quantized oracle`,
      visibilityToGcl: 'oracle calls only: repeated noisy f(x,y), no hidden activations and no teacher weights'
    },
    profile,
    noise: { seed, noiseStd, quantization, querySamples, aggregation: 'median' },
    config: { xRange, yRange, samplesPerSlice, testRows: inputs.length, maxTerms: teacher.hidden.length },
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
      maxDeltaError,
      maxAlphaError,
      maxBetaError,
      allKinkLinesRecoveredWithin008: model.terms.length === teacher.hidden.length
        && maxDeltaError < 0.08
        && maxAlphaError < 0.08
        && maxBetaError < 0.08,
      maeBelow004: metrics.mae < 0.04,
      r2Above0998: metrics.r2 > 0.998,
      beatsPlaneBy80Percent: metrics.mae < planeMetrics.mae * 0.2
    },
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Noisy 2D Black-Box ReLU Decompile Benchmark',
    '',
    `- Teacher: fixed 2-input/3-hidden ${profile} non-vertical ReLU MLP`,
    '- GCL access: repeated noisy oracle calls, `f(x,y)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Noise std: ${noiseStd}`,
    `- Quantization step: ${quantization}`,
    `- Query samples per x,y: ${querySamples}`,
    `- Samples per horizontal slice: ${samplesPerSlice}`,
    `- Test rows: ${inputs.length}`,
    '',
    '| Model | Test MAE vs clean teacher | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Plane baseline | ${planeMetrics.mae.toExponential(6)} | ${planeMetrics.rmse.toExponential(6)} | ${planeMetrics.r2.toFixed(9)} |`,
    `| GCL noisy 2D black-box ReLU decompiler | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(9)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Kink lines recovered within 0.08: ${report.gates.allKinkLinesRecoveredWithin008 ? 'yes' : 'no'}`,
    `- Max delta error: ${maxDeltaError.toExponential(6)}`,
    `- Max alpha error: ${maxAlphaError.toExponential(6)}`,
    `- Max beta error: ${maxBetaError.toExponential(6)}`,
    `- MAE < 0.04: ${report.gates.maeBelow004 ? 'yes' : 'no'} (${metrics.mae.toExponential(6)})`,
    `- R2 > 0.998: ${report.gates.r2Above0998 ? 'yes' : 'no'} (${metrics.r2.toFixed(12)})`,
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
    lane: 'nn-noisy-blackbox-relu2d-decompile',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      terms: model.terms.length,
      mae: metrics.mae,
      r2: metrics.r2,
      maxDeltaError,
      maxAlphaError,
      maxBetaError,
      noiseStd,
      quantization,
      querySamples
    },
    notes: [
      'This benchmark tests noisy and quantized 2D black-box ReLU decompilation from repeated oracle calls only.',
      `It recovers ${profile} non-vertical kink-line geometry; this is not arbitrary high-dimensional network decompilation.`
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
