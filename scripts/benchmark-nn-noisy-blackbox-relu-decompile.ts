#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileNoisyBlackBoxRelu1D,
  predictNoisyBlackBoxReluModel
} from '../src/solvers/noisy-blackbox-relu-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu-decompile', 'smoke');

const teacher = {
  hidden: [
    { weight: 1.0, bias: 1.4, readout: 1.1 },
    { weight: 1.0, bias: -0.25, readout: -0.7 },
    { weight: 1.0, bias: -1.2, readout: 0.45 },
    { weight: 1.0, bias: 2.15, readout: -0.3 },
    { weight: 1.0, bias: -2.35, readout: 0.22 }
  ],
  baseSlope: 0.35,
  bias: 0.25
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

function cleanTeacher(x: number): number {
  return teacher.bias
    + teacher.baseSlope * x
    + teacher.hidden.reduce((sum, unit) => sum + unit.readout * Math.max(0, unit.weight * x + unit.bias), 0);
}

function makeNoisyOracle(seed: number, noiseStd: number, quantization: number): (x: number) => number {
  const random = makeRandom(seed);
  return (x: number) => {
    const noisy = cleanTeacher(x) + gaussian(random) * noiseStd;
    return quantization > 0 ? Math.round(noisy / quantization) * quantization : noisy;
  };
}

function fitLinear(xs: number[], ys: number[]): { intercept: number; slope: number; predicted: number[] } {
  const xMean = xs.reduce((sum, value) => sum + value, 0) / xs.length;
  const yMean = ys.reduce((sum, value) => sum + value, 0) / ys.length;
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < xs.length; i++) {
    numerator += (xs[i] - xMean) * (ys[i] - yMean);
    denominator += (xs[i] - xMean) ** 2;
  }
  const slope = denominator > 0 ? numerator / denominator : 0;
  const intercept = yMean - slope * xMean;
  return { intercept, slope, predicted: xs.map(x => intercept + slope * x) };
}

function makeGrid(count: number, min: number, max: number, offset = 0): number[] {
  return Array.from({ length: count }, (_, index) => min + offset + ((max - min - 2 * offset) * index) / (count - 1));
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const sampleMin = Number(process.env.GCL_NOISY_BLACKBOX_RELU_MIN ?? -3);
  const sampleMax = Number(process.env.GCL_NOISY_BLACKBOX_RELU_MAX ?? 3);
  const samples = Math.max(80, Number(process.env.GCL_NOISY_BLACKBOX_RELU_SAMPLES ?? 421));
  const querySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU_QUERY_SAMPLES ?? 41));
  const noiseStd = Number(process.env.GCL_NOISY_BLACKBOX_RELU_NOISE_STD ?? 0.015);
  const quantization = Number(process.env.GCL_NOISY_BLACKBOX_RELU_QUANTIZATION ?? 0.005);
  const seed = Number(process.env.GCL_NOISY_BLACKBOX_RELU_SEED ?? 260523);
  const testXs = makeGrid(801, sampleMin, sampleMax, 0.0013);
  const actual = testXs.map(cleanTeacher);
  const linear = fitLinear(testXs, actual);
  const linearMetrics = regressionMetrics(actual, linear.predicted);
  const model = decompileNoisyBlackBoxRelu1D(
    makeNoisyOracle(seed, noiseStd, quantization),
    {
      min: sampleMin,
      max: sampleMax,
      maxTerms: teacher.hidden.length,
      samples,
      querySamples,
      aggregation: 'median',
      minSegmentPoints: 20
    }
  );
  const predicted = testXs.map(x => predictNoisyBlackBoxReluModel(model, x));
  const metrics = regressionMetrics(actual, predicted);
  const expectedKinks = teacher.hidden.map(unit => -unit.bias / unit.weight).sort((a, b) => a - b);
  const recoveredKinks = model.terms.map(term => term.kink).sort((a, b) => a - b);
  const maxKinkError = expectedKinks.reduce((maxError, expectedKink, index) => {
    const recovered = recoveredKinks[index];
    return Math.max(maxError, recovered === undefined ? Infinity : Math.abs(expectedKink - recovered));
  }, 0);
  const pass = model.terms.length === expectedKinks.length
    && maxKinkError < 0.06
    && metrics.mae < 0.02
    && metrics.r2 > 0.999
    && metrics.mae < linearMetrics.mae * 0.15;

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'noise-tolerant black-box ReLU circuit decompilation',
    blackBox: {
      type: '1-input/5-hidden ReLU MLP teacher with noisy quantized oracle',
      visibilityToGcl: 'oracle calls only: repeated noisy f(x), no hidden activations and no teacher weights'
    },
    noise: { seed, noiseStd, quantization, querySamples, aggregation: 'median' },
    config: { sampleMin, sampleMax, samples, testRows: testXs.length, maxTerms: teacher.hidden.length },
    auditTeacherCircuit: teacher,
    baselines: {
      linear: {
        expression: `${linear.intercept} + ${linear.slope} * x`,
        metrics: linearMetrics
      }
    },
    gcl: {
      expression: model.expression,
      baseIntercept: model.baseIntercept,
      baseSlope: model.baseSlope,
      terms: model.terms,
      plateaus: model.plateaus,
      metrics
    },
    gates: {
      pass,
      expectedKinks,
      recoveredKinks,
      maxKinkError,
      allKinksRecoveredWithin006: model.terms.length === expectedKinks.length && maxKinkError < 0.06,
      maeBelow002: metrics.mae < 0.02,
      r2Above0999: metrics.r2 > 0.999,
      beatsLinearBy85Percent: metrics.mae < linearMetrics.mae * 0.15
    },
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Noisy Black-Box ReLU Decompile Benchmark',
    '',
    '- Teacher: fixed 1-input/5-hidden ReLU MLP',
    '- GCL access: repeated noisy oracle calls, `f(x)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Noise std: ${noiseStd}`,
    `- Quantization step: ${quantization}`,
    `- Query samples per x: ${querySamples}`,
    `- Sample rows: ${samples}`,
    `- Test rows: ${testXs.length}`,
    '',
    '| Model | Test MAE vs clean teacher | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Linear baseline | ${linearMetrics.mae.toExponential(6)} | ${linearMetrics.rmse.toExponential(6)} | ${linearMetrics.r2.toFixed(9)} |`,
    `| GCL noisy black-box ReLU decompiler | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(9)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Kinks recovered within 0.06: ${report.gates.allKinksRecoveredWithin006 ? 'yes' : 'no'} (${maxKinkError.toExponential(6)})`,
    `- MAE < 0.02: ${report.gates.maeBelow002 ? 'yes' : 'no'} (${metrics.mae.toExponential(6)})`,
    `- R2 > 0.999: ${report.gates.r2Above0999 ? 'yes' : 'no'} (${metrics.r2.toFixed(12)})`,
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
    lane: 'nn-noisy-blackbox-relu-decompile',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      terms: model.terms.length,
      mae: metrics.mae,
      r2: metrics.r2,
      maxKinkError,
      noiseStd,
      quantization,
      querySamples
    },
    notes: [
      'This benchmark tests noisy and quantized black-box ReLU decompilation from repeated oracle calls only.',
      'It is a 1D robustness gate, not arbitrary production API decompilation.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
