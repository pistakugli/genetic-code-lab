#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileBlackBoxRelu1D,
  predictBlackBoxReluBatch
} from '../src/solvers/blackbox-relu-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-blackbox-relu-decompile', 'smoke');

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

function relu(value: number): number {
  return Math.max(0, value);
}

function blackBoxTeacher(x: number): number {
  return teacher.bias
    + teacher.baseSlope * x
    + teacher.hidden.reduce((sum, unit) => sum + unit.readout * relu(unit.weight * x + unit.bias), 0);
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
  return {
    intercept,
    slope,
    predicted: xs.map(x => intercept + slope * x)
  };
}

function makeGrid(count: number, min: number, max: number, offset = 0): number[] {
  return Array.from({ length: count }, (_, index) => min + offset + ((max - min - 2 * offset) * index) / (count - 1));
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const sampleMin = Number(process.env.GCL_BLACKBOX_RELU_MIN ?? -3);
  const sampleMax = Number(process.env.GCL_BLACKBOX_RELU_MAX ?? 3);
  const samples = Math.max(64, Number(process.env.GCL_BLACKBOX_RELU_SAMPLES ?? 1201));
  const testXs = makeGrid(701, sampleMin, sampleMax, 0.0017);
  const actual = testXs.map(blackBoxTeacher);
  const linear = fitLinear(testXs, actual);
  const linearMetrics = regressionMetrics(actual, linear.predicted);

  const model = decompileBlackBoxRelu1D(blackBoxTeacher, {
    min: sampleMin,
    max: sampleMax,
    samples,
    slopeTolerance: Number(process.env.GCL_BLACKBOX_RELU_SLOPE_TOL ?? 1e-8),
    minPlateauIntervals: Number(process.env.GCL_BLACKBOX_RELU_MIN_PLATEAU ?? 8)
  });
  const predicted = predictBlackBoxReluBatch(model, testXs);
  const metrics = regressionMetrics(actual, predicted);
  const expectedKinks = teacher.hidden.map(unit => -unit.bias / unit.weight).sort((a, b) => a - b);
  const recoveredKinks = model.terms.map(term => term.kink).sort((a, b) => a - b);
  const maxKinkError = expectedKinks.reduce((maxError, expectedKink, index) => {
    const recovered = recoveredKinks[index];
    return Math.max(maxError, recovered === undefined ? Infinity : Math.abs(expectedKink - recovered));
  }, 0);
  const pass = model.terms.length === expectedKinks.length
    && maxKinkError < 1e-8
    && metrics.mae < 1e-9
    && metrics.r2 > 0.999999999
    && metrics.mae < linearMetrics.mae * 0.001;

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'black-box ReLU circuit decompilation',
    blackBox: {
      type: '1-input/5-hidden ReLU MLP teacher',
      visibilityToGcl: 'oracle calls only: the decompiler receives f(x), not hidden activations or teacher weights'
    },
    config: {
      sampleMin,
      sampleMax,
      samples,
      testRows: testXs.length
    },
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
      allKinksRecovered: model.terms.length === expectedKinks.length && maxKinkError < 1e-8,
      maeBelow1eMinus9: metrics.mae < 1e-9,
      r2Above999999999: metrics.r2 > 0.999999999,
      beatsLinearBy1000x: metrics.mae < linearMetrics.mae * 0.001
    },
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Black-Box ReLU Decompile Benchmark',
    '',
    '- Teacher: fixed 1-input/5-hidden ReLU MLP',
    '- GCL access: oracle calls only, `f(x)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Sample rows: ${samples}`,
    `- Test rows: ${testXs.length}`,
    '',
    '| Model | Test MAE | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Linear baseline | ${linearMetrics.mae.toExponential(6)} | ${linearMetrics.rmse.toExponential(6)} | ${linearMetrics.r2.toFixed(9)} |`,
    `| GCL black-box ReLU decompiler | ${metrics.mae.toExponential(6)} | ${metrics.rmse.toExponential(6)} | ${metrics.r2.toFixed(9)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- All kinks recovered: ${report.gates.allKinksRecovered ? 'yes' : 'no'}`,
    `- Max kink error: ${maxKinkError.toExponential(6)}`,
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
    lane: 'nn-blackbox-relu-decompile',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      terms: model.terms.length,
      mae: metrics.mae,
      r2: metrics.r2,
      maxKinkError
    },
    notes: [
      'This benchmark tests black-box ReLU circuit decompilation from oracle calls only.',
      'It recovers a canonical equivalent 1D ReLU circuit, not hidden activations and not arbitrary high-dimensional networks.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
