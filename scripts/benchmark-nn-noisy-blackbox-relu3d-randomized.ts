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

type Teacher = {
  seed: number;
  bias: number;
  base: [number, number, number];
  hidden: TeacherTerm[];
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu3d-randomized', 'seed-sweep');

function makeRandom(seed: number): () => number {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeTeacher(seed: number): Teacher {
  const random = makeRandom(seed);
  const readoutTemplates = [0.78, -0.52, 0.36];
  const betaTemplates = [-0.82, 0.02, 0.88];
  const hidden = readoutTemplates.map((readout, index) => ({
    alphaY: randBetween(random, -0.34, 0.34),
    alphaZ: randBetween(random, -0.30, 0.30),
    beta: betaTemplates[index] + randBetween(random, -0.07, 0.07),
    readout: readout + (index % 2 === 0 ? 1 : -1) * randBetween(random, 0.012, 0.036)
  }));
  return {
    seed,
    bias: randBetween(random, -0.22, 0.22),
    base: [
      randBetween(random, -0.30, 0.30),
      randBetween(random, -0.18, 0.18),
      randBetween(random, -0.15, 0.15)
    ],
    hidden
  };
}

function cleanTeacher(teacher: Teacher, x: number, y: number, z: number): number {
  return teacher.bias
    + teacher.base[0] * x
    + teacher.base[1] * y
    + teacher.base[2] * z
    + teacher.hidden.reduce(
      (sum, unit) => sum + unit.readout * Math.max(0, x - unit.alphaY * y - unit.alphaZ * z - unit.beta),
      0
    );
}

function makeNoisyOracle(
  teacher: Teacher,
  seed: number,
  noiseStd: number,
  quantization: number
): (x: number, y: number, z: number) => number {
  const random = makeRandom(seed);
  return (x: number, y: number, z: number) => {
    const noisy = cleanTeacher(teacher, x, y, z) + gaussian(random) * noiseStd;
    return quantization > 0 ? Math.round(noisy / quantization) * quantization : noisy;
  };
}

function makeGrid(): Array<[number, number, number]> {
  const inputs: Array<[number, number, number]> = [];
  for (let i = 0; i < 21; i++) {
    for (let j = 0; j < 15; j++) {
      for (let k = 0; k < 13; k++) {
        inputs.push([
          -2.75 + (5.5 * i) / 20,
          -1.25 + (2.5 * j) / 14,
          -1.05 + (2.1 * k) / 12
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
  teacher: Teacher,
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

function parseSeeds(): number[] {
  const raw = process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_SEEDS ?? '101,2026,4242,7777,31415';
  return raw.split(',')
    .map(value => Number(value.trim()))
    .filter(value => Number.isInteger(value));
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const xRange: [number, number] = [-3, 3];
  const yRange: [number, number] = [-1.35, 1.35];
  const zRange: [number, number] = [-1.15, 1.15];
  const seeds = parseSeeds();
  const noiseStd = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_NOISE_STD ?? 0.005);
  const quantization = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_QUANTIZATION ?? 0.002);
  const sliceCount = Math.max(5, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_SLICE_COUNT ?? 7));
  const ySliceCount = Math.max(5, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_Y_SLICE_COUNT ?? 11));
  const samplesPerSlice = Math.max(80, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_SAMPLES_PER_SLICE ?? 301));
  const querySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_QUERY_SAMPLES ?? 21));
  const inputs = makeGrid();

  const results = seeds.map(seed => {
    const teacher = makeTeacher(seed);
    const model = decompileNoisyBlackBoxRelu3D(
      makeNoisyOracle(teacher, seed + 900001, noiseStd, quantization),
      {
        xRange,
        yRange,
        zRange,
        maxTerms: teacher.hidden.length,
        sliceCount,
        ySliceCount,
        samplesPerSlice,
        querySamples,
        minSegmentPoints: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_MIN_SEGMENT ?? 16),
        clusterSlopeTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_CLUSTER_TOL ?? 0.10),
        kinkLineTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_KINK_LINE_TOL ?? 0.11),
        hyperplaneTolerance: Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_RANDOM_HYPERPLANE_TOL ?? 0.12)
      }
    );
    const actual = inputs.map(([x, y, z]) => cleanTeacher(teacher, x, y, z));
    const predicted = predictNoisyBlackBoxRelu3DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const hyperplane = fitHyperplane(inputs, actual);
    const hyperplaneMetrics = regressionMetrics(actual, hyperplane.predicted);
    const geometry = maxGeometryError(teacher, model.terms);
    const pass = model.terms.length === teacher.hidden.length
      && geometry.maxDeltaError < 0.12
      && geometry.maxAlphaYError < 0.12
      && geometry.maxAlphaZError < 0.12
      && geometry.maxBetaError < 0.12
      && metrics.mae < 0.08
      && metrics.r2 > 0.995
      && metrics.mae < hyperplaneMetrics.mae * 0.40;
    return {
      seed,
      pass,
      teacher,
      baselines: {
        hyperplane: { metrics: hyperplaneMetrics }
      },
      gcl: {
        expression: model.expression,
        terms: model.terms,
        metrics
      },
      gates: geometry
    };
  });

  const pass = results.every(result => result.pass);
  const summary = {
    pass,
    seedCount: results.length,
    passedSeeds: results.filter(result => result.pass).length,
    maxMae: Math.max(...results.map(result => result.gcl.metrics.mae)),
    minR2: Math.min(...results.map(result => result.gcl.metrics.r2)),
    maxDeltaError: Math.max(...results.map(result => result.gates.maxDeltaError)),
    maxAlphaYError: Math.max(...results.map(result => result.gates.maxAlphaYError)),
    maxAlphaZError: Math.max(...results.map(result => result.gates.maxAlphaZError)),
    maxBetaError: Math.max(...results.map(result => result.gates.maxBetaError))
  };

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'randomized noisy 3D black-box ReLU hyperplane decompilation seed sweep',
    blackBox: {
      type: 'randomized 3-input/3-hidden non-vertical ReLU MLP teachers with noisy quantized oracle',
      visibilityToGcl: 'oracle calls only: repeated noisy f(x,y,z), no hidden activations and no teacher weights'
    },
    teacherProfile: 'non-degenerate randomized: beta templates are separated so this gate does not conflate ordinary randomization with near-collision/cancellation hard cases',
    noise: { noiseStd, quantization, querySamples, aggregation: 'median' },
    config: { seeds, xRange, yRange, zRange, sliceCount, ySliceCount, samplesPerSlice, testRowsPerSeed: inputs.length },
    summary,
    results,
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Randomized Noisy 3D Black-Box ReLU Decompile Benchmark',
    '',
    '- Teachers: randomized 3-input/3-hidden non-vertical ReLU MLPs',
    '- Teacher profile: non-degenerate randomized, separated beta templates',
    '- GCL access: repeated noisy oracle calls, `f(x,y,z)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Seeds: ${seeds.join(', ')}`,
    `- Noise std: ${noiseStd}`,
    `- Quantization step: ${quantization}`,
    `- Query samples per x,y,z: ${querySamples}`,
    `- Z slices: ${sliceCount}`,
    `- Y slices per z-slice: ${ySliceCount}`,
    `- Samples per horizontal line: ${samplesPerSlice}`,
    `- Test rows per seed: ${inputs.length}`,
    '',
    '| Seed | Pass | Terms | MAE | R2 | Max geom error |',
    '|---:|---|---:|---:|---:|---:|',
    ...results.map(result => {
      const maxGeom = Math.max(
        result.gates.maxDeltaError,
        result.gates.maxAlphaYError,
        result.gates.maxAlphaZError,
        result.gates.maxBetaError
      );
      return `| ${result.seed} | ${result.pass ? 'yes' : 'no'} | ${result.gcl.terms.length} | ${result.gcl.metrics.mae.toExponential(6)} | ${result.gcl.metrics.r2.toFixed(12)} | ${maxGeom.toExponential(6)} |`;
    }),
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Passed seeds: ${summary.passedSeeds}/${summary.seedCount}`,
    `- Max MAE: ${summary.maxMae.toExponential(6)}`,
    `- Min R2: ${summary.minR2.toFixed(12)}`,
    `- Max delta-slope error: ${summary.maxDeltaError.toExponential(6)}`,
    `- Max alphaY error: ${summary.maxAlphaYError.toExponential(6)}`,
    `- Max alphaZ error: ${summary.maxAlphaZError.toExponential(6)}`,
    `- Max beta error: ${summary.maxBetaError.toExponential(6)}`,
    ''
  ].join('\n');

  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, md);
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-noisy-blackbox-relu3d-randomized',
    status: pass ? 'pass' : 'fail',
    summary,
    notes: [
      'This benchmark tests 3D black-box ReLU hyperplane stitching across deterministic randomized teachers with repeated noisy and quantized oracle calls only.',
      'It reduces fixed-teacher risk for noisy 3D recovery but still does not prove arbitrary ND, deep, adversarial near-collision, or production API decompilation.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
