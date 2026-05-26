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

type Teacher = {
  seed: number;
  bias: number;
  base: [number, number, number];
  hidden: TeacherTerm[];
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-blackbox-relu3d-randomized', 'seed-sweep');

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

function makeTeacher(seed: number): Teacher {
  const random = makeRandom(seed);
  const readoutTemplates = [0.72, -0.54, 0.39, -0.28];
  const hidden = readoutTemplates.map((readout, index) => ({
    alphaY: randBetween(random, -0.42, 0.42),
    alphaZ: randBetween(random, -0.36, 0.36),
    beta: randBetween(random, -1.05, 1.05),
    readout: readout + (index % 2 === 0 ? 1 : -1) * randBetween(random, 0.015, 0.045)
  }));
  return {
    seed,
    bias: randBetween(random, -0.25, 0.25),
    base: [
      randBetween(random, -0.35, 0.35),
      randBetween(random, -0.22, 0.22),
      randBetween(random, -0.18, 0.18)
    ],
    hidden
  };
}

function evaluateTeacher(teacher: Teacher, x: number, y: number, z: number): number {
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
  for (let i = 0; i < 19; i++) {
    for (let j = 0; j < 13; j++) {
      for (let k = 0; k < 11; k++) {
        inputs.push([
          -2.75 + (5.5 * i) / 18,
          -1.2 + (2.4 * j) / 12,
          -1.05 + (2.1 * k) / 10
        ]);
      }
    }
  }
  return inputs;
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
  const raw = process.env.GCL_BLACKBOX_RELU3D_RANDOM_SEEDS ?? '101,2026,4242,7777,31415';
  return raw.split(',')
    .map(value => Number(value.trim()))
    .filter(value => Number.isInteger(value));
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const xRange: [number, number] = [-3, 3];
  const yRange: [number, number] = [-1.25, 1.25];
  const zRange: [number, number] = [-1.1, 1.1];
  const seeds = parseSeeds();
  const samplesPerSlice = Math.max(64, Number(process.env.GCL_BLACKBOX_RELU3D_RANDOM_SAMPLES_PER_SLICE ?? 801));
  const sliceCount = Math.max(5, Number(process.env.GCL_BLACKBOX_RELU3D_RANDOM_SLICE_COUNT ?? 7));
  const ySliceCount = Math.max(5, Number(process.env.GCL_BLACKBOX_RELU3D_RANDOM_Y_SLICE_COUNT ?? 21));
  const inputs = makeGrid();

  const results = seeds.map(seed => {
    const teacher = makeTeacher(seed);
    const model = decompileBlackBoxRelu3D(
      (x, y, z) => evaluateTeacher(teacher, x, y, z),
      {
        xRange,
        yRange,
        zRange,
        sliceCount,
        ySliceCount,
        samplesPerSlice,
        slopeTolerance: Number(process.env.GCL_BLACKBOX_RELU3D_RANDOM_SLOPE_TOL ?? 1e-8),
        clusterTolerance: Number(process.env.GCL_BLACKBOX_RELU3D_RANDOM_CLUSTER_TOL ?? 1e-7)
      }
    );
    const actual = inputs.map(([x, y, z]) => evaluateTeacher(teacher, x, y, z));
    const predicted = predictBlackBoxRelu3DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const geometry = maxGeometryError(teacher, model.terms);
    const pass = model.terms.length === teacher.hidden.length
      && geometry.maxDeltaError < 1e-8
      && geometry.maxAlphaYError < 1e-8
      && geometry.maxAlphaZError < 1e-8
      && geometry.maxBetaError < 1e-8
      && metrics.mae < 1e-9
      && metrics.r2 > 0.999999999;
    return {
      seed,
      pass,
      teacher,
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
    task: 'randomized 3D black-box ReLU hyperplane decompilation seed sweep',
    blackBox: {
      type: 'randomized 3-input/4-hidden non-vertical ReLU MLP teachers',
      visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y,z), not hidden activations or teacher weights'
    },
    config: { seeds, xRange, yRange, zRange, sliceCount, ySliceCount, samplesPerSlice, testRowsPerSeed: inputs.length },
    summary,
    results,
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Randomized 3D Black-Box ReLU Decompile Benchmark',
    '',
    '- Teachers: randomized 3-input/4-hidden non-vertical ReLU MLPs',
    '- GCL access: oracle calls only, `f(x,y,z)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    `- Seeds: ${seeds.join(', ')}`,
    `- Z slices: ${sliceCount}`,
    `- Y slices per z-slice: ${ySliceCount}`,
    `- Samples per 2D slice line: ${samplesPerSlice}`,
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
    lane: 'nn-blackbox-relu3d-randomized',
    status: pass ? 'pass' : 'fail',
    summary,
    notes: [
      'This benchmark tests clean 3D black-box ReLU hyperplane stitching across deterministic randomized teachers.',
      'It reduces hardcoded-teacher risk but still does not prove arbitrary ND, noisy randomized, or deep network decompilation.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
