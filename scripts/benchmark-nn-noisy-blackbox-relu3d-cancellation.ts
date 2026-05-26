#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileNoisyBlackBoxRelu3D,
  predictNoisyBlackBoxRelu3DBatch
} from '../src/solvers/noisy-blackbox-relu3d-decompiler';
import {
  refineNoisyBlackBoxRelu3DGlobal,
  predictNoisyBlackBoxRelu3DGlobalModel
} from '../src/solvers/noisy-blackbox-relu3d-global-refiner';
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
  id: string;
  bias: number;
  base: [number, number, number];
  hidden: TeacherTerm[];
  note: string;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu3d-cancellation', 'hard-case');

const teachers: Teacher[] = [
  {
    id: 'close-opposite-kink-pair-a',
    bias: 0.12,
    base: [0.18, -0.09, 0.06],
    hidden: [
      { alphaY: -0.344, alphaZ: -0.135, beta: -0.351, readout: 0.786 },
      { alphaY: 0.21045, alphaZ: -0.23107, beta: 0.47786, readout: -0.54557 },
      { alphaY: 0.07408, alphaZ: -0.05761, beta: 0.44405, readout: 0.38307 }
    ],
    note: 'Two nearby opposite-sign kinks partially cancel, causing slice-only noisy stitching to miss structure.'
  },
  {
    id: 'close-opposite-kink-pair-b',
    bias: -0.08,
    base: [-0.14, 0.07, -0.05],
    hidden: [
      { alphaY: 0.28, alphaZ: 0.18, beta: -0.62, readout: 0.72 },
      { alphaY: -0.12, alphaZ: 0.23, beta: 0.28, readout: -0.50 },
      { alphaY: -0.02, alphaZ: 0.08, beta: 0.33, readout: 0.36 }
    ],
    note: 'Second close opposite-sign pair with different orientation checks that the fix is not seed-specific.'
  }
];

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

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const started = Date.now();
  const xRange: [number, number] = [-3, 3];
  const yRange: [number, number] = [-1.35, 1.35];
  const zRange: [number, number] = [-1.15, 1.15];
  const noiseStd = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_CANCELLATION_NOISE_STD ?? 0.005);
  const quantization = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_CANCELLATION_QUANTIZATION ?? 0.002);
  const querySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_CANCELLATION_QUERY_SAMPLES ?? 15));
  const restarts = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_CANCELLATION_RESTARTS ?? 96));
  const inputs = makeGrid();

  const results = teachers.map((teacher, index) => {
    const actual = inputs.map(([x, y, z]) => cleanTeacher(teacher, x, y, z));
    const hyperplane = fitHyperplane(inputs, actual);
    const hyperplaneMetrics = regressionMetrics(actual, hyperplane.predicted);
    const sliceModel = decompileNoisyBlackBoxRelu3D(
      makeNoisyOracle(teacher, 810000 + index * 917, noiseStd, quantization),
      {
        xRange,
        yRange,
        zRange,
        maxTerms: teacher.hidden.length,
        sliceCount: 7,
        ySliceCount: 11,
        samplesPerSlice: 301,
        querySamples: 21,
        minSegmentPoints: 16,
        clusterSlopeTolerance: 0.10,
        kinkLineTolerance: 0.11,
        hyperplaneTolerance: 0.12
      }
    );
    const slicePredicted = predictNoisyBlackBoxRelu3DBatch(sliceModel, inputs);
    const sliceMetrics = regressionMetrics(actual, slicePredicted);
    const refined = refineNoisyBlackBoxRelu3DGlobal(
      makeNoisyOracle(teacher, 910000 + index * 917, noiseStd, quantization),
      {
        xRange,
        yRange,
        zRange,
        termCount: teacher.hidden.length,
        initialModel: sliceModel,
        sampleCounts: [9, 7, 7],
        querySamples,
        restarts,
        coordinatePasses: 2,
        randomSeed: 120000 + index * 997,
        ridge: 1e-8,
        alphaYRange: [-0.65, 0.65],
        alphaZRange: [-0.60, 0.60],
        betaRange: [-1.8, 1.8]
      }
    );
    const refinedPredicted = inputs.map(([x, y, z]) => predictNoisyBlackBoxRelu3DGlobalModel(refined.model, x, y, z));
    const refinedMetrics = regressionMetrics(actual, refinedPredicted);
    const pass = refined.model.terms.length === teacher.hidden.length
      && refinedMetrics.mae < 0.012
      && refinedMetrics.r2 > 0.999
      && refinedMetrics.mae < hyperplaneMetrics.mae * 0.15
      && refinedMetrics.mae < sliceMetrics.mae * 0.30;
    return {
      teacherId: teacher.id,
      pass,
      teacher,
      baselines: {
        linearHyperplane: { metrics: hyperplaneMetrics },
        sliceOnlyStitching: {
          expression: sliceModel.expression,
          terms: sliceModel.terms,
          metrics: sliceMetrics
        }
      },
      gclGlobalRefiner: {
        expression: refined.model.expression,
        terms: refined.model.terms,
        metrics: refinedMetrics,
        trainingMse: refined.trainingMse,
        sampleCount: refined.sampleCount,
        restarts: refined.restarts,
        objectiveEvaluations: refined.objectiveEvaluations
      },
      gates: {
        termCountMatches: refined.model.terms.length === teacher.hidden.length,
        maeBelow0012: refinedMetrics.mae < 0.012,
        r2Above0999: refinedMetrics.r2 > 0.999,
        beatsLinearBy85Percent: refinedMetrics.mae < hyperplaneMetrics.mae * 0.15,
        beatsSliceOnlyBy70Percent: refinedMetrics.mae < sliceMetrics.mae * 0.30
      }
    };
  });

  const pass = results.every(result => result.pass);
  const summary = {
    pass,
    teacherCount: results.length,
    passedTeachers: results.filter(result => result.pass).length,
    maxRefinedMae: Math.max(...results.map(result => result.gclGlobalRefiner.metrics.mae)),
    minRefinedR2: Math.min(...results.map(result => result.gclGlobalRefiner.metrics.r2)),
    maxSliceOnlyMae: Math.max(...results.map(result => result.baselines.sliceOnlyStitching.metrics.mae)),
    maxLinearMae: Math.max(...results.map(result => result.baselines.linearHyperplane.metrics.mae)),
    worstRefinedVsSliceRatio: Math.max(...results.map(result => result.gclGlobalRefiner.metrics.mae / Math.max(1e-12, result.baselines.sliceOnlyStitching.metrics.mae)))
  };

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'noisy 3D black-box ReLU near-collision/cancellation hard-case recovery',
    blackBox: {
      type: '3-input/3-hidden ReLU MLP teachers with close opposite-sign kink pairs, stochastic output noise, and quantization',
      visibilityToGcl: 'oracle calls only: repeated noisy f(x,y,z), no hidden activations and no teacher weights'
    },
    claimScope: {
      proves: 'transparent output-equivalent symbolic 3D ReLU surface recovery on noisy near-cancellation cases that break slice-only stitching',
      doesNotClaim: 'unique recovery of the original hidden-unit parameterization for cancelling near-collision pairs; equivalent circuits can be functionally indistinguishable under noisy black-box outputs'
    },
    noise: { noiseStd, quantization, querySamples, aggregation: 'median' },
    config: { xRange, yRange, zRange, restarts, testRowsPerTeacher: inputs.length },
    summary,
    results,
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Noisy 3D Black-Box ReLU Cancellation Hard-Case Benchmark',
    '',
    '- Teachers: 3-input/3-hidden ReLU MLPs with close opposite-sign kink pairs',
    '- GCL access: repeated noisy oracle calls, `f(x,y,z)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    '- Fix under test: slice-only stitching plus global parametric ReLU surface refinement',
    '',
    '## Claim Scope',
    '',
    '- Proves: output-equivalent transparent symbolic 3D ReLU surface recovery on noisy near-cancellation cases.',
    '- Does not claim: unique recovery of the original hidden-unit parameterization for cancelling near-collision pairs.',
    '',
    '| Teacher | Pass | Slice MAE | Refined MAE | Refined R2 | Refined/Slice MAE |',
    '|---|---|---:|---:|---:|---:|',
    ...results.map(result => {
      const ratio = result.gclGlobalRefiner.metrics.mae / Math.max(1e-12, result.baselines.sliceOnlyStitching.metrics.mae);
      return `| ${result.teacherId} | ${result.pass ? 'yes' : 'no'} | ${result.baselines.sliceOnlyStitching.metrics.mae.toExponential(6)} | ${result.gclGlobalRefiner.metrics.mae.toExponential(6)} | ${result.gclGlobalRefiner.metrics.r2.toFixed(12)} | ${ratio.toExponential(6)} |`;
    }),
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Passed teachers: ${summary.passedTeachers}/${summary.teacherCount}`,
    `- Max refined MAE: ${summary.maxRefinedMae.toExponential(6)}`,
    `- Min refined R2: ${summary.minRefinedR2.toFixed(12)}`,
    `- Worst refined/slice MAE ratio: ${summary.worstRefinedVsSliceRatio.toExponential(6)}`,
    ''
  ].join('\n');

  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, md);
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-noisy-blackbox-relu3d-cancellation',
    status: pass ? 'pass' : 'fail',
    summary,
    notes: [
      'This benchmark promotes the previously exposed noisy 3D near-collision/cancellation failure into a release-gated hard case.',
      'The gate checks functional symbolic surface recovery and improvement over slice-only stitching; it intentionally does not claim unique hidden-unit identity for cancelling pairs.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
