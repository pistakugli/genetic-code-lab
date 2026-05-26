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
  profile: 'near-parallel' | 'beta-collision' | 'cancellation' | 'oblique-wide' | 'low-amplitude-mixed';
  bias: number;
  base: [number, number, number];
  hidden: TeacherTerm[];
  requiresSliceImprovement?: boolean;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-noisy-blackbox-relu3d-adversarial', 'sweep');

const teachers: Teacher[] = [
  {
    id: 'near-parallel-separated-offsets',
    profile: 'near-parallel',
    bias: 0.08,
    base: [0.16, -0.07, 0.04],
    hidden: [
      { alphaY: -0.26, alphaZ: 0.18, beta: -0.95, readout: 0.78 },
      { alphaY: -0.22, alphaZ: 0.16, beta: 0.05, readout: -0.46 },
      { alphaY: -0.18, alphaZ: 0.14, beta: 1.05, readout: 0.32 }
    ]
  },
  {
    id: 'same-offset-crossing-hyperplanes',
    profile: 'beta-collision',
    bias: -0.04,
    base: [-0.12, 0.05, 0.03],
    hidden: [
      { alphaY: 0.34, alphaZ: -0.22, beta: -0.15, readout: 0.64 },
      { alphaY: -0.31, alphaZ: 0.25, beta: -0.12, readout: -0.42 },
      { alphaY: 0.05, alphaZ: 0.10, beta: 0.95, readout: 0.25 }
    ]
  },
  {
    id: 'close-opposite-kink-pair-a',
    profile: 'cancellation',
    bias: 0.12,
    base: [0.18, -0.09, 0.06],
    hidden: [
      { alphaY: -0.344, alphaZ: -0.135, beta: -0.351, readout: 0.786 },
      { alphaY: 0.21045, alphaZ: -0.23107, beta: 0.47786, readout: -0.54557 },
      { alphaY: 0.07408, alphaZ: -0.05761, beta: 0.44405, readout: 0.38307 }
    ],
    requiresSliceImprovement: true
  },
  {
    id: 'close-opposite-kink-pair-b',
    profile: 'cancellation',
    bias: -0.08,
    base: [-0.14, 0.07, -0.05],
    hidden: [
      { alphaY: 0.28, alphaZ: 0.18, beta: -0.62, readout: 0.72 },
      { alphaY: -0.12, alphaZ: 0.23, beta: 0.28, readout: -0.50 },
      { alphaY: -0.02, alphaZ: 0.08, beta: 0.33, readout: 0.36 }
    ],
    requiresSliceImprovement: true
  },
  {
    id: 'oblique-wide-orientation-spread',
    profile: 'oblique-wide',
    bias: 0.03,
    base: [0.11, 0.04, -0.08],
    hidden: [
      { alphaY: -0.48, alphaZ: 0.42, beta: -0.70, readout: 0.68 },
      { alphaY: 0.52, alphaZ: -0.37, beta: 0.18, readout: -0.52 },
      { alphaY: -0.12, alphaZ: -0.44, beta: 0.85, readout: 0.31 }
    ]
  },
  {
    id: 'low-amplitude-opposite-mixed',
    profile: 'low-amplitude-mixed',
    bias: 0.10,
    base: [0.07, -0.04, 0.03],
    hidden: [
      { alphaY: -0.30, alphaZ: 0.22, beta: -0.80, readout: 0.72 },
      { alphaY: 0.12, alphaZ: -0.18, beta: 0.25, readout: -0.22 },
      { alphaY: 0.04, alphaZ: -0.10, beta: 0.38, readout: 0.16 }
    ],
    requiresSliceImprovement: true
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

function makeCountedNoisyOracle(
  teacher: Teacher,
  seed: number,
  noiseStd: number,
  quantization: number
): { oracle: (x: number, y: number, z: number) => number; getCalls: () => number } {
  const random = makeRandom(seed);
  let calls = 0;
  return {
    oracle: (x: number, y: number, z: number) => {
      calls += 1;
      const noisy = cleanTeacher(teacher, x, y, z) + gaussian(random) * noiseStd;
      return quantization > 0 ? Math.round(noisy / quantization) * quantization : noisy;
    },
    getCalls: () => calls
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
  const noiseStd = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_ADVERSARIAL_NOISE_STD ?? 0.005);
  const quantization = Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_ADVERSARIAL_QUANTIZATION ?? 0.002);
  const sliceQuerySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_ADVERSARIAL_SLICE_QUERY_SAMPLES ?? 17));
  const refinerQuerySamples = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_ADVERSARIAL_REFINER_QUERY_SAMPLES ?? 13));
  const restarts = Math.max(1, Number(process.env.GCL_NOISY_BLACKBOX_RELU3D_ADVERSARIAL_RESTARTS ?? 80));
  const inputs = makeGrid();

  const results = teachers.map((teacher, index) => {
    const actual = inputs.map(([x, y, z]) => cleanTeacher(teacher, x, y, z));
    const hyperplane = fitHyperplane(inputs, actual);
    const hyperplaneMetrics = regressionMetrics(actual, hyperplane.predicted);
    const sliceOracle = makeCountedNoisyOracle(teacher, 710000 + index * 1009, noiseStd, quantization);
    const sliceModel = decompileNoisyBlackBoxRelu3D(
      sliceOracle.oracle,
      {
        xRange,
        yRange,
        zRange,
        maxTerms: teacher.hidden.length,
        sliceCount: 7,
        ySliceCount: 11,
        samplesPerSlice: 301,
        querySamples: sliceQuerySamples,
        minSegmentPoints: 16,
        clusterSlopeTolerance: 0.11,
        kinkLineTolerance: 0.12,
        hyperplaneTolerance: 0.13
      }
    );
    const slicePredicted = predictNoisyBlackBoxRelu3DBatch(sliceModel, inputs);
    const sliceMetrics = regressionMetrics(actual, slicePredicted);
    const refinerOracle = makeCountedNoisyOracle(teacher, 810000 + index * 1009, noiseStd, quantization);
    const refined = refineNoisyBlackBoxRelu3DGlobal(
      refinerOracle.oracle,
      {
        xRange,
        yRange,
        zRange,
        termCount: teacher.hidden.length,
        initialModel: sliceModel,
        sampleCounts: [9, 7, 7],
        querySamples: refinerQuerySamples,
        restarts,
        coordinatePasses: 2,
        randomSeed: 130000 + index * 997,
        ridge: 1e-8,
        alphaYRange: [-0.75, 0.75],
        alphaZRange: [-0.70, 0.70],
        betaRange: [-1.9, 1.9]
      }
    );
    const refinedPredicted = inputs.map(([x, y, z]) => predictNoisyBlackBoxRelu3DGlobalModel(refined.model, x, y, z));
    const refinedMetrics = regressionMetrics(actual, refinedPredicted);
    const sliceImprovementOk = teacher.requiresSliceImprovement
      ? refinedMetrics.mae < sliceMetrics.mae * 0.45
      : refinedMetrics.mae <= Math.max(sliceMetrics.mae * 1.10, 0.006);
    const pass = refined.model.terms.length === teacher.hidden.length
      && refinedMetrics.mae < 0.020
      && refinedMetrics.r2 > 0.9985
      && refinedMetrics.mae < hyperplaneMetrics.mae * 0.25
      && sliceImprovementOk;
    return {
      teacherId: teacher.id,
      profile: teacher.profile,
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
      queryBudget: {
        sliceOracleCalls: sliceOracle.getCalls(),
        refinerOracleCalls: refinerOracle.getCalls(),
        totalOracleCalls: sliceOracle.getCalls() + refinerOracle.getCalls(),
        sliceQuerySamples,
        refinerQuerySamples,
        testRows: inputs.length
      },
      gates: {
        termCountMatches: refined.model.terms.length === teacher.hidden.length,
        maeBelow002: refinedMetrics.mae < 0.020,
        r2Above09985: refinedMetrics.r2 > 0.9985,
        beatsLinearBy75Percent: refinedMetrics.mae < hyperplaneMetrics.mae * 0.25,
        sliceImprovementOrNoRegression: sliceImprovementOk
      }
    };
  });

  const pass = results.every(result => result.pass);
  const summary = {
    pass,
    teacherCount: results.length,
    passedTeachers: results.filter(result => result.pass).length,
    profiles: Array.from(new Set(results.map(result => result.profile))),
    maxRefinedMae: Math.max(...results.map(result => result.gclGlobalRefiner.metrics.mae)),
    minRefinedR2: Math.min(...results.map(result => result.gclGlobalRefiner.metrics.r2)),
    maxSliceOnlyMae: Math.max(...results.map(result => result.baselines.sliceOnlyStitching.metrics.mae)),
    maxLinearMae: Math.max(...results.map(result => result.baselines.linearHyperplane.metrics.mae)),
    worstRefinedVsSliceRatio: Math.max(...results.map(result => result.gclGlobalRefiner.metrics.mae / Math.max(1e-12, result.baselines.sliceOnlyStitching.metrics.mae))),
    totalOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.totalOracleCalls, 0),
    maxOracleCallsPerTeacher: Math.max(...results.map(result => result.queryBudget.totalOracleCalls))
  };

  const report = {
    generatedAt: new Date().toISOString(),
    task: 'noisy 3D black-box ReLU adversarial sweep with query-budget accounting',
    blackBox: {
      type: '3-input/3-hidden ReLU MLP teachers with adversarial 3D hyperplane geometry, stochastic output noise, and quantization',
      visibilityToGcl: 'oracle calls only: repeated noisy f(x,y,z), no hidden activations and no teacher weights'
    },
    claimScope: {
      proves: 'output-equivalent symbolic 3D ReLU surface recovery across a wider adversarial shallow-3D sweep with explicit query-budget accounting',
      doesNotClaim: 'general-ND recovery, deep ReLU composition recovery, or superiority over external piecewise-linear extraction methods'
    },
    noise: { noiseStd, quantization, aggregation: 'median' },
    config: { xRange, yRange, zRange, restarts, sliceQuerySamples, refinerQuerySamples, testRowsPerTeacher: inputs.length },
    summary,
    results,
    runtimeMs: Date.now() - started
  };

  const md = [
    '# NN Noisy 3D Black-Box ReLU Adversarial Sweep',
    '',
    '- Teachers: 3-input/3-hidden ReLU MLPs with adversarial shallow-3D geometry',
    '- Profiles: near-parallel, beta-collision, cancellation, oblique-wide, low-amplitude-mixed',
    '- GCL access: repeated noisy oracle calls, `f(x,y,z)`',
    '- Hidden activations passed to GCL: no',
    '- Teacher weights passed to GCL: no',
    '- Query budget is counted from actual oracle calls inside slice stitching and global refinement.',
    '',
    '## Claim Scope',
    '',
    '- Proves: output-equivalent transparent symbolic 3D ReLU surface recovery across this adversarial shallow-3D sweep.',
    '- Does not claim: general-ND recovery, deep ReLU composition recovery, or superiority over external piecewise-linear extraction methods.',
    '',
    '| Teacher | Profile | Pass | Slice MAE | Refined MAE | Refined R2 | Oracle calls |',
    '|---|---|---|---:|---:|---:|---:|',
    ...results.map(result => `| ${result.teacherId} | ${result.profile} | ${result.pass ? 'yes' : 'no'} | ${result.baselines.sliceOnlyStitching.metrics.mae.toExponential(6)} | ${result.gclGlobalRefiner.metrics.mae.toExponential(6)} | ${result.gclGlobalRefiner.metrics.r2.toFixed(12)} | ${result.queryBudget.totalOracleCalls} |`),
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Passed teachers: ${summary.passedTeachers}/${summary.teacherCount}`,
    `- Max refined MAE: ${summary.maxRefinedMae.toExponential(6)}`,
    `- Min refined R2: ${summary.minRefinedR2.toFixed(12)}`,
    `- Worst refined/slice MAE ratio: ${summary.worstRefinedVsSliceRatio.toExponential(6)}`,
    `- Total oracle calls: ${summary.totalOracleCalls}`,
    `- Max oracle calls per teacher: ${summary.maxOracleCallsPerTeacher}`,
    ''
  ].join('\n');

  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, md);
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-noisy-blackbox-relu3d-adversarial',
    status: pass ? 'pass' : 'fail',
    summary,
    notes: [
      'This benchmark expands the noisy 3D black-box ReLU lane beyond smoke and two cancellation teachers.',
      'It also introduces explicit oracle query-budget accounting for the shallow-3D adversarial sweep.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
