#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileGeluFfnStructure,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type TeacherAtom = {
  direction: number[];
  inputScale: number;
  inputBias: number;
  outputWeight: number;
};

type Teacher = {
  id: string;
  dimension: number;
  atoms: TeacherAtom[];
  outputBias: number;
  predict: (input: number[]) => number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-structure-finder',
  'controlled-single-and-two-channel'
);

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
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

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function makeDirection(seed: number, dimension: number, previous: number[][] = []): number[] {
  const random = makeRandom(seed);
  let vector = Array.from({ length: dimension }, () => gaussian(random));
  for (const direction of previous) {
    const projection = dot(vector, direction);
    vector = vector.map((value, index) => value - projection * direction[index]);
  }
  return normalize(vector);
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => -1.5 + 3 * random())
  ));
}

function makeTeacher(id: string, dimension: number, seed: number, atomSpecs: Array<Omit<TeacherAtom, 'direction'>>, outputBias: number): Teacher {
  const directions: number[][] = [];
  for (let index = 0; index < atomSpecs.length; index++) {
    directions.push(makeDirection(seed + index * 101, dimension, directions));
  }
  const atoms = atomSpecs.map((spec, index) => ({ ...spec, direction: directions[index] }));
  return {
    id,
    dimension,
    atoms,
    outputBias,
    predict: input => outputBias + atoms.reduce((sum, atom) => (
      sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
    ), 0)
  };
}

function bestDirectionProjection(trueDirection: number[], recoveredWeights: number[][]): number {
  return Math.max(...recoveredWeights.map(weight => Math.abs(dot(trueDirection, normalize(weight)))));
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const cases = [
  {
    teacher: makeTeacher('gelu-structure-single-96d', 96, 8101, [
      { inputScale: 1.45, inputBias: -0.22, outputWeight: 1.35 }
    ], 0.17),
    config: { gradientProbeCount: 10, trainingSampleCount: 1600, validationSampleCount: 700, restarts: 8, iterations: 1000, learningRate: 0.035 },
    gates: { minR2: 0.999, maxMae: 0.015, minProjection: 0.995, maxWeightNormError: 0.14, maxBiasError: 0.14 }
  },
  {
    teacher: makeTeacher('gelu-structure-two-channel-96d', 96, 8201, [
      { inputScale: 1.25, inputBias: -0.18, outputWeight: 0.92 },
      { inputScale: 1.55, inputBias: 0.24, outputWeight: -0.74 }
    ], -0.08),
    config: { gradientProbeCount: 14, trainingSampleCount: 2600, validationSampleCount: 800, restarts: 12, iterations: 1200, learningRate: 0.028 },
    gates: { minR2: 0.992, maxMae: 0.06, minProjection: 0.985, maxWeightNormError: 0.35, maxBiasError: 0.35 }
  }
];

const results = cases.map(testCase => {
  const teacher = testCase.teacher;
  const model = decompileGeluFfnStructure(teacher.predict, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-1.5, 1.5] as [number, number]),
    channelCount: teacher.atoms.length,
    activeRank: teacher.atoms.length,
    randomSeed: 910000 + teacher.dimension + teacher.atoms.length,
    ...testCase.config
  });
  const validationInputs = makeInputs(730000 + teacher.dimension + teacher.atoms.length, teacher.dimension, 1000);
  const actual = validationInputs.map(teacher.predict);
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const recoveredWeights = model.atoms.map(atom => atom.inputWeight);
  const projectionScores = teacher.atoms.map(atom => bestDirectionProjection(atom.direction, recoveredWeights));
  const normErrors = teacher.atoms.map(atom => (
    Math.min(...model.atoms.map(recovered => Math.abs(recovered.inputWeightNorm - atom.inputScale)))
  ));
  const biasErrors = teacher.atoms.map(atom => (
    Math.min(...model.atoms.map(recovered => Math.abs(Math.abs(recovered.inputBias) - Math.abs(atom.inputBias))))
  ));
  const maxWeightNormError = Math.max(...normErrors);
  const maxBiasError = Math.max(...biasErrors);
  const minProjection = Math.min(...projectionScores);
  const meanBaseline = regressionMetrics(actual, actual.map(() => actual.reduce((sum, value) => sum + value, 0) / actual.length));
  return {
    teacherId: teacher.id,
    dimension: teacher.dimension,
    channelCount: teacher.atoms.length,
    thresholds: testCase.gates,
    gcl: {
      metrics,
      expression: model.expression,
      atoms: model.atoms.map(atom => ({
        inputWeightNorm: atom.inputWeightNorm,
        inputBias: atom.inputBias,
        outputWeight: atom.outputWeight
      })),
      structure: {
        projectionScores,
        minProjection,
        maxWeightNormError,
        maxBiasError
      },
      optimizer: model.optimizer,
      queryBudget: model.queryBudget
    },
    baselines: [
      {
        id: 'mean-output',
        family: 'constant',
        metrics: meanBaseline
      }
    ],
    pass: metrics.r2 >= testCase.gates.minR2
      && metrics.mae <= testCase.gates.maxMae
      && minProjection >= testCase.gates.minProjection
      && maxWeightNormError <= testCase.gates.maxWeightNormError
      && maxBiasError <= testCase.gates.maxBiasError
      && metrics.mae < meanBaseline.mae
  };
});

const summary = {
  pass: true,
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  maxMae: Math.max(...results.map(result => result.gcl.metrics.mae)),
  minR2: Math.min(...results.map(result => result.gcl.metrics.r2)),
  minDirectionProjection: Math.min(...results.map(result => result.gcl.structure.minProjection)),
  maxWeightNormError: Math.max(...results.map(result => result.gcl.structure.maxWeightNormError)),
  maxBiasError: Math.max(...results.map(result => result.gcl.structure.maxBiasError)),
  maxRawOracleCalls: Math.max(...results.map(result => result.gcl.queryBudget.rawOracleCalls))
};

const gates = {
  pass: true,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  coversSingleAndTwoChannel: results.some(result => result.channelCount === 1) && results.some(result => result.channelCount === 2),
  minR2AboveThresholds: results.every(result => result.gcl.metrics.r2 >= result.thresholds.minR2),
  maxMaeBelowThresholds: results.every(result => result.gcl.metrics.mae <= result.thresholds.maxMae),
  directionProjectionAboveThresholds: results.every(result => result.gcl.structure.minProjection >= result.thresholds.minProjection),
  structuralParameterErrorsBelowThresholds: results.every(result => (
    result.gcl.structure.maxWeightNormError <= result.thresholds.maxWeightNormError
    && result.gcl.structure.maxBiasError <= result.thresholds.maxBiasError
  )),
  beatsMeanBaseline: results.every(result => result.gcl.metrics.mae < result.baselines[0].metrics.mae)
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'GELU FFN structural finder for controlled high-dimensional channels',
  blackBox: {
    type: '96D single/two-channel GELU FFN teachers',
    visibilityToGcl: 'oracle calls only: f(vector), no weights, no hidden activations'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit GELU atom structure for controlled high-dimensional one/two-channel FFN teachers, including directions and approximate atom parameters',
    doesNotClaim: 'full GPT-2 FFN extraction, arbitrary channel count recovery, exact parameter identity under non-identifiability, SwiGLU recovery, or broad SOTA status'
  },
  gates: { ...gates, pass },
  summary: { ...summary, pass },
  results,
  conclusion: pass
    ? 'GELU FFN structural finder gate passed for controlled single/two-channel high-dimensional teachers; next step is GPT-2 top-K structural stress and rank selection.'
    : 'GELU FFN structural finder gate failed; inspect the recorded structural errors before promoting.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# GELU FFN Structure Finder Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Summary',
  '',
  `- Pass: ${pass}`,
  `- Teachers: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Min direction projection: ${summary.minDirectionProjection}`,
  `- Max weight-norm error: ${summary.maxWeightNormError}`,
  `- Max bias error: ${summary.maxBiasError}`,
  `- Max raw oracle calls: ${summary.maxRawOracleCalls}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Pass: ${result.pass}`,
    `- Dimension: ${result.dimension}`,
    `- Channel count: ${result.channelCount}`,
    `- MAE: ${result.gcl.metrics.mae}`,
    `- R2: ${result.gcl.metrics.r2}`,
    `- Min direction projection: ${result.gcl.structure.minProjection}`,
    `- Max weight-norm error: ${result.gcl.structure.maxWeightNormError}`,
    `- Max bias error: ${result.gcl.structure.maxBiasError}`,
    `- Raw oracle calls: ${result.gcl.queryBudget.rawOracleCalls}`,
    ''
  ].join('\n')),
  '## Conclusion',
  '',
  report.conclusion
].join('\n').trimEnd();

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-structure-finder',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'First explicit structural FFN finder lane for GELU atoms.',
    'This is not a KNN/polynomial surrogate gate; it records recovered atom directions and approximate atom parameters.',
    'This is not a full GPT-2 FFN extraction claim.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates,
  conclusion: report.conclusion
}, null, 2));

if (!pass) process.exit(1);
