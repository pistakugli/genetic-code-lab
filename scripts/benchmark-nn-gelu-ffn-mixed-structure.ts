#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileRankAdaptiveGeluFfnStructure,
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
  'nn-gelu-ffn-mixed-structure',
  'rank-adaptive-three-channel'
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

function subtractProjection(vector: number[], basis: number[][]): number[] {
  let output = [...vector];
  for (const direction of basis) {
    const projection = dot(output, direction);
    output = output.map((value, index) => value - projection * direction[index]);
  }
  return output;
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function makeDirection(seed: number, dimension: number, previous: number[][] = []): number[] {
  const random = makeRandom(seed);
  return normalize(subtractProjection(
    Array.from({ length: dimension }, () => gaussian(random)),
    previous
  ));
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => -1.5 + 3 * random())
  ));
}

function makeTeacher(): Teacher {
  const dimension = 32;
  const directions: number[][] = [];
  for (const seed of [7101, 7202, 7303]) directions.push(makeDirection(seed, dimension, directions));
  const specs = [
    { inputScale: 1.25, inputBias: -0.2, outputWeight: 0.9 },
    { inputScale: 1.05, inputBias: 0.15, outputWeight: -0.7 },
    { inputScale: 0.85, inputBias: 0.32, outputWeight: 0.5 }
  ];
  const atoms = specs.map((spec, index) => ({ ...spec, direction: directions[index] }));
  return {
    id: 'mixed-gelu-three-channel-32d',
    dimension,
    atoms,
    outputBias: 0.07,
    predict: input => 0.07 + atoms.reduce((sum, atom) => (
      sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
    ), 0)
  };
}

function bestDirectionProjection(trueDirection: number[], recoveredWeights: number[][]): number {
  return Math.max(...recoveredWeights.map(weight => Math.abs(dot(trueDirection, normalize(weight)))));
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teacher = makeTeacher();
const model = decompileRankAdaptiveGeluFfnStructure(teacher.predict, {
  dimension: teacher.dimension,
  ranges: Array.from({ length: teacher.dimension }, () => [-1.5, 1.5] as [number, number]),
  maxChannelCount: 4,
  gradientProbeCount: 18,
  trainingSampleCount: 1900,
  validationSampleCount: 450,
  randomSeed: 8811,
  restarts: 10,
  iterations: 700,
  learningRate: 0.03
});
const validationInputs = makeInputs(9501, teacher.dimension, 900);
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
const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
const meanBaseline = regressionMetrics(actual, actual.map(() => mean));
const minProjection = Math.min(...projectionScores);
const maxWeightNormError = Math.max(...normErrors);
const maxBiasError = Math.max(...biasErrors);

const gates = {
  recoversRankAdaptiveStructure: model.recoveryKind === 'rank-adaptive-gelu-ffn-structure',
  doesNotUseTrueChannelCount: model.rawChannelCount === 4 && model.maxChannelCount === 4,
  effectiveChannelCountMatches: model.effectiveChannelCount === teacher.atoms.length,
  usesInternalCandidateSelection: (model.adaptiveSelection?.candidateCount ?? 0) >= 4,
  duplicatePairsAbsent: (model.adaptiveSelection?.duplicatePairCount ?? 0) === 0,
  heldoutR2Above0999: metrics.r2 > 0.999,
  heldoutMaeBelow002: metrics.mae < 0.02,
  minDirectionProjectionAbove0995: minProjection > 0.995,
  structuralParameterErrorsBounded: maxWeightNormError < 0.18 && maxBiasError < 0.18,
  beatsMeanBy90Percent: metrics.mae < meanBaseline.mae * 0.1
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'rank-adaptive mixed-channel GELU FFN structure recovery',
  blackBox: {
    type: '32D three-channel GELU FFN block with overcomplete max-channel budget',
    visibilityToGcl: 'oracle calls only: f(vector); true channel count, directions, biases, and readout weights are hidden'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit mixed GELU atom structure when given a maximum channel budget but not the true channel count',
    doesNotClaim: 'full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, or broad SOTA status'
  },
  teacher: {
    id: teacher.id,
    dimension: teacher.dimension,
    trueChannelCount: teacher.atoms.length,
    maxChannelBudgetGivenToGcl: 4
  },
  gates: { ...gates, pass },
  summary: {
    pass,
    trueChannelCount: teacher.atoms.length,
    rawChannelCount: model.rawChannelCount,
    effectiveChannelCount: model.effectiveChannelCount,
    prunedWeakAtomCount: model.adaptiveSelection?.prunedWeakAtomCount ?? 0,
    duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
    mae: metrics.mae,
    r2: metrics.r2,
    minDirectionProjection: minProjection,
    maxWeightNormError,
    maxBiasError,
    rawOracleCalls: model.queryBudget.rawOracleCalls
  },
  baselines: [
    {
      id: 'mean-output',
      family: 'constant',
      metrics: meanBaseline
    }
  ],
  gcl: {
    metrics,
    expression: model.expression,
    atoms: model.atoms.map(atom => ({
      inputWeightNorm: atom.inputWeightNorm,
      inputBias: atom.inputBias,
      outputWeight: atom.outputWeight,
      strength: Math.abs(atom.outputWeight) * atom.inputWeightNorm
    })),
    adaptiveSelection: model.adaptiveSelection,
    structure: {
      projectionScores,
      minProjection,
      maxWeightNormError,
      maxBiasError
    },
    queryBudget: model.queryBudget
  },
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Rank-Adaptive Mixed GELU FFN Structure Benchmark',
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
  `- True channel count: ${report.summary.trueChannelCount}`,
  `- Max channel budget given to GCL: ${report.summary.rawChannelCount}`,
  `- Effective recovered channel count: ${report.summary.effectiveChannelCount}`,
  `- Pruned weak atoms: ${report.summary.prunedWeakAtomCount}`,
  `- Duplicate pairs: ${report.summary.duplicatePairCount}`,
  `- MAE: ${report.summary.mae}`,
  `- R2: ${report.summary.r2}`,
  `- Min direction projection: ${report.summary.minDirectionProjection}`,
  `- Max weight-norm error: ${report.summary.maxWeightNormError}`,
  `- Max bias error: ${report.summary.maxBiasError}`,
  `- Raw oracle calls: ${report.summary.rawOracleCalls}`,
  '',
  '## Gate',
  '',
  ...Object.entries(gates).map(([key, value]) => `- ${key}: ${value}`),
  '',
  '## Conclusion',
  '',
  pass
    ? 'The rank-adaptive mixed GELU structure gate passed. GCL recovered explicit atoms and the effective channel count without being told the true count.'
    : 'The rank-adaptive mixed GELU structure gate failed. Do not promote Stage C until the recorded failure is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-mixed-structure',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'First Stage C mixed-channel GELU structure gate.',
    'GCL is given only a max channel budget, not the true channel count.',
    'The solver internally compares smaller and overcomplete atom candidates and must recover the effective count.',
    'This gate fails if the recovered object is only an output surrogate without enough explicit atoms.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
