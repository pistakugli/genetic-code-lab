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
  epsilon: number;
  dimension: number;
  atoms: TeacherAtom[];
  trueNearParallelCosine: number;
  predict: (input: number[]) => number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-near-parallel-family',
  'epsilon-sweep'
);
const epsilons = [1e-1, 1e-2, 1e-3, 1e-4];
const nearParallelDirectionCosine = 0.999;

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

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => -1.5 + 3 * random())
  ));
}

function makeTeacher(epsilon: number): Teacher {
  const dimension = 16;
  const random = makeRandom(9001);
  const first = normalize(Array.from({ length: dimension }, () => gaussian(random)));
  const perturbation = normalize(subtractProjection(
    Array.from({ length: dimension }, () => gaussian(random)),
    [first]
  ));
  const second = normalize(first.map((value, index) => value + epsilon * perturbation[index]));
  const third = normalize(subtractProjection(
    Array.from({ length: dimension }, () => gaussian(random)),
    [first, perturbation]
  ));
  const atoms = [
    { direction: first, inputScale: 1.2, inputBias: -0.42, outputWeight: 0.9 },
    { direction: second, inputScale: 1.05, inputBias: 0.28, outputWeight: -0.7 },
    { direction: third, inputScale: 0.85, inputBias: 0.35, outputWeight: 0.52 }
  ];
  return {
    id: `near-parallel-epsilon-${epsilon}`,
    epsilon,
    dimension,
    atoms,
    trueNearParallelCosine: Math.abs(dot(first, second)),
    predict: input => 0.05 + atoms.reduce((sum, atom) => (
      sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
    ), 0)
  };
}

function bestDirectionProjection(trueDirection: number[], recoveredWeights: number[][]): number {
  return Math.max(...recoveredWeights.map(weight => Math.abs(dot(trueDirection, normalize(weight)))));
}

function maxRecoveredCosine(recoveredWeights: number[][]): number {
  let maximum = 0;
  for (let left = 0; left < recoveredWeights.length; left++) {
    for (let right = left + 1; right < recoveredWeights.length; right++) {
      maximum = Math.max(maximum, Math.abs(dot(normalize(recoveredWeights[left]), normalize(recoveredWeights[right]))));
    }
  }
  return maximum;
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = epsilons.map((epsilon, index) => {
  const teacher = makeTeacher(epsilon);
  const model = decompileRankAdaptiveGeluFfnStructure(teacher.predict, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: 4,
    selectionRestarts: 2,
    gradientProbeCount: 18,
    trainingSampleCount: 1300,
    validationSampleCount: 360,
    randomSeed: 17000 + index * 131,
    restarts: 8,
    iterations: 520,
    learningRate: 0.03,
    duplicateDirectionCosine: nearParallelDirectionCosine,
    duplicateBiasTolerance: 0.12,
    nearParallelDirectionCosine,
    atomRelativeStrengthThreshold: 0.035,
    pruneValidationToleranceRatio: 0.35,
    pruneValidationToleranceAbsolute: 0.006
  });
  const validationInputs = makeInputs(18000 + index, teacher.dimension, 600);
  const actual = validationInputs.map(teacher.predict);
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const recoveredWeights = model.atoms.map(atom => atom.inputWeight);
  const projectionScores = teacher.atoms.map(atom => bestDirectionProjection(atom.direction, recoveredWeights));
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const meanBaseline = regressionMetrics(actual, actual.map(() => mean));
  const expectedNearParallelPairs = teacher.trueNearParallelCosine >= nearParallelDirectionCosine ? 1 : 0;
  const gates = {
    effectiveChannelCountMatches: model.effectiveChannelCount === teacher.atoms.length,
    duplicatePairsAbsent: (model.adaptiveSelection?.duplicatePairCount ?? 0) === 0,
    nearParallelDetectionMatches: (model.adaptiveSelection?.nearParallelPairCount ?? 0) >= expectedNearParallelPairs,
    directionRecoveryStable: Math.min(...projectionScores) > 0.999,
    heldoutR2Above09998: metrics.r2 > 0.9998,
    heldoutMaeBelow0006: metrics.mae < 0.006,
    beatsMeanBy95Percent: metrics.mae < meanBaseline.mae * 0.05,
    rawOracleCallsBelow25000: model.queryBudget.rawOracleCalls < 25000
  };
  return {
    epsilon,
    teacher: {
      id: teacher.id,
      dimension: teacher.dimension,
      trueChannelCount: teacher.atoms.length,
      trueNearParallelCosine: teacher.trueNearParallelCosine,
      expectedNearParallelPairs
    },
    summary: {
      pass: Object.values(gates).every(Boolean),
      effectiveChannelCount: model.effectiveChannelCount,
      rawChannelCount: model.rawChannelCount,
      selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
      selectedCandidateRank: model.adaptiveSelection?.selectedCandidateRank,
      duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
      nearParallelPairCount: model.adaptiveSelection?.nearParallelPairCount ?? 0,
      mae: metrics.mae,
      r2: metrics.r2,
      minDirectionProjection: Math.min(...projectionScores),
      maxRecoveredDirectionCosine: maxRecoveredCosine(recoveredWeights),
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates,
    baselines: {
      mean: meanBaseline
    },
    gcl: {
      metrics,
      projectionScores,
      adaptiveSelection: model.adaptiveSelection,
      queryBudget: model.queryBudget,
      atoms: model.atoms.map(atom => ({
        inputWeightNorm: atom.inputWeightNorm,
        inputBias: atom.inputBias,
        outputWeight: atom.outputWeight,
        strength: Math.abs(atom.outputWeight) * atom.inputWeightNorm
      }))
    }
  };
});

const pass = rows.every(row => row.summary.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'near-parallel mixed GELU FFN structure recovery epsilon family',
  blackBox: {
    type: '16D three-channel GELU FFN family with w2 = normalize(w1 + epsilon * orthogonal_delta)',
    visibilityToGcl: 'oracle calls only: f(vector); true channel count, directions, epsilon, biases, and readout weights are hidden'
  },
  claimScope: {
    provesIfPasses: 'GCL can keep mixed GELU atom count, direction recovery, and near-parallel diagnostics stable as two channels become nearly parallel in a controlled clean family',
    doesNotClaim: 'noisy mixed-block recovery, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, or broad SOTA status'
  },
  config: {
    epsilons,
    nearParallelDirectionCosine,
    maxChannelBudgetGivenToGcl: 4
  },
  gates: {
    pass,
    allEffectiveCountsMatch: rows.every(row => row.gates.effectiveChannelCountMatches),
    allDirectionRecoveryStable: rows.every(row => row.gates.directionRecoveryStable),
    allNearParallelDiagnosticsMatch: rows.every(row => row.gates.nearParallelDetectionMatches),
    allDuplicatePairsAbsent: rows.every(row => row.gates.duplicatePairsAbsent),
    allR2Above09998: rows.every(row => row.gates.heldoutR2Above09998),
    allMaeBelow0006: rows.every(row => row.gates.heldoutMaeBelow0006),
    allRawOracleCallsBelow25000: rows.every(row => row.gates.rawOracleCallsBelow25000)
  },
  summary: {
    pass,
    epsilonCount: rows.length,
    minR2: Math.min(...rows.map(row => row.summary.r2)),
    maxMae: Math.max(...rows.map(row => row.summary.mae)),
    minDirectionProjection: Math.min(...rows.map(row => row.summary.minDirectionProjection)),
    effectiveChannelCounts: rows.map(row => row.summary.effectiveChannelCount),
    duplicatePairCounts: rows.map(row => row.summary.duplicatePairCount),
    nearParallelPairCounts: rows.map(row => row.summary.nearParallelPairCount),
    maxRawOracleCalls: Math.max(...rows.map(row => row.summary.rawOracleCalls))
  },
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Near-Parallel Mixed GELU FFN Family Benchmark',
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
  `- Epsilons: ${epsilons.join(', ')}`,
  `- Effective channel counts: ${report.summary.effectiveChannelCounts.join(', ')}`,
  `- Duplicate pair counts: ${report.summary.duplicatePairCounts.join(', ')}`,
  `- Near-parallel pair counts: ${report.summary.nearParallelPairCounts.join(', ')}`,
  `- Max MAE: ${report.summary.maxMae}`,
  `- Min R2: ${report.summary.minR2}`,
  `- Min direction projection: ${report.summary.minDirectionProjection}`,
  `- Max raw oracle calls: ${report.summary.maxRawOracleCalls}`,
  '',
  '## Epsilon Rows',
  '',
  ...rows.map(row => (
    `- eps=${row.epsilon}: pass=${row.summary.pass}, effective=${row.summary.effectiveChannelCount}, nearPairs=${row.summary.nearParallelPairCount}, duplicates=${row.summary.duplicatePairCount}, MAE=${row.summary.mae}, R2=${row.summary.r2}, minProjection=${row.summary.minDirectionProjection}`
  )),
  '',
  '## Conclusion',
  '',
  pass
    ? 'The near-parallel family gate passed. GCL preserved explicit mixed GELU structure recovery while detecting near-parallel channel geometry.'
    : 'The near-parallel family gate failed. Do not promote the mixed GELU structure lane as near-parallel-stable until this report is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-near-parallel-family',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Stage C hard-case family for near-parallel mixed GELU channels.',
    'Measures output fit, direction recovery, effective channel count, duplicate diagnostics, near-parallel diagnostics, and query budget.',
    'This gate fails if high output fit is achieved while structure recovery collapses.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
