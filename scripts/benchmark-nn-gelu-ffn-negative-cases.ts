#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  canonicalizeGeluFfnStructure,
  decompileRankAdaptiveGeluFfnStructure,
  GeluFfnStructureAtom,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type NegativeCaseRow = {
  id: string;
  family: string;
  expectedCorrectBehavior: string;
  pass: boolean;
  summary: Record<string, unknown>;
  gates: Record<string, boolean>;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-negative-cases',
  'formalization-negative-cases'
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
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
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

function makeDirections(dimension: number, count: number, seed: number): number[][] {
  const random = makeRandom(seed);
  const directions: number[][] = [];
  for (let index = 0; index < count; index++) {
    directions.push(normalize(subtractProjection(
      Array.from({ length: dimension }, () => gaussian(random)),
      directions
    )));
  }
  return directions;
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

function makeAtom(direction: number[], inputWeightNorm: number, inputBias: number, outputWeight: number): GeluFfnStructureAtom {
  return {
    inputWeight: direction.map(value => value * inputWeightNorm),
    inputWeightNorm,
    inputBias,
    outputWeight
  };
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function atomStrengthSummary(model: { atoms: GeluFfnStructureAtom[] }): number[] {
  return model.atoms
    .map(atom => Math.abs(atom.outputWeight) * norm(atom.inputWeight))
    .sort((a, b) => b - a);
}

function runCanonicalManualNegativeCase(): NegativeCaseRow {
  const directions = [
    normalize([1, 0.2, -0.1, 0.05]),
    normalize([-0.3, 0.9, 0.1, -0.2]),
    normalize([0.2, -0.1, 0.97, 0.03])
  ];
  const atoms = [
    makeAtom(directions[0], 1.2, -0.25, 0.35),
    makeAtom(directions[0], 1.2, -0.25, 0.55),
    makeAtom(directions[1], 1.05, 0.1, 0.72),
    makeAtom(directions[1], 1.05, 0.1, -0.72),
    makeAtom(directions[2], 0.92, 0.35, 0.004)
  ];
  const signature = canonicalizeGeluFfnStructure({ atoms }, {
    duplicateDirectionCosine: 0.999,
    duplicateBiasTolerance: 0.04,
    duplicateNormTolerance: 0.04,
    atomRelativeStrengthThreshold: 0.02,
    cancellationOutputTolerance: 0.04
  });
  const gates = {
    duplicateSplitCollapsed: signature.mergedDuplicateCount >= 1,
    cancellationRemoved: signature.removedCancellationCount >= 2,
    weakAtomRemoved: signature.removedWeakAtomCount >= 1,
    onlyObservableAtomRemains: signature.canonicalAtomCount === 1
  };
  return {
    id: 'manual-duplicate-cancellation-weak-canonicalizer',
    family: 'exact duplicate split + equal/opposite cancellation + weak atom',
    expectedCorrectBehavior: 'collapse duplicate split, remove cancelling pair, prune weak atom, and keep only observable canonical support',
    pass: Object.values(gates).every(Boolean),
    summary: {
      canonicalAtomCount: signature.canonicalAtomCount,
      mergedDuplicateCount: signature.mergedDuplicateCount,
      removedCancellationCount: signature.removedCancellationCount,
      removedWeakAtomCount: signature.removedWeakAtomCount
    },
    gates
  };
}

function runWeakChannelBelowNoiseCase(): NegativeCaseRow {
  const dimension = 12;
  const directions = makeDirections(dimension, 3, 281001);
  const strongAtoms = [
    { direction: directions[0], inputScale: 1.15, inputBias: -0.24, outputWeight: 0.82 },
    { direction: directions[1], inputScale: 0.92, inputBias: 0.2, outputWeight: -0.61 }
  ];
  const weakAtom = { direction: directions[2], inputScale: 1.05, inputBias: -0.1, outputWeight: 0.0025 };
  const noiselessOracle = (input: number[]): number => (
    0.03
    + strongAtoms.reduce((sum, atom) => sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias), 0)
    + weakAtom.outputWeight * gelu(weakAtom.inputScale * dot(weakAtom.direction, input) + weakAtom.inputBias)
  );
  const noisyRandom = makeRandom(281101);
  const noisyOracle = (input: number[]): number => {
    const noisy = noiselessOracle(input) + gaussian(noisyRandom) * 0.01;
    return Math.round(noisy / 0.005) * 0.005;
  };
  const model = decompileRankAdaptiveGeluFfnStructure(noisyOracle, {
    dimension,
    ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: 4,
    selectionRestarts: 1,
    gradientProbeCount: 12,
    trainingSampleCount: 900,
    validationSampleCount: 280,
    randomSeed: 281201,
    restarts: 5,
    iterations: 360,
    learningRate: 0.03,
    querySamples: 7,
    aggregation: 'median',
    atomRelativeStrengthThreshold: 0.08,
    atomAbsoluteStrengthThreshold: 0.02,
    pruneValidationToleranceRatio: 0.6,
    pruneValidationToleranceAbsolute: 0.03
  });
  const inputs = makeInputs(281301, dimension, 700);
  const actual = inputs.map(noiselessOracle);
  const predicted = predictGeluFfnStructureBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const weakContribution = inputs.map(input => Math.abs(weakAtom.outputWeight * gelu(weakAtom.inputScale * dot(weakAtom.direction, input) + weakAtom.inputBias)));
  const maxWeakContribution = Math.max(...weakContribution);
  const signature = canonicalizeGeluFfnStructure(model, {
    atomRelativeStrengthThreshold: 0.04,
    atomAbsoluteStrengthThreshold: 0.01
  });
  const gates = {
    outputStillFaithful: metrics.r2 > 0.99 && metrics.mae < 0.05,
    weakSignalBelowNoiseFloor: maxWeakContribution < 0.015,
    hiddenWeakAtomNotPromotedAsSemanticRecovery: (model.effectiveChannelCount ?? model.atoms.length) < 3 && signature.canonicalAtomCount < 3,
    weakOrPrunedEvidenceRecorded: (model.adaptiveSelection?.prunedWeakAtomCount ?? 0) > 0 || signature.removedWeakAtomCount > 0,
    rawOracleCallsBelow80000: model.queryBudget.rawOracleCalls < 80000
  };
  return {
    id: 'weak-channel-below-noise',
    family: 'weak hidden channel under stochastic output noise and quantization',
    expectedCorrectBehavior: 'keep output fidelity but refuse to claim all original hidden channels are identifiable',
    pass: Object.values(gates).every(Boolean),
    summary: {
      effectiveChannelCount: model.effectiveChannelCount,
      canonicalAtomCount: signature.canonicalAtomCount,
      selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
      bestCandidateMetrics: model.adaptiveSelection?.bestCandidateMetrics,
      selectedValidationMetrics: model.validationMetrics,
      prunedWeakAtomCount: model.adaptiveSelection?.prunedWeakAtomCount ?? 0,
      signatureRemovedWeakAtomCount: signature.removedWeakAtomCount,
      mae: metrics.mae,
      r2: metrics.r2,
      maxWeakContribution,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates
  };
}

function runWeakChannelAboveNoiseCase(): NegativeCaseRow {
  const dimension = 12;
  const directions = makeDirections(dimension, 3, 281701);
  const strongAtoms = [
    { direction: directions[0], inputScale: 1.12, inputBias: -0.22, outputWeight: 0.78 },
    { direction: directions[1], inputScale: 0.94, inputBias: 0.24, outputWeight: -0.57 }
  ];
  const weakAtom = { direction: directions[2], inputScale: 1.08, inputBias: -0.08, outputWeight: 0.09 };
  const strongOnlyOracle = (input: number[]): number => (
    0.025
    + strongAtoms.reduce((sum, atom) => sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias), 0)
  );
  const noiselessOracle = (input: number[]): number => (
    strongOnlyOracle(input)
    + weakAtom.outputWeight * gelu(weakAtom.inputScale * dot(weakAtom.direction, input) + weakAtom.inputBias)
  );
  const noiseStd = 0.003;
  const quantization = 0.001;
  const noisyRandom = makeRandom(281801);
  const noisyOracle = (input: number[]): number => {
    const noisy = noiselessOracle(input) + gaussian(noisyRandom) * noiseStd;
    return Math.round(noisy / quantization) * quantization;
  };
  const model = decompileRankAdaptiveGeluFfnStructure(noisyOracle, {
    dimension,
    ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: 4,
    selectionRestarts: 2,
    gradientProbeCount: 16,
    trainingSampleCount: 1100,
    validationSampleCount: 340,
    randomSeed: 281901,
    restarts: 6,
    iterations: 420,
    learningRate: 0.028,
    querySamples: 7,
    aggregation: 'median',
    atomRelativeStrengthThreshold: 0.015,
    atomAbsoluteStrengthThreshold: 0.004,
    pruneValidationToleranceRatio: 0.2,
    pruneValidationToleranceAbsolute: 0.01
  });
  const inputs = makeInputs(282001, dimension, 800);
  const actual = inputs.map(noiselessOracle);
  const predicted = predictGeluFfnStructureBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const strongOnlyMetrics = regressionMetrics(actual, inputs.map(strongOnlyOracle));
  const weakContribution = inputs.map(input => Math.abs(weakAtom.outputWeight * gelu(weakAtom.inputScale * dot(weakAtom.direction, input) + weakAtom.inputBias)));
  const maxWeakContribution = Math.max(...weakContribution);
  const meanWeakContribution = weakContribution.reduce((sum, value) => sum + value, 0) / weakContribution.length;
  const signature = canonicalizeGeluFfnStructure(model, {
    atomRelativeStrengthThreshold: 0.012,
    atomAbsoluteStrengthThreshold: 0.003
  });
  const gates = {
    outputFaithful: metrics.r2 > 0.997 && metrics.mae < 0.02,
    weakSignalAboveNoiseFloor: maxWeakContribution > 3 * noiseStd + quantization && meanWeakContribution > noiseStd + quantization,
    strongOnlyUnderfitsHeldoutOracle: strongOnlyMetrics.mae > Math.max(0.015, metrics.mae * 2.5),
    exactlyResolvableThreeAtomsRecovered: (model.effectiveChannelCount ?? model.atoms.length) === 3 && signature.canonicalAtomCount === 3,
    rawOracleCallsBelow120000: model.queryBudget.rawOracleCalls < 120000
  };
  return {
    id: 'weak-channel-above-noise-must-recover',
    family: 'weak but resolvable hidden channel above stochastic output noise and quantization',
    expectedCorrectBehavior: 'recover the third channel when held-out oracle evidence makes it identifiable; do not always collapse to the smallest model',
    pass: Object.values(gates).every(Boolean),
    summary: {
      effectiveChannelCount: model.effectiveChannelCount,
      canonicalAtomCount: signature.canonicalAtomCount,
      selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
      selectedCandidateRank: model.adaptiveSelection?.selectedCandidateRank,
      bestCandidateMetrics: model.adaptiveSelection?.bestCandidateMetrics,
      selectedValidationMetrics: model.validationMetrics,
      prunedWeakAtomCount: model.adaptiveSelection?.prunedWeakAtomCount ?? 0,
      selectedAtomStrengths: atomStrengthSummary(model),
      mae: metrics.mae,
      r2: metrics.r2,
      strongOnlyMae: strongOnlyMetrics.mae,
      strongOnlyR2: strongOnlyMetrics.r2,
      maxWeakContribution,
      meanWeakContribution,
      noiseStd,
      quantization,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates
  };
}

function runActivationMismatchCase(): NegativeCaseRow {
  const dimension = 8;
  const direction = makeDirections(dimension, 1, 282001)[0];
  const oracle = (input: number[]): number => (
    0.1 + 0.85 * Math.sin(3.2 * dot(direction, input) - 0.35)
  );
  const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
    dimension,
    ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: 3,
    selectionRestarts: 1,
    gradientProbeCount: 10,
    trainingSampleCount: 760,
    validationSampleCount: 260,
    randomSeed: 282101,
    restarts: 5,
    iterations: 340,
    learningRate: 0.028,
    atomRelativeStrengthThreshold: 0.035,
    pruneValidationToleranceRatio: 0.3,
    pruneValidationToleranceAbsolute: 0.006
  });
  const inputs = makeInputs(282201, dimension, 800);
  const actual = inputs.map(oracle);
  const predicted = predictGeluFfnStructureBatch(model, inputs);
  const metrics = regressionMetrics(actual, predicted);
  const worstError = worstAbsoluteError(actual, predicted);
  const signature = canonicalizeGeluFfnStructure(model);
  const wouldPromoteStructure = metrics.r2 > 0.999 && metrics.mae < 0.01 && worstError < 0.08;
  const gates = {
    mismatchNotPromotedAsGeluStructure: !wouldPromoteStructure,
    residualFailureIsMeasurable: metrics.r2 < 0.999 || worstError >= 0.08,
    attemptedModelRecordedAtoms: signature.canonicalAtomCount >= 1,
    rawOracleCallsBelow45000: model.queryBudget.rawOracleCalls < 45000
  };
  return {
    id: 'mis-specified-activation-family',
    family: 'sinusoidal teacher passed to GELU structure finder',
    expectedCorrectBehavior: 'do not promote a GELU structure claim when held-out residuals show activation-family mismatch',
    pass: Object.values(gates).every(Boolean),
    summary: {
      effectiveChannelCount: model.effectiveChannelCount,
      canonicalAtomCount: signature.canonicalAtomCount,
      mae: metrics.mae,
      r2: metrics.r2,
      worstError,
      wouldPromoteStructure,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates
  };
}

function runOvercompleteBudgetCase(): NegativeCaseRow {
  const dimension = 10;
  const directions = makeDirections(dimension, 2, 283001);
  const atoms = [
    { direction: directions[0], inputScale: 1.1, inputBias: -0.18, outputWeight: 0.76 },
    { direction: directions[1], inputScale: 0.88, inputBias: 0.27, outputWeight: -0.58 }
  ];
  const oracle = (input: number[]): number => (
    -0.04 + atoms.reduce((sum, atom) => sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias), 0)
  );
  const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
    dimension,
    ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: 5,
    selectionRestarts: 1,
    gradientProbeCount: 12,
    trainingSampleCount: 850,
    validationSampleCount: 280,
    randomSeed: 283101,
    restarts: 5,
    iterations: 360,
    learningRate: 0.03,
    atomRelativeStrengthThreshold: 0.03,
    pruneValidationToleranceRatio: 0.35,
    pruneValidationToleranceAbsolute: 0.006
  });
  const inputs = makeInputs(283201, dimension, 700);
  const metrics = regressionMetrics(inputs.map(oracle), predictGeluFfnStructureBatch(model, inputs));
  const signature = canonicalizeGeluFfnStructure(model);
  const gates = {
    selectedMinimumEquivalentCount: model.effectiveChannelCount === atoms.length && signature.canonicalAtomCount === atoms.length,
    overcompleteBudgetNotUsedAsClaim: model.rawChannelCount === 5 && (model.effectiveChannelCount ?? 0) < model.rawChannelCount,
    heldoutR2Above0999: metrics.r2 > 0.999,
    heldoutMaeBelow002: metrics.mae < 0.02,
    rawOracleCallsBelow70000: model.queryBudget.rawOracleCalls < 70000
  };
  return {
    id: 'overcomplete-budget-minimum-equivalent',
    family: 'two-channel GELU teacher with five-channel solver budget',
    expectedCorrectBehavior: 'select the smallest held-out-valid structure instead of promoting the overcomplete budget',
    pass: Object.values(gates).every(Boolean),
    summary: {
      rawChannelCount: model.rawChannelCount,
      effectiveChannelCount: model.effectiveChannelCount,
      canonicalAtomCount: signature.canonicalAtomCount,
      selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
      mae: metrics.mae,
      r2: metrics.r2,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    },
    gates
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const rows = [
  runCanonicalManualNegativeCase(),
  runWeakChannelBelowNoiseCase(),
  runWeakChannelAboveNoiseCase(),
  runActivationMismatchCase(),
  runOvercompleteBudgetCase()
];
const pass = rows.every(row => row.pass);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'GELU FFN formalization negative-case benchmark',
  formalization: 'docs/neural-decompiler-formalization.md',
  blackBox: {
    visibilityToGcl: 'oracle calls only for solver cases; manual canonicalizer case uses recovered-atom objects to test canonical equivalence rules'
  },
  claimScope: {
    provesIfPasses: 'GCL has executable guardrails for duplicate/cancellation canonicalization, weak-under-noise non-identifiability, weak-but-resolvable channel recovery, activation-family mismatch rejection, and minimum equivalent structure under overcomplete budgets',
    doesNotClaim: 'semantic identity of original hidden neurons, noisy full mixed-block recovery, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM decompilation, or broad SOTA status'
  },
  gates: {
    pass,
    allNegativeCasesPassed: rows.every(row => row.pass),
    noUnsupportedStructurePromotion: rows
      .filter(row => row.id.includes('weak-channel') || row.id.includes('mis-specified'))
      .every(row => row.pass),
    resolvableWeakChannelRecoveryPassed: rows.find(row => row.id === 'weak-channel-above-noise-must-recover')?.pass === true,
    minimumEquivalentSelectionPassed: rows.find(row => row.id === 'overcomplete-budget-minimum-equivalent')?.pass === true,
    canonicalizerNegativeCasesPassed: rows.find(row => row.id === 'manual-duplicate-cancellation-weak-canonicalizer')?.pass === true
  },
  summary: {
    pass,
    caseCount: rows.length,
    passedCases: rows.filter(row => row.pass).length,
    failedCases: rows.filter(row => !row.pass).map(row => row.id)
  },
  rows,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# GELU FFN Formalization Negative-Case Benchmark',
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
  `- Cases: ${report.summary.caseCount}`,
  `- Passed cases: ${report.summary.passedCases}`,
  `- Failed cases: ${report.summary.failedCases.join(', ') || 'none'}`,
  '',
  '## Cases',
  '',
  '| Case | Family | Expected behavior | Pass |',
  '| --- | --- | --- | --- |',
  ...rows.map(row => `| ${row.id} | ${row.family} | ${row.expectedCorrectBehavior} | ${row.pass} |`),
  '',
  '## Gate Details',
  '',
  ...rows.flatMap(row => [
    `### ${row.id}`,
    '',
    `- Family: ${row.family}`,
    `- Expected correct behavior: ${row.expectedCorrectBehavior}`,
    `- Pass: ${row.pass}`,
    `- Summary: \`${JSON.stringify(row.summary)}\``,
    ...Object.entries(row.gates).map(([key, value]) => `- ${key}: ${value}`),
    ''
  ]),
  '## Conclusion',
  '',
  pass
    ? 'The negative-case gate passed. The formalization is now backed by executable checks that prevent unsupported structure promotion in these benchmarked failure modes.'
    : 'The negative-case gate failed. Do not promote the formalization as enforced until the failing case is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-negative-cases',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Executable negative-case gate for docs/neural-decompiler-formalization.md.',
    'Fails if weak/noisy or activation-family-mismatch cases are promoted as unsupported structure recovery.',
    'Fails if a weak but measurable held-out channel is collapsed away by minimum-model bias.',
    'Keeps semantic hidden-neuron identity and full LLM FFN extraction out of scope.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
