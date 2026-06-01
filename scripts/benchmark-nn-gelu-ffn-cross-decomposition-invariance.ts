#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  canonicalizeGeluFfnStructure,
  compareGeluFfnCanonicalSignatures,
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
  representation: 'minimal' | 'overcomplete-equivalent';
  dimension: number;
  outputBias: number;
  atoms: TeacherAtom[];
  hiddenAtomCount: number;
  hiddenRedundantAtomCount: number;
  predict: (input: number[]) => number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-cross-decomposition-invariance',
  'equivalent-reparameterizations'
);

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

function makeDirections(dimension: number): number[][] {
  const random = makeRandom(260701);
  const directions: number[][] = [];
  for (let index = 0; index < 4; index++) {
    directions.push(normalize(subtractProjection(
      Array.from({ length: dimension }, () => gaussian(random)),
      directions
    )));
  }
  return directions;
}

function evaluateAtoms(outputBias: number, atoms: TeacherAtom[], input: number[]): number {
  return outputBias + atoms.reduce((sum, atom) => (
    sum + atom.outputWeight * gelu(atom.inputScale * dot(atom.direction, input) + atom.inputBias)
  ), 0);
}

function makeTeachers(): { minimal: Teacher; overcomplete: Teacher; canonicalAtoms: TeacherAtom[] } {
  const dimension = 16;
  const directions = makeDirections(dimension);
  const outputBias = 0.045;
  const canonicalAtoms = [
    { direction: directions[0], inputScale: 1.18, inputBias: -0.32, outputWeight: 0.86 },
    { direction: directions[1], inputScale: 0.97, inputBias: 0.18, outputWeight: -0.64 },
    { direction: directions[2], inputScale: 0.78, inputBias: 0.43, outputWeight: 0.52 }
  ];
  const cancelAtom = { direction: directions[3], inputScale: 1.08, inputBias: -0.14, outputWeight: 0.41 };
  const overcompleteAtoms = [
    canonicalAtoms[2],
    { ...canonicalAtoms[0], outputWeight: 0.31 },
    cancelAtom,
    canonicalAtoms[1],
    { ...canonicalAtoms[0], outputWeight: canonicalAtoms[0].outputWeight - 0.31 },
    { ...cancelAtom, outputWeight: -cancelAtom.outputWeight }
  ];
  return {
    canonicalAtoms,
    minimal: {
      id: 'minimal-three-channel-gelu',
      representation: 'minimal',
      dimension,
      outputBias,
      atoms: canonicalAtoms,
      hiddenAtomCount: canonicalAtoms.length,
      hiddenRedundantAtomCount: 0,
      predict: input => evaluateAtoms(outputBias, canonicalAtoms, input)
    },
    overcomplete: {
      id: 'overcomplete-equivalent-three-channel-gelu',
      representation: 'overcomplete-equivalent',
      dimension,
      outputBias,
      atoms: overcompleteAtoms,
      hiddenAtomCount: overcompleteAtoms.length,
      hiddenRedundantAtomCount: overcompleteAtoms.length - canonicalAtoms.length,
      predict: input => evaluateAtoms(outputBias, overcompleteAtoms, input)
    }
  };
}

function bestDirectionProjection(trueDirection: number[], recoveredWeights: number[][]): number {
  return Math.max(...recoveredWeights.map(weight => Math.abs(dot(trueDirection, normalize(weight)))));
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teachers = makeTeachers();
const validationInputs = makeInputs(260781, teachers.minimal.dimension, 900);
const oracleDelta = Math.max(...validationInputs.map(input => Math.abs(
  teachers.minimal.predict(input) - teachers.overcomplete.predict(input)
)));

const runs = [teachers.minimal, teachers.overcomplete].map((teacher, index) => {
  const model = decompileRankAdaptiveGeluFfnStructure(teacher.predict, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-1.5, 1.5] as [number, number]),
    maxChannelCount: 5,
    selectionRestarts: 2,
    gradientProbeCount: 18,
    trainingSampleCount: 1500,
    validationSampleCount: 450,
    randomSeed: 260810 + index * 997,
    restarts: 10,
    iterations: 700,
    learningRate: 0.03,
    duplicateDirectionCosine: 0.999,
    duplicateBiasTolerance: 0.08,
    atomRelativeStrengthThreshold: 0.03,
    pruneValidationToleranceRatio: 0.28,
    pruneValidationToleranceAbsolute: 0.004
  });
  const actual = validationInputs.map(teacher.predict);
  const predicted = predictGeluFfnStructureBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const recoveredWeights = model.atoms.map(atom => atom.inputWeight);
  const projectionScores = teachers.canonicalAtoms.map(atom => bestDirectionProjection(atom.direction, recoveredWeights));
  const signature = canonicalizeGeluFfnStructure(model, {
    duplicateDirectionCosine: 0.999,
    duplicateBiasTolerance: 0.08,
    duplicateNormTolerance: 0.08,
    atomRelativeStrengthThreshold: 0.02
  });
  return {
    teacher: {
      id: teacher.id,
      representation: teacher.representation,
      hiddenAtomCount: teacher.hiddenAtomCount,
      hiddenRedundantAtomCount: teacher.hiddenRedundantAtomCount
    },
    model,
    signature,
    metrics,
    projectionScores,
    summary: {
      effectiveChannelCount: model.effectiveChannelCount,
      rawChannelCount: model.rawChannelCount,
      selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
      duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
      nearParallelPairCount: model.adaptiveSelection?.nearParallelPairCount ?? 0,
      canonicalAtomCount: signature.canonicalAtomCount,
      mergedDuplicateCount: signature.mergedDuplicateCount,
      removedCancellationCount: signature.removedCancellationCount,
      minDirectionProjection: Math.min(...projectionScores),
      mae: metrics.mae,
      r2: metrics.r2,
      rawOracleCalls: model.queryBudget.rawOracleCalls
    }
  };
});

const signatureComparison = compareGeluFfnCanonicalSignatures(runs[0].signature, runs[1].signature);
const minimalPredicted = predictGeluFfnStructureBatch(runs[0].model, validationInputs);
const overcompletePredicted = predictGeluFfnStructureBatch(runs[1].model, validationInputs);
const modelAgreementMetrics = regressionMetrics(minimalPredicted, overcompletePredicted);
const gates = {
  oracleFunctionsEquivalent: oracleDelta < 1e-12,
  bothEffectiveCountsMatch: runs.every(run => run.summary.effectiveChannelCount === teachers.canonicalAtoms.length),
  bothCanonicalCountsMatch: runs.every(run => run.summary.canonicalAtomCount === teachers.canonicalAtoms.length),
  bothHeldoutR2Above09998: runs.every(run => run.metrics.r2 > 0.9998),
  bothHeldoutMaeBelow0008: runs.every(run => run.metrics.mae < 0.008),
  bothDirectionRecoveryStable: runs.every(run => run.summary.minDirectionProjection > 0.998),
  canonicalSignatureInvariant: signatureComparison.signatureDistance < 0.08 && signatureComparison.maxAtomDistance < 0.12,
  modelPredictionsAgree: modelAgreementMetrics.mae < 0.006 && modelAgreementMetrics.r2 > 0.9998,
  rawOracleCallsBelow50000Each: runs.every(run => run.summary.rawOracleCalls < 50000)
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'GELU FFN cross-decomposition invariance benchmark',
  blackBox: {
    type: '16D GELU FFN pair with two mathematically equivalent hidden decompositions',
    visibilityToGcl: 'oracle calls only: GCL sees f(vector), not hidden atom lists, duplicate splits, cancellation pairs, directions, biases, or readout weights'
  },
  claimScope: {
    provesIfPasses: 'GCL returns the same canonical recovered GELU structure for two distinct internal decompositions of the same clean function, under different solver seeds',
    doesNotClaim: 'semantic identity of original hidden neurons, noisy invariance, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D LLM block decompilation, or broad SOTA status'
  },
  teacherAudit: {
    dimension: teachers.minimal.dimension,
    canonicalHiddenAtomCount: teachers.canonicalAtoms.length,
    minimalHiddenAtomCount: teachers.minimal.hiddenAtomCount,
    overcompleteHiddenAtomCount: teachers.overcomplete.hiddenAtomCount,
    overcompleteHiddenRedundantAtomCount: teachers.overcomplete.hiddenRedundantAtomCount,
    oracleDeltaOnValidation: oracleDelta,
    reparameterization: 'Teacher B permutes atoms, splits one GELU atom into two identical-support readout pieces, and adds an equal/opposite GELU cancellation pair.'
  },
  runs: runs.map(run => ({
    teacher: run.teacher,
    summary: run.summary,
    metrics: run.metrics,
    projectionScores: run.projectionScores,
    adaptiveSelection: run.model.adaptiveSelection,
    canonicalSignature: {
      canonicalAtomCount: run.signature.canonicalAtomCount,
      removedWeakAtomCount: run.signature.removedWeakAtomCount,
      mergedDuplicateCount: run.signature.mergedDuplicateCount,
      removedCancellationCount: run.signature.removedCancellationCount,
      atoms: run.signature.atoms.map(atom => ({
        inputWeightNorm: atom.inputWeightNorm,
        inputBias: atom.inputBias,
        outputWeight: atom.outputWeight,
        strength: atom.strength
      }))
    },
    queryBudget: run.model.queryBudget
  })),
  signatureComparison,
  modelAgreementMetrics,
  gates: { ...gates, pass },
  summary: {
    pass,
    oracleDelta,
    effectiveChannelCounts: runs.map(run => run.summary.effectiveChannelCount),
    canonicalAtomCounts: runs.map(run => run.summary.canonicalAtomCount),
    signatureDistance: signatureComparison.signatureDistance,
    maxAtomDistance: signatureComparison.maxAtomDistance,
    maxMae: Math.max(...runs.map(run => run.metrics.mae)),
    minR2: Math.min(...runs.map(run => run.metrics.r2)),
    minDirectionProjection: Math.min(...runs.map(run => run.summary.minDirectionProjection)),
    modelAgreementMae: modelAgreementMetrics.mae,
    maxRawOracleCalls: Math.max(...runs.map(run => run.summary.rawOracleCalls))
  },
  runtimeMs: Date.now() - started
};

const markdown = [
  '# GELU FFN Cross-Decomposition Invariance Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Teacher Audit',
  '',
  `- Dimension: ${report.teacherAudit.dimension}`,
  `- Canonical hidden atom count: ${report.teacherAudit.canonicalHiddenAtomCount}`,
  `- Minimal hidden atom count: ${report.teacherAudit.minimalHiddenAtomCount}`,
  `- Overcomplete hidden atom count: ${report.teacherAudit.overcompleteHiddenAtomCount}`,
  `- Overcomplete hidden redundant atom count: ${report.teacherAudit.overcompleteHiddenRedundantAtomCount}`,
  `- Oracle delta on validation: ${report.teacherAudit.oracleDeltaOnValidation}`,
  '',
  '## Summary',
  '',
  `- Pass: ${pass}`,
  `- Effective channel counts: ${report.summary.effectiveChannelCounts.join(', ')}`,
  `- Canonical atom counts: ${report.summary.canonicalAtomCounts.join(', ')}`,
  `- Signature distance: ${report.summary.signatureDistance}`,
  `- Max atom distance: ${report.summary.maxAtomDistance}`,
  `- Max MAE: ${report.summary.maxMae}`,
  `- Min R2: ${report.summary.minR2}`,
  `- Min direction projection: ${report.summary.minDirectionProjection}`,
  `- Model agreement MAE: ${report.summary.modelAgreementMae}`,
  `- Max raw oracle calls: ${report.summary.maxRawOracleCalls}`,
  '',
  '## Per Representation',
  '',
  '| Representation | Hidden atoms | Effective | Canonical | MAE | R2 | Min direction projection | Oracle calls |',
  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ...runs.map(run => `| ${run.teacher.representation} | ${run.teacher.hiddenAtomCount} | ${run.summary.effectiveChannelCount} | ${run.summary.canonicalAtomCount} | ${run.summary.mae} | ${run.summary.r2} | ${run.summary.minDirectionProjection} | ${run.summary.rawOracleCalls} |`),
  '',
  '## Gate',
  '',
  ...Object.entries(gates).map(([key, value]) => `- ${key}: ${value}`),
  '',
  '## Conclusion',
  '',
  pass
    ? 'The cross-decomposition invariance gate passed. GCL recovered the same canonical GELU structure from two equivalent hidden decompositions of the same oracle function.'
    : 'The cross-decomposition invariance gate failed. Do not promote the GELU structure lane as canonical-structure invariant until this report is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-cross-decomposition-invariance',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'Stage C identifiability gate for canonical GELU structure recovery.',
    'Uses two mathematically equivalent hidden decompositions and requires the same recovered canonical signature.',
    'Keeps full GPT-2 FFN extraction and noisy invariance out of scope.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!pass) process.exit(1);
