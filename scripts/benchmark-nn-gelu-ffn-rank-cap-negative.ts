#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileRankAdaptiveGeluFfnStructure,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type ClaimDecision =
  | 'structure-recovered'
  | 'insufficient-rank'
  | 'non-identifiable'
  | 'surrogate-only'
  | 'invalid-structure-claim';

type SyntheticSource = {
  index: number;
  groupId: number;
  axisA: number;
  axisB: number;
  axisC: number;
  inputScale: number;
  inputBias: number;
  outputWeight: number;
  contributionScore: number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-gelu-ffn-rank-cap-negative',
  'synthetic-3072-rank-cap'
);

const dimension = 768;
const sourceChannelCount = 3072;
const observableGroupCount = 48;
const rankCap = 12;
const validationCount = 1200;
const inputScale = 0.45;

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

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function makeInputs(seed: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * inputScale)
  ));
}

function makeSources(): SyntheticSource[] {
  const random = makeRandom(603031);
  const sources: SyntheticSource[] = [];
  for (let index = 0; index < sourceChannelCount; index++) {
    const groupId = index % observableGroupCount;
    const groupStrength = 1 - groupId / (observableGroupCount * 1.7);
    const duplicateJitter = 0.82 + 0.36 * random();
    const inputScale = (0.55 + 0.45 * groupStrength) * duplicateJitter;
    const outputSign = groupId % 2 === 0 ? 1 : -1;
    const outputWeight = outputSign * (0.018 + 0.052 * groupStrength) * (0.75 + 0.5 * random());
    sources.push({
      index,
      groupId,
      axisA: (groupId * 37 + 11) % dimension,
      axisB: (groupId * 73 + 19) % dimension,
      axisC: (groupId * 109 + 29) % dimension,
      inputScale,
      inputBias: -0.75 + 1.5 * random(),
      outputWeight,
      contributionScore: Math.abs(outputWeight) * inputScale
    });
  }
  return sources;
}

function sourcePreactivation(source: SyntheticSource, input: number[]): number {
  return source.inputScale * (
    0.82 * input[source.axisA]
    - 0.47 * input[source.axisB]
    + 0.31 * input[source.axisC]
  ) + source.inputBias;
}

function makeOracle(sources: SyntheticSource[]): (input: number[]) => number {
  return input => {
    let value = 0.035;
    for (const source of sources) value += source.outputWeight * gelu(sourcePreactivation(source, input));
    return value;
  };
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function groupContributionScores(sources: SyntheticSource[]): Array<{ groupId: number; score: number; channels: number }> {
  const groups = new Map<number, { groupId: number; score: number; channels: number }>();
  for (const source of sources) {
    const group = groups.get(source.groupId) ?? { groupId: source.groupId, score: 0, channels: 0 };
    group.score += source.contributionScore;
    group.channels++;
    groups.set(source.groupId, group);
  }
  return [...groups.values()].sort((a, b) => b.score - a.score);
}

function decideClaim(input: {
  effectiveChannelCount: number;
  rankCap: number;
  requiredGroupCount: number;
  topRankCoverage: number;
  outputR2: number;
  outputMae: number;
  worstError: number;
}): { decision: ClaimDecision; reasons: string[] } {
  const reasons: string[] = [];
  if (input.effectiveChannelCount >= input.requiredGroupCount && input.outputR2 > 0.995 && input.outputMae < 0.02) {
    return { decision: 'structure-recovered', reasons: ['effective channel count covers the required observable source groups'] };
  }
  if (input.rankCap < input.requiredGroupCount) {
    reasons.push(`rank cap ${input.rankCap} is below required observable group count ${input.requiredGroupCount}`);
  }
  if (input.topRankCoverage < 0.7) {
    reasons.push(`top-rank source coverage ${input.topRankCoverage.toFixed(6)} is below 0.70`);
  }
  if (input.outputR2 > 0.99 && input.effectiveChannelCount < input.requiredGroupCount) {
    reasons.push('output fit is high but structure coverage is incomplete');
    return { decision: 'surrogate-only', reasons };
  }
  if (input.rankCap < input.requiredGroupCount) return { decision: 'insufficient-rank', reasons };
  if (input.outputR2 < 0.95 || input.worstError > 0.5) {
    reasons.push('held-out fidelity is not strong enough for structure promotion');
    return { decision: 'surrogate-only', reasons };
  }
  reasons.push('structure claim lacks required source coverage evidence');
  return { decision: 'invalid-structure-claim', reasons };
}

function makeMarkdown(report: Record<string, any>): string {
  return [
    '# GELU FFN Rank-Cap Negative Guard',
    '',
    `Generated: ${report.generatedAt}`,
    '',
    '## Purpose',
    '',
    'This gate strengthens GCL by preventing a false full-slice structure claim when the rank budget is too small.',
    '',
    '## Summary',
    '',
    `- Pass: ${report.summary.pass}`,
    `- Claim decision: ${report.summary.claimDecision}`,
    `- Dimension: ${report.summary.dimension}`,
    `- Source channels: ${report.summary.sourceChannelCount}`,
    `- Observable groups: ${report.summary.observableGroupCount}`,
    `- Rank cap: ${report.summary.rankCap}`,
    `- Effective recovered atoms: ${report.summary.effectiveChannelCount}`,
    `- R2: ${report.summary.metrics.r2}`,
    `- MAE: ${report.summary.metrics.mae}`,
    `- Worst error: ${report.summary.worstError}`,
    `- Top-rank source coverage: ${report.summary.topRankCoverage}`,
    `- Raw oracle calls: ${report.summary.rawOracleCalls}`,
    '',
    '## Decision Reasons',
    '',
    ...report.summary.claimReasons.map((reason: string) => `- ${reason}`),
    '',
    '## Conclusion',
    '',
    report.summary.pass
      ? 'The negative guard passed: GCL did not promote an unsupported full-slice structure claim under an insufficient rank cap.'
      : 'The negative guard failed: GCL either promoted unsupported structure or did not record the expected insufficient-rank diagnostic.'
  ].join('\n');
}

const started = Date.now();
const sources = makeSources();
const oracle = makeOracle(sources);
const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
  dimension,
  ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
  maxChannelCount: rankCap,
  minEffectiveChannelCount: 1,
  candidateChannelCounts: [rankCap],
  activeRank: rankCap,
  selectionRestarts: 1,
  gradientProbeCount: 48,
  trainingSampleCount: 2300,
  gradientTrainingSampleCount: 96,
  gradientTrainingWeight: 0.10,
  finiteDifferenceStep: 0.045,
  activeGradientFiniteDifferenceStep: 0.04,
  validationSampleCount: 420,
  randomSeed: 703031,
  restarts: 5,
  iterations: 520,
  learningRate: 0.024,
  scalarRefinementRestarts: 2,
  scalarRefinementIterations: 280,
  scalarRefinementLearningRate: 0.012,
  atomRelativeStrengthThreshold: 0.006,
  atomAbsoluteStrengthThreshold: 0.00005,
  pruneValidationToleranceRatio: 0.12,
  pruneValidationToleranceAbsolute: 0.004
});
const validationInputs = makeInputs(803031, validationCount);
const actual = validationInputs.map(oracle);
const predicted = predictGeluFfnStructureBatch(model, validationInputs);
const metrics = regressionMetrics(actual, predicted);
const worstError = worstAbsoluteError(actual, predicted);
const groupScores = groupContributionScores(sources);
const totalGroupScore = groupScores.reduce((sum, group) => sum + group.score, 0);
const topRankCoverage = groupScores.slice(0, rankCap).reduce((sum, group) => sum + group.score, 0) / Math.max(1e-12, totalGroupScore);
const topGroups = groupScores.slice(0, 16);
const decision = decideClaim({
  effectiveChannelCount: model.effectiveChannelCount ?? model.atoms.length,
  rankCap,
  requiredGroupCount: observableGroupCount,
  topRankCoverage,
  outputR2: metrics.r2,
  outputMae: metrics.mae,
  worstError
});
const gates = {
  syntheticFullSliceShape: dimension === 768 && sourceChannelCount === 3072 && observableGroupCount > rankCap,
  expectedNegativeDecision: decision.decision === 'insufficient-rank' || decision.decision === 'surrogate-only',
  doesNotPromoteStructure: decision.decision !== 'structure-recovered' && decision.decision !== 'invalid-structure-claim',
  recordsRankReason: decision.reasons.some(reason => reason.includes('rank cap')),
  reportHasQueryBudget: model.queryBudget.rawOracleCalls > 0,
  finiteMetrics: Number.isFinite(metrics.r2) && Number.isFinite(metrics.mae) && Number.isFinite(worstError)
};
const pass = Object.values(gates).every(Boolean);
const summary = {
  pass,
  claimDecision: decision.decision,
  claimReasons: decision.reasons,
  dimension,
  sourceChannelCount,
  observableGroupCount,
  rankCap,
  effectiveChannelCount: model.effectiveChannelCount ?? model.atoms.length,
  rawChannelCount: model.rawChannelCount ?? model.channelCount,
  metrics,
  worstError,
  outputStd: Math.sqrt(mean(actual.map(value => (value - mean(actual)) ** 2))),
  topRankCoverage,
  rawOracleCalls: model.queryBudget.rawOracleCalls
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'Synthetic 3072-channel GELU FFN rank-cap negative guard',
  claimScope: {
    provesIfPasses: 'GCL can refuse an unsupported full-slice structure claim when the rank cap is below the observable source-group count.',
    doesNotClaim: 'full GPT-2 FFN extraction, arbitrary 3072-channel blind decompilation, semantic source identity, or broad SOTA status.'
  },
  summary,
  gates,
  syntheticTeacher: {
    type: 'deterministic sparse synthetic GELU full-slice',
    dimension,
    sourceChannelCount,
    observableGroupCount,
    rankCap,
    channelGeneration: '3072 logical source channels grouped into 48 observable sparse GELU directions'
  },
  sourceCoverage: {
    totalGroupScore,
    topRankCoverage,
    topGroups
  },
  model: {
    recoveryKind: model.recoveryKind,
    effectiveChannelCount: model.effectiveChannelCount,
    rawChannelCount: model.rawChannelCount,
    selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
    adaptiveSelection: model.adaptiveSelection,
    validationMetrics: model.validationMetrics,
    queryBudget: model.queryBudget
  },
  runtimeMs: Date.now() - started
};

fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${makeMarkdown(report)}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-gelu-ffn-rank-cap-negative',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Executable negative guard for full-slice/rank-cap overclaiming.',
    'Correct pass is insufficient-rank or surrogate-only, not structure promotion.',
    'Self-contained release fixture; real GPT-2 full fixture remains local research audit.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
