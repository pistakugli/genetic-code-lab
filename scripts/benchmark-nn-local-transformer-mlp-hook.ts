#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileRankAdaptiveGeluFfnStructure,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import {
  GeluFfnSourceNeuron,
  modelAgreementMae,
  predictGeluFfnSourceNeurons,
  verifyGeluFfnNeuronProvenance
} from '../src/solvers/gelu-ffn-provenance-verifier';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type LocalTransformerHookRow = {
  promptId: number;
  tokenPosition: number;
  tokenId: number;
  residualInput: number[];
  mlpInput: number[];
  fcPreactivations: number[];
  fcActivations: number[];
  targetOutput: number;
  targetContributions: number[];
};

type LocalTransformerMlpBlock = {
  dimension: number;
  expansionCount: number;
  targetOutputIndex: number;
  tokenEmbedding: number[][];
  positionEmbedding: number[][];
  attentionMatrix: number[][];
  layerNormGamma: number[];
  layerNormBeta: number[];
  fcWeights: number[][];
  fcBias: number[];
  projWeights: number[][];
  projBias: number[];
  sourceHiddenIndices: number[];
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-local-transformer-mlp-hook',
  'hooked-gelu-slice'
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

function dot(left: number[], right: number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index++) sum += left[index] * right[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(1e-12, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = norm(values);
  return values.map(value => value / length);
}

function subtractProjection(vector: number[], basis: number[][]): number[] {
  let outputVector = [...vector];
  for (const direction of basis) {
    const projection = dot(outputVector, direction);
    outputVector = outputVector.map((value, index) => value - projection * direction[index]);
  }
  return outputVector;
}

function makeDirection(random: () => number, dimension: number, previous: number[][] = []): number[] {
  return normalize(subtractProjection(
    Array.from({ length: dimension }, () => gaussian(random)),
    previous
  ));
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function layerNorm(values: number[], gamma: number[], beta: number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const scale = 1 / Math.sqrt(variance + 1e-5);
  return values.map((value, index) => (value - mean) * scale * gamma[index] + beta[index]);
}

function matVec(matrix: number[][], vector: number[]): number[] {
  return matrix.map(row => dot(row, vector));
}

function addScaled(left: number[], right: number[], scale: number): number[] {
  return left.map((value, index) => value + scale * right[index]);
}

function tokenize(text: string, vocabSize: number): number[] {
  return Array.from(text).map(char => char.charCodeAt(0) % vocabSize);
}

function makeLocalTransformerMlpBlock(): LocalTransformerMlpBlock {
  const random = makeRandom(73001);
  const dimension = 20;
  const expansionCount = 10;
  const vocabSize = 64;
  const maxPosition = 48;
  const targetOutputIndex = 7;
  const tokenEmbedding = Array.from({ length: vocabSize }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * 0.18)
  ));
  const positionEmbedding = Array.from({ length: maxPosition }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * 0.08)
  ));
  const attentionMatrix = Array.from({ length: dimension }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * 0.11 / Math.sqrt(dimension))
  ));
  const layerNormGamma = Array.from({ length: dimension }, () => 0.9 + 0.25 * random());
  const layerNormBeta = Array.from({ length: dimension }, () => gaussian(random) * 0.03);

  const sourceHiddenIndices = [1, 4, 8];
  const sourceDirections: number[][] = [];
  for (let index = 0; index < sourceHiddenIndices.length; index++) {
    sourceDirections.push(makeDirection(random, dimension, sourceDirections));
  }
  const fcWeights = Array.from({ length: expansionCount }, () => makeDirection(random, dimension).map(value => value * (0.5 + random())));
  const fcBias = Array.from({ length: expansionCount }, () => gaussian(random) * 0.18);
  const sourceScales = [1.18, 0.96, 0.82];
  const sourceBiases = [-0.24, 0.18, -0.06];
  for (const [localIndex, hiddenIndex] of sourceHiddenIndices.entries()) {
    fcWeights[hiddenIndex] = sourceDirections[localIndex].map(value => value * sourceScales[localIndex]);
    fcBias[hiddenIndex] = sourceBiases[localIndex];
  }

  const projWeights = Array.from({ length: expansionCount }, () => (
    Array.from({ length: dimension }, () => gaussian(random) * 0.025)
  ));
  for (let hidden = 0; hidden < expansionCount; hidden++) projWeights[hidden][targetOutputIndex] = 0;
  const sourceOutputWeights = [0.72, -0.58, 0.41];
  for (const [localIndex, hiddenIndex] of sourceHiddenIndices.entries()) {
    projWeights[hiddenIndex][targetOutputIndex] = sourceOutputWeights[localIndex];
  }
  const projBias = Array.from({ length: dimension }, () => gaussian(random) * 0.025);
  projBias[targetOutputIndex] = 0.035;

  return {
    dimension,
    expansionCount,
    targetOutputIndex,
    tokenEmbedding,
    positionEmbedding,
    attentionMatrix,
    layerNormGamma,
    layerNormBeta,
    fcWeights,
    fcBias,
    projWeights,
    projBias,
    sourceHiddenIndices
  };
}

function runPrompt(block: LocalTransformerMlpBlock, prompt: string, promptId: number): LocalTransformerHookRow[] {
  const tokenIds = tokenize(prompt, block.tokenEmbedding.length).slice(0, block.positionEmbedding.length);
  const baseResiduals = tokenIds.map((tokenId, position) => (
    block.tokenEmbedding[tokenId].map((value, axis) => value + block.positionEmbedding[position][axis])
  ));
  const rows: LocalTransformerHookRow[] = [];
  for (let position = 0; position < baseResiduals.length; position++) {
    const context = new Array(block.dimension).fill(0);
    for (let previous = 0; previous <= position; previous++) {
      const weight = (previous + 1) / (position + 1);
      for (let axis = 0; axis < block.dimension; axis++) context[axis] += baseResiduals[previous][axis] * weight;
    }
    const contextScale = (position + 1) / ((position + 1) * (position + 2) / 2);
    const attentionOutput = matVec(block.attentionMatrix, context.map(value => value * contextScale));
    const residualInput = addScaled(baseResiduals[position], attentionOutput, 0.45);
    const mlpInput = layerNorm(residualInput, block.layerNormGamma, block.layerNormBeta);
    const fcPreactivations = block.fcWeights.map((weight, hidden) => dot(weight, mlpInput) + block.fcBias[hidden]);
    const fcActivations = fcPreactivations.map(gelu);
    const targetContributions = block.projWeights.map((weight, hidden) => weight[block.targetOutputIndex] * fcActivations[hidden]);
    const targetOutput = block.projBias[block.targetOutputIndex]
      + targetContributions.reduce((sum, value) => sum + value, 0);
    rows.push({
      promptId,
      tokenPosition: position,
      tokenId: tokenIds[position],
      residualInput,
      mlpInput,
      fcPreactivations,
      fcActivations,
      targetOutput,
      targetContributions
    });
  }
  return rows;
}

function makeHookRows(block: LocalTransformerMlpBlock): LocalTransformerHookRow[] {
  const prompts = [
    'GCL maps transformer MLP structure from hooked activations.',
    'Canonical atoms need provenance, not only output fit.',
    'The local block exposes residual stream rows and GELU preactivations.',
    'Structure recovery should identify source channels and why they react.',
    'Shared gates, weak atoms, and duplicate splits require explicit gates.',
    'Release evidence must record query budget and held-out activation rows.',
    'A transformer hook is only trusted when shapes and outputs match.',
    'Neuron decompilation starts with local source mapping before GPT stress.'
  ];
  return prompts.flatMap((prompt, promptId) => runPrompt(block, prompt, promptId));
}

function makeSources(block: LocalTransformerMlpBlock): GeluFfnSourceNeuron[] {
  return block.sourceHiddenIndices.map(hiddenIndex => ({
    id: `local.transformer.h0.mlp.c_fc[${hiddenIndex}]`,
    layer: 'local.transformer.h0.mlp.c_fc',
    index: hiddenIndex,
    origin: `local transformer MLP expansion neuron ${hiddenIndex} feeding output slice ${block.targetOutputIndex}`,
    inputWeight: [...block.fcWeights[hiddenIndex]],
    inputBias: block.fcBias[hiddenIndex],
    outputWeight: block.projWeights[hiddenIndex][block.targetOutputIndex],
    activation: 'gelu'
  }));
}

function predictTargetFromMlpInput(block: LocalTransformerMlpBlock, input: number[]): number {
  let value = block.projBias[block.targetOutputIndex];
  for (let hidden = 0; hidden < block.expansionCount; hidden++) {
    value += block.projWeights[hidden][block.targetOutputIndex]
      * gelu(dot(block.fcWeights[hidden], input) + block.fcBias[hidden]);
  }
  return value;
}

function worstAbsoluteError(actual: number[], predicted: number[]): number {
  return Math.max(0, ...actual.map((value, index) => Math.abs(value - predicted[index])));
}

function validateHookRows(block: LocalTransformerMlpBlock, rows: LocalTransformerHookRow[]): {
  maxPreactivationError: number;
  maxActivationError: number;
  maxTargetOutputError: number;
} {
  let maxPreactivationError = 0;
  let maxActivationError = 0;
  let maxTargetOutputError = 0;
  for (const row of rows) {
    if (row.mlpInput.length !== block.dimension || row.residualInput.length !== block.dimension) {
      throw new Error('Local transformer hook row dimension mismatch');
    }
    if (row.fcPreactivations.length !== block.expansionCount || row.fcActivations.length !== block.expansionCount) {
      throw new Error('Local transformer hook expansion dimension mismatch');
    }
    for (let hidden = 0; hidden < block.expansionCount; hidden++) {
      const expectedPreactivation = dot(block.fcWeights[hidden], row.mlpInput) + block.fcBias[hidden];
      const expectedActivation = gelu(expectedPreactivation);
      maxPreactivationError = Math.max(maxPreactivationError, Math.abs(expectedPreactivation - row.fcPreactivations[hidden]));
      maxActivationError = Math.max(maxActivationError, Math.abs(expectedActivation - row.fcActivations[hidden]));
    }
    const expectedTarget = predictTargetFromMlpInput(block, row.mlpInput);
    maxTargetOutputError = Math.max(maxTargetOutputError, Math.abs(expectedTarget - row.targetOutput));
  }
  return { maxPreactivationError, maxActivationError, maxTargetOutputError };
}

function makeMarkdown(report: Record<string, any>): string {
  const rows = report.rows as Array<Record<string, any>>;
  return [
    '# Local Transformer MLP Hook Benchmark',
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
    `- Pass: ${report.summary.pass}`,
    `- Hook rows: ${report.summary.hookRowCount}`,
    `- Effective channels: ${report.summary.effectiveChannelCount}`,
    `- Max hook MAE: ${report.summary.hookMetrics.mae}`,
    `- Hook R2: ${report.summary.hookMetrics.r2}`,
    `- Worst hook error: ${report.summary.hookWorstError}`,
    `- Provenance confidence min: ${report.summary.minProvenanceConfidence}`,
    `- Raw oracle calls: ${report.summary.rawOracleCalls}`,
    '',
    '## Hook Integrity',
    '',
    `- Max preactivation error: ${report.summary.hookIntegrity.maxPreactivationError}`,
    `- Max activation error: ${report.summary.hookIntegrity.maxActivationError}`,
    `- Max target output error: ${report.summary.hookIntegrity.maxTargetOutputError}`,
    '',
    '## Provenance Matches',
    '',
    ...rows.flatMap(row => [
      `- recovered atom ${row.recoveredAtomIndex} -> ${row.sourceGroupId}`,
      `  - source ids: ${row.sourceIds.join(', ')}`,
      `  - layer: ${row.sourceLayer}`,
      `  - indices: ${row.sourceIndices.join(', ')}`,
      `  - confidence: ${row.confidence}`,
      `  - direction cosine: ${row.directionCosine}`,
      `  - contribution correlation: ${row.contributionCorrelation}`,
      `  - why: ${row.whyReacts.responseRule}`
    ]),
    '',
    '## Conclusion',
    '',
    report.summary.pass
      ? 'The local transformer hook gate passed. GCL recovered explicit GELU source structure from a hooked local transformer MLP output slice and mapped recovered atoms back to source expansion neurons with why-reacts evidence.'
      : 'The local transformer hook gate failed. Do not claim hooked transformer source mapping until the failing condition is repaired.'
  ].join('\n');
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const block = makeLocalTransformerMlpBlock();
const hookRows = makeHookRows(block);
const hookIntegrity = validateHookRows(block, hookRows);
const sources = makeSources(block);
const oracle = (input: number[]): number => predictTargetFromMlpInput(block, input);
const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
  dimension: block.dimension,
  ranges: Array.from({ length: block.dimension }, () => [-2.4, 2.4] as [number, number]),
  maxChannelCount: 4,
  minEffectiveChannelCount: 1,
  selectionRestarts: 2,
  gradientProbeCount: 18,
  trainingSampleCount: 1250,
  validationSampleCount: 420,
  randomSeed: 73501,
  restarts: 8,
  iterations: 560,
  learningRate: 0.027,
  atomRelativeStrengthThreshold: 0.018,
  atomAbsoluteStrengthThreshold: 0.003,
  pruneValidationToleranceRatio: 0.2,
  pruneValidationToleranceAbsolute: 0.008
});
const hookInputs = hookRows.map(row => row.mlpInput);
const hookActual = hookRows.map(row => row.targetOutput);
const hookPredicted = predictGeluFfnStructureBatch(model, hookInputs);
const hookMetrics = regressionMetrics(hookActual, hookPredicted);
const hookWorstError = worstAbsoluteError(hookActual, hookPredicted);
const agreementMae = modelAgreementMae(model, sources, block.projBias[block.targetOutputIndex], hookInputs);
const provenance = verifyGeluFfnNeuronProvenance(model, sources, {
  probeInputs: hookInputs.slice(0, 320),
  minDirectionCosine: 0.99,
  maxInputWeightNormError: 0.24,
  maxInputBiasError: 0.18,
  maxOutputWeightError: 0.24,
  minActivationCorrelation: 0.994,
  minContributionCorrelation: 0.994,
  maxContributionMae: 0.02,
  minConfidence: 0.74
});
const matchConfidences = provenance.matches.map(match => match.confidence);
const gates = {
  hookRowsCaptured: hookRows.length >= 300,
  hookShapesValid: hookRows.every(row => row.mlpInput.length === block.dimension && row.fcPreactivations.length === block.expansionCount),
  hookPreactivationIntegrity: hookIntegrity.maxPreactivationError < 1e-10,
  hookActivationIntegrity: hookIntegrity.maxActivationError < 1e-10,
  hookOutputIntegrity: hookIntegrity.maxTargetOutputError < 1e-10,
  recoveredExpectedEffectiveChannels: model.effectiveChannelCount === sources.length,
  hookOutputFaithful: hookMetrics.r2 > 0.998 && hookMetrics.mae < 0.015 && hookWorstError < 0.08,
  modelAgreementSmall: agreementMae < 0.015,
  provenanceVerified: provenance.verified,
  allSourceGroupsMapped: provenance.gates.allSourceGroupsMatched,
  whyReactEvidencePresent: provenance.matches.every(match => match.whyReacts.topTriggers.length >= 3),
  rawOracleCallsBelow140000: model.queryBudget.rawOracleCalls < 140000
};
const pass = Object.values(gates).every(Boolean);
const summary = {
  pass,
  hookRowCount: hookRows.length,
  promptCount: new Set(hookRows.map(row => row.promptId)).size,
  dimension: block.dimension,
  expansionCount: block.expansionCount,
  targetOutputIndex: block.targetOutputIndex,
  sourceNeuronCount: sources.length,
  effectiveChannelCount: model.effectiveChannelCount,
  rawChannelCount: model.rawChannelCount,
  hookMetrics,
  hookWorstError,
  agreementMae,
  hookIntegrity,
  sourceGroupCount: provenance.sourceGroupCount,
  minProvenanceConfidence: matchConfidences.length ? Math.min(...matchConfidences) : 0,
  rawOracleCalls: model.queryBudget.rawOracleCalls
};
const report = {
  generatedAt: new Date().toISOString(),
  task: 'Local transformer MLP hook GELU structure recovery',
  blackBoxBoundary: {
    model: 'deterministic local transformer-style block with token embeddings, causal residual mixing, layer norm, and GELU MLP',
    solverInput: 'oracle over hooked MLP input vectors plus source-neuron manifest produced by the local hook',
    hookRows: 'captured residual input, MLP input, fc preactivation, GELU activation, per-source contribution, and target output for tokenized prompt rows'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit GELU source structure for a hooked local transformer MLP output slice and map recovered atoms to source expansion neurons using real captured activation rows and source provenance.',
    doesNotClaim: 'full GPT-2/LLM FFN extraction, arbitrary transformer block decompilation, output-only original-neuron identity without hooks, or broad SOTA status.'
  },
  summary,
  gates,
  sources: sources.map(source => ({
    id: source.id,
    layer: source.layer,
    index: source.index,
    origin: source.origin,
    inputWeightNorm: norm(source.inputWeight),
    inputBias: source.inputBias,
    outputWeight: source.outputWeight
  })),
  model: {
    recoveryKind: model.recoveryKind,
    effectiveChannelCount: model.effectiveChannelCount,
    rawChannelCount: model.rawChannelCount,
    selectedCandidateKind: model.adaptiveSelection?.selectedCandidateKind,
    validationMetrics: model.validationMetrics,
    queryBudget: model.queryBudget
  },
  provenance,
  rows: provenance.matches,
  runtimeMs: Date.now() - started
};

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${makeMarkdown(report)}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-local-transformer-mlp-hook',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Release-gated local transformer MLP hook benchmark.',
    'Uses captured residual/MLP/preactivation/activation/output rows from a deterministic local transformer-style block.',
    'Provenance-mode source mapping requires the hook/source manifest and does not claim output-only original-neuron identity.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
