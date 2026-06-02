#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileRankAdaptiveSwigluFfnStructure,
  predictSwigluFfnStructureBatch,
  SwigluFfnStructurePair
} from '../src/solvers/swiglu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type TeacherBlock = {
  valueDirection: number[];
  valueScale: number;
  valueBias: number;
  gateDirection: number[];
  gateScale: number;
  gateBias: number;
  outputWeight: number;
};

type Teacher = {
  id: string;
  dimension: number;
  blocks: TeacherBlock[];
  outputBias: number;
  predict: (input: number[]) => number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-swiglu-ffn-mixed-structure',
  'rank-adaptive-two-block'
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

function makeDirection(seed: number, dimension: number, previous: number[][] = []): number[] {
  const random = makeRandom(seed);
  return normalize(subtractProjection(
    Array.from({ length: dimension }, () => gaussian(random)),
    previous
  ));
}

function silu(value: number): number {
  if (value > 40) return value;
  if (value < -40) return 0;
  return value / (1 + Math.exp(-value));
}

function makeInputs(seed: number, dimension: number, count: number, radius = 1.35): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => -radius + 2 * radius * random())
  ));
}

function makeTeacher(): Teacher {
  const dimension = 24;
  const directions: number[][] = [];
  for (const seed of [9101, 9202, 9303, 9404]) directions.push(makeDirection(seed, dimension, directions));
  const blocks: TeacherBlock[] = [
    {
      valueDirection: directions[0],
      valueScale: 1.05,
      valueBias: -0.18,
      gateDirection: directions[1],
      gateScale: 1.25,
      gateBias: 0.22,
      outputWeight: 1.15
    },
    {
      valueDirection: directions[2],
      valueScale: 0.82,
      valueBias: 0.14,
      gateDirection: directions[3],
      gateScale: 1.08,
      gateBias: -0.16,
      outputWeight: -0.88
    }
  ];
  return {
    id: 'mixed-swiglu-two-block-24d',
    dimension,
    blocks,
    outputBias: 0.045,
    predict: input => 0.045 + blocks.reduce((sum, block) => {
      const value = block.valueScale * dot(block.valueDirection, input) + block.valueBias;
      const gate = block.gateScale * dot(block.gateDirection, input) + block.gateBias;
      return sum + block.outputWeight * value * silu(gate);
    }, 0)
  };
}

function directionProjection(trueDirection: number[], recoveredWeight: number[]): number {
  return Math.abs(dot(trueDirection, normalize(recoveredWeight)));
}

function bestAssignment(
  blocks: TeacherBlock[],
  pairs: SwigluFfnStructurePair[],
  row = 0,
  used: boolean[] = new Array(pairs.length).fill(false),
  selected: Array<{ trueIndex: number; recoveredIndex: number; valueProjection: number; gateProjection: number; score: number }> = []
): { score: number; selected: Array<{ trueIndex: number; recoveredIndex: number; valueProjection: number; gateProjection: number; score: number }> } {
  if (row >= blocks.length) {
    return {
      score: selected.reduce((sum, item) => sum + item.score, 0),
      selected: [...selected]
    };
  }
  let best = { score: -Infinity, selected: [] as Array<{ trueIndex: number; recoveredIndex: number; valueProjection: number; gateProjection: number; score: number }> };
  for (let col = 0; col < pairs.length; col++) {
    if (used[col]) continue;
    const valueProjection = directionProjection(blocks[row].valueDirection, pairs[col].valueWeight);
    const gateProjection = directionProjection(blocks[row].gateDirection, pairs[col].gateWeight);
    const score = (valueProjection + gateProjection) / 2;
    used[col] = true;
    selected.push({ trueIndex: row, recoveredIndex: col, valueProjection, gateProjection, score });
    const candidate = bestAssignment(blocks, pairs, row + 1, used, selected);
    if (candidate.score > best.score) best = candidate;
    selected.pop();
    used[col] = false;
  }
  return best;
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teacher = makeTeacher();
const ranges = Array.from({ length: teacher.dimension }, () => [-1.35, 1.35] as [number, number]);
const model = decompileRankAdaptiveSwigluFfnStructure(teacher.predict, {
  dimension: teacher.dimension,
  ranges,
  maxPairCount: 3,
  minEffectivePairCount: 1,
  activeRank: 6,
  gradientProbeCount: 24,
  trainingSampleCount: 3200,
  validationSampleCount: 600,
  randomSeed: 9701,
  restarts: 12,
  iterations: 850,
  learningRate: 0.024,
  selectionRestarts: 1,
  pruneValidationToleranceAbsolute: 0.003,
  pruneValidationToleranceRatio: 0.12
});

const validationInputs = makeInputs(9801, teacher.dimension, 1100);
const actual = validationInputs.map(teacher.predict);
const predicted = predictSwigluFfnStructureBatch(model, validationInputs);
const metrics = regressionMetrics(actual, predicted);
const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
const meanBaseline = regressionMetrics(actual, actual.map(() => mean));
const assignment = bestAssignment(teacher.blocks, model.pairs);
const minValueDirectionProjection = Math.min(...assignment.selected.map(item => item.valueProjection));
const minGateDirectionProjection = Math.min(...assignment.selected.map(item => item.gateProjection));
const meanPairProjection = assignment.selected.reduce((sum, item) => sum + item.score, 0) / Math.max(1, assignment.selected.length);
const maxGateNormError = Math.max(...teacher.blocks.map((block, index) => {
  const matched = assignment.selected.find(item => item.trueIndex === index);
  if (!matched) return Infinity;
  return Math.abs(model.pairs[matched.recoveredIndex].gateWeightNorm - block.gateScale);
}));
const maxGateBiasError = Math.max(...teacher.blocks.map((block, index) => {
  const matched = assignment.selected.find(item => item.trueIndex === index);
  if (!matched) return Infinity;
  return Math.abs(Math.abs(model.pairs[matched.recoveredIndex].gateBias) - Math.abs(block.gateBias));
}));

const gates = {
  recoversRankAdaptiveStructure: model.recoveryKind === 'rank-adaptive-swiglu-ffn-structure',
  doesNotUseTruePairCount: model.rawPairCount === 3 && model.maxPairCount === 3,
  effectivePairCountMatches: model.effectivePairCount === teacher.blocks.length,
  usesInternalCandidateSelection: (model.adaptiveSelection?.candidateCount ?? 0) >= 3,
  duplicatePairsAbsent: (model.adaptiveSelection?.duplicatePairCount ?? 0) === 0,
  heldoutR2Above0995: metrics.r2 > 0.995,
  heldoutMaeBelow0035: metrics.mae < 0.035,
  minValueDirectionProjectionAbove099: minValueDirectionProjection > 0.99,
  minGateDirectionProjectionAbove099: minGateDirectionProjection > 0.99,
  meanPairProjectionAbove0993: meanPairProjection > 0.993,
  gateParameterErrorsBounded: maxGateNormError < 0.28 && maxGateBiasError < 0.28,
  predictionSanityBeatsMeanBy90Percent: metrics.mae < meanBaseline.mae * 0.1,
  rawOracleCallsBelow150k: model.queryBudget.rawOracleCalls < 150000
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'rank-adaptive mixed-block SwiGLU FFN structure recovery',
  blackBox: {
    type: '24D two-block SwiGLU FFN output slice with overcomplete max-pair budget',
    visibilityToGcl: 'oracle calls only: f(vector); true pair count, value/gate directions, biases, and readout weights are hidden'
  },
  claimScope: {
    provesIfPasses: 'GCL can recover explicit SwiGLU value/gate pair structure when given a maximum pair budget but not the true pair count',
    doesNotClaim: 'noisy SwiGLU recovery, full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production transformer block decompilation, semantic original-neuron identity from output-only access, or broad SOTA status'
  },
  teacher: {
    id: teacher.id,
    dimension: teacher.dimension,
    truePairCount: teacher.blocks.length,
    maxPairBudgetGivenToGcl: 3
  },
  gates: { ...gates, pass },
  summary: {
    pass,
    truePairCount: teacher.blocks.length,
    rawPairCount: model.rawPairCount,
    effectivePairCount: model.effectivePairCount,
    prunedWeakPairCount: model.adaptiveSelection?.prunedWeakPairCount ?? 0,
    duplicatePairCount: model.adaptiveSelection?.duplicatePairCount ?? 0,
    mae: metrics.mae,
    r2: metrics.r2,
    minValueDirectionProjection,
    minGateDirectionProjection,
    meanPairProjection,
    maxGateNormError,
    maxGateBiasError,
    rawOracleCalls: model.queryBudget.rawOracleCalls
  },
  predictionSanityBaselines: [
    {
      id: 'mean-output',
      family: 'constant',
      role: 'prediction-sanity only',
      metrics: meanBaseline
    }
  ],
  gcl: {
    metrics,
    expression: model.expression,
    pairs: model.pairs.map(pair => ({
      valueWeightNorm: pair.valueWeightNorm,
      valueBias: pair.valueBias,
      gateWeightNorm: pair.gateWeightNorm,
      gateBias: pair.gateBias,
      outputWeight: pair.outputWeight,
      strength: Math.abs(pair.outputWeight) * pair.valueWeightNorm * pair.gateWeightNorm
    })),
    adaptiveSelection: model.adaptiveSelection,
    structure: {
      assignment: assignment.selected,
      minValueDirectionProjection,
      minGateDirectionProjection,
      meanPairProjection,
      maxGateNormError,
      maxGateBiasError
    },
    queryBudget: model.queryBudget
  },
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Rank-Adaptive Mixed SwiGLU FFN Structure Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Structure Summary',
  '',
  `- Pass: ${pass}`,
  `- True pair count: ${report.summary.truePairCount}`,
  `- Max pair budget given to GCL: ${report.summary.rawPairCount}`,
  `- Effective recovered pair count: ${report.summary.effectivePairCount}`,
  `- Pruned weak pairs: ${report.summary.prunedWeakPairCount}`,
  `- Duplicate pairs: ${report.summary.duplicatePairCount}`,
  `- Min value-direction projection: ${report.summary.minValueDirectionProjection}`,
  `- Min gate-direction projection: ${report.summary.minGateDirectionProjection}`,
  `- Mean pair projection: ${report.summary.meanPairProjection}`,
  `- Max gate-norm error: ${report.summary.maxGateNormError}`,
  `- Max gate-bias error: ${report.summary.maxGateBiasError}`,
  `- Raw oracle calls: ${report.summary.rawOracleCalls}`,
  '',
  '## Prediction Sanity',
  '',
  `- MAE: ${report.summary.mae}`,
  `- R2: ${report.summary.r2}`,
  `- Mean baseline MAE: ${meanBaseline.mae}`,
  `- GCL/mean MAE ratio: ${report.summary.mae / Math.max(1e-12, meanBaseline.mae)}`,
  '',
  '## Gate',
  '',
  ...Object.entries(gates).map(([key, value]) => `- ${key}: ${value}`),
  '',
  '## Conclusion',
  '',
  pass
    ? 'The mixed SwiGLU structure gate passed. GCL recovered explicit value/gate pairs and the effective pair count without being told the true count.'
    : 'The mixed SwiGLU structure gate failed. Do not promote Stage C SwiGLU until the recorded failure is repaired.'
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-swiglu-ffn-mixed-structure',
  status: pass ? 'pass' : 'fail',
  summary: report.summary,
  notes: [
    'First Stage C mixed-block SwiGLU structure gate.',
    'GCL is given only a max pair budget, not the true pair count.',
    'The solver must return explicit value/gate pairs, not only an output predictor.',
    'Prediction baselines are sanity checks only and are not structure-decompiler competitors.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary: report.summary,
  gates: report.gates
}, null, 2));

if (!pass) process.exit(1);
