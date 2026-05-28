#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerBatch
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { buildDeepBlackBoxReluNdAdjacencyGeometry } from '../src/solvers/deep-blackbox-relu-nd-adjacency-geometry';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

type Unit = { w: number[]; b: number };
type Teacher = {
  id: string;
  seed: number;
  dimension: number;
  l1: Unit[];
  l2: Unit[];
  out: { w: number[]; b: number; base: number[] };
};

const repoRoot = path.resolve(__dirname, '..');
const outputDir = path.join(repoRoot, 'benchmarks', 'research', 'nn-nd-adjacency-geometry-5d', 'single-probe');
const jsonOutputPath = path.join(outputDir, 'report.json');
const markdownOutputPath = path.join(outputDir, 'report.md');

function relu(value: number): number {
  return Math.max(0, value);
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, Math.sqrt(dot(values, values)));
  return values.map(value => value / length);
}

function makeTeacher(seed: number, dimension: number): Teacher {
  const random = makeRandom(seed);
  const l1 = Array.from({ length: 4 }, () => ({
    w: normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
      .map(value => value * randomBetween(random, 0.6, 1.45)),
    b: randomBetween(random, -0.75, 0.75)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      return sign * randomBetween(random, 0.2, 1.0);
    }),
    b: randomBetween(random, -0.55, 0.55)
  }));
  return {
    id: `clean-5d-${seed}`,
    seed,
    dimension,
    l1,
    l2,
    out: {
      w: Array.from({ length: 4 }, (_, index) => {
        const sign = index % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.3, 1.2);
      }),
      b: randomBetween(random, -0.1, 0.1),
      base: Array.from({ length: dimension }, () => randomBetween(random, -0.06, 0.06))
    }
  };
}

function predictTeacher(teacher: Teacher, input: number[]): number {
  const h1 = teacher.l1.map(unit => relu(dot(unit.w, input) + unit.b));
  const h2 = teacher.l2.map(unit => relu(dot(unit.w, h1) + unit.b));
  return teacher.out.b + dot(teacher.out.base, input) + dot(teacher.out.w, h2);
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => randomBetween(random, -1.75, 1.75))
  ));
}

fs.mkdirSync(outputDir, { recursive: true });
const started = Date.now();
const teacher = makeTeacher(29, 5);
let rawOracleCalls = 0;
const oracle = (input: number[]): number => {
  rawOracleCalls++;
  return predictTeacher(teacher, input);
};

const samplerConfig = {
  dimension: 5,
  ranges: Array.from({ length: 5 }, () => [-2, 2] as [number, number]),
  sampleCount: 18000,
  gradientStep: 1e-5,
  slopeTolerance: 1e-5,
  interceptTolerance: 1e-5,
  minRegionSupport: 6,
  neighborCount: 21,
  predictionMode: 'region-vote' as const,
  randomSeed: 260526 + teacher.seed + teacher.dimension,
  ridge: 1e-8
};
const geometryConfig = {
  assignmentSampleLimit: 3600,
  neighborCount: 14,
  minContactCount: 3,
  minWitnessCount: 2,
  minTransitionAgreement: 0.4,
  maxNormalizedProjectionRmse: 1.25,
  minPassingEdges: 30,
  minLargestComponentRatio: 0.12
};

const model = decompileDeepBlackBoxReluNdRegionSampler(oracle, samplerConfig);
const geometry = buildDeepBlackBoxReluNdAdjacencyGeometry(model, geometryConfig);
const validationInputs = makeInputs(310000 + teacher.seed + teacher.dimension, teacher.dimension, 2400);
const actual = validationInputs.map(input => predictTeacher(teacher, input));
const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, validationInputs);
const metrics = regressionMetrics(actual, predicted);
const hardCriteria = {
  r2Above099: metrics.r2 > 0.99,
  regionCountAbove50: model.regionCount > 50,
  oracleCallsBelow2M: rawOracleCalls < 2_000_000
};
const geometryCriteria = {
  geometryPass: geometry.pass,
  edgeCountAbove30: geometry.edgeCount >= 30,
  largestComponentAbove12Percent: geometry.largestComponentRatio >= 0.12
};
const pass = Object.values(hardCriteria).every(Boolean) && Object.values(geometryCriteria).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: 'single 5D ND adjacency-geometry probe for deep black-box ReLU region sampler',
  blackBox: {
    type: '5D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teacher',
    visibilityToGcl: 'oracle calls only: f(vector), no hidden activations, layer widths, teacher weights, or true adjacency'
  },
  claimScope: {
    provesIfPasses: 'a single controlled 5D clean folded-ReLU probe satisfies the user-defined next-step thresholds',
    doesNotClaim: '5D release readiness, noisy 5D robustness, 6D readiness, topology-perfect full cell-complex recovery, or superiority over external extraction methods'
  },
  hardCriteria,
  geometryCriteria,
  pass,
  teacher: {
    id: teacher.id,
    seed: teacher.seed,
    dimension: teacher.dimension,
    firstLayerWidth: teacher.l1.length,
    secondLayerWidth: teacher.l2.length
  },
  recovery: {
    recoveryKind: model.recoveryKind,
    regionCount: model.regionCount,
    deepOnlyRegionEstimate: model.regionCount - teacher.l1.length - 1,
    retainedSampleCount: model.retainedSampleCount,
    droppedSampleCount: model.droppedSampleCount,
    retainedSampleRatio: model.retainedSampleCount / model.sampleCount,
    predictionMode: model.predictionMode
  },
  geometry: {
    pass: geometry.pass,
    candidatePairCount: geometry.candidatePairCount,
    edgeCount: geometry.edgeCount,
    largestComponentSize: geometry.largestComponentSize,
    largestComponentRatio: geometry.largestComponentRatio,
    maxNormalizedProjectionRmse: geometry.maxNormalizedProjectionRmse,
    maxProjectionRmse: geometry.maxProjectionRmse,
    minSupportScore: geometry.minSupportScore,
    sampleEdges: geometry.edges.slice(0, 8).map(edge => ({
      id: edge.id,
      regionA: edge.regionA,
      regionB: edge.regionB,
      contacts: edge.contacts,
      witnessCount: edge.witnessCount,
      transitionAgreement: edge.transitionAgreement,
      normalizedProjectionRmse: edge.normalizedProjectionRmse,
      supportScore: edge.supportScore
    }))
  },
  metrics,
  queryBudget: {
    rawOracleCalls,
    finiteDifferenceCallsPerSample: 1 + 2 * teacher.dimension,
    validationPoints: validationInputs.length,
    geometryOracleCalls: 0
  },
  nextStep: pass
    ? '5D clean probe passed. Next valid step is a release-candidate 5D sweep before any 6D attempt.'
    : '5D probe failed. Improve the ND sampler or geometry witness builder before attempting 6D.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# 5D ND Adjacency-Geometry Probe',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Result',
  '',
  `- Pass: ${report.pass}`,
  `- R2: ${metrics.r2}`,
  `- MAE: ${metrics.mae}`,
  `- Region count: ${model.regionCount}`,
  `- Raw oracle calls: ${rawOracleCalls}`,
  `- Geometry pass: ${geometry.pass}`,
  `- Passing adjacency edges: ${geometry.edgeCount}`,
  `- Largest-component ratio: ${geometry.largestComponentRatio}`,
  `- Max normalized projection RMSE: ${geometry.maxNormalizedProjectionRmse}`,
  '',
  '## Hard Criteria',
  '',
  `- R2 > 0.99: ${hardCriteria.r2Above099}`,
  `- regionCount > 50: ${hardCriteria.regionCountAbove50}`,
  `- oracleCalls < 2000000: ${hardCriteria.oracleCallsBelow2M}`,
  '',
  '## Claim Boundary',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Next Step',
  '',
  report.nextStep
].join('\n');

fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(markdownOutputPath, `${markdown}\n`);
console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, jsonOutputPath),
  r2: metrics.r2,
  mae: metrics.mae,
  regionCount: model.regionCount,
  rawOracleCalls,
  geometry: {
    pass: geometry.pass,
    edgeCount: geometry.edgeCount,
    largestComponentRatio: geometry.largestComponentRatio,
    maxNormalizedProjectionRmse: geometry.maxNormalizedProjectionRmse
  },
  hardCriteria,
  geometryCriteria,
  nextStep: report.nextStep
}, null, 2));

if (!pass) process.exit(1);
