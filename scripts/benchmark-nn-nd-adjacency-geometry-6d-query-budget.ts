#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerBatch
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { buildDeepBlackBoxReluNdAdjacencyGeometry } from '../src/solvers/deep-blackbox-relu-nd-adjacency-geometry';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Profile = 'clean' | 'near-parallel' | 'cancellation' | 'low-amplitude' | 'weak-support';
type Tier = 'low' | 'mid' | 'high';
type Unit = { w: number[]; b: number };
type Teacher = {
  id: string;
  seed: number;
  dimension: number;
  profile: Profile;
  l1: Unit[];
  l2: Unit[];
  out: { w: number[]; b: number; base: number[] };
};
type SamplerConfig = {
  sampleCount: number;
  querySamples: number;
  gradientStep: number;
  slopeTolerance: number;
  interceptTolerance: number;
  minRegionSupport: number;
  neighborCount: number;
  predictionMode: 'region-vote' | 'local-affine-knn';
  localAffineNeighborCount?: number;
  aggregation?: 'mean' | 'median' | 'trimmed-mean';
  randomSeed: number;
};
type GeometryConfig = {
  assignmentSampleLimit: number;
  neighborCount: number;
  minContactCount: number;
  minWitnessCount: number;
  minTransitionAgreement: number;
  maxNormalizedProjectionRmse: number;
  minPassingEdges: number;
  minLargestComponentRatio: number;
};
type Gates = {
  maxMae: number;
  minR2: number;
  minRegionCount: number;
  maxSelectedRawOracleCalls: number;
};
type Candidate = {
  id: string;
  tier: Tier;
  sampler: SamplerConfig;
};
type TeacherCase = {
  teacher: Teacher;
  noisy: boolean;
  candidates: Candidate[];
  geometry: GeometryConfig;
  gates: Gates;
};

const repoRoot = path.resolve(__dirname, '..');
const previousSweepSelectedRawOracleCalls = 8_105_500;
const previousSweepSearchRawOracleCalls = 19_220_500;
process.env.GCL_BENCHMARK_ROOT = process.env.GCL_BENCHMARK_ROOT ?? path.join('benchmarks', 'research');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-nd-adjacency-geometry-6d-query-budget',
  'budget-ladder-hard-profiles'
);

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

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, Math.sqrt(dot(values, values)));
  return values.map(value => value / length);
}

function makeDirections(random: () => number, dimension: number, profile: Profile): number[][] {
  if (profile === 'near-parallel') {
    const base = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
    return Array.from({ length: 4 }, (_, index) => {
      const perturbation = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
        .map(value => value * (0.035 + index * 0.025));
      return normalize(base.map((value, axis) => value + perturbation[axis]));
    });
  }
  if (profile === 'weak-support') {
    const dominant = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
    return Array.from({ length: 4 }, (_, index) => {
      const direction = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
      return normalize(direction.map((value, axis) => value + (index === 0 ? 0 : 0.18) * dominant[axis]));
    });
  }
  return Array.from({ length: 4 }, () => normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1))));
}

function makeTeacher(seed: number, dimension: number, profile: Profile): Teacher {
  const random = makeRandom(seed);
  const directions = makeDirections(random, dimension, profile);
  const firstLayerScale = profile === 'low-amplitude' ? [0.45, 0.9] : [0.6, 1.35];
  const l1 = directions.map((direction, index) => ({
    w: direction.map(value => value * randomBetween(random, firstLayerScale[0], firstLayerScale[1])),
    b: profile === 'weak-support' && index === 3
      ? randomBetween(random, 0.95, 1.25)
      : randomBetween(random, -0.75, 0.75)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      const amplitude = profile === 'cancellation'
        ? randomBetween(random, 0.5, 1.05)
        : randomBetween(random, 0.2, 1.0);
      return sign * amplitude * (profile === 'weak-support' && weightIndex === 3 ? 0.45 : 1);
    }),
    b: profile === 'weak-support' && unitIndex === 3
      ? randomBetween(random, 0.6, 1.0)
      : randomBetween(random, -0.55, 0.55)
  }));
  const outputScale = profile === 'low-amplitude' ? 0.65 : 1;
  return {
    id: `${profile}-6d-${seed}`,
    seed,
    dimension,
    profile,
    l1,
    l2,
    out: {
      w: Array.from({ length: 4 }, (_, index) => {
        const sign = index % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.3, 1.2) * outputScale;
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

function makeNoisyOracle(teacher: Teacher): (input: number[]) => number {
  const random = makeRandom(160000 + teacher.seed + teacher.dimension);
  return (input: number[]) => {
    const noiseScale = teacher.profile === 'low-amplitude' || teacher.profile === 'weak-support' ? 0.0018 : 0.0025;
    const noisy = predictTeacher(teacher, input) + gaussian(random) * noiseScale;
    return Math.round(noisy / 0.001) * 0.001;
  };
}

function makeInputs(seed: number, dimension: number, count: number): number[][] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => (
    Array.from({ length: dimension }, () => randomBetween(random, -1.75, 1.75))
  ));
}

function makeCandidate(id: string, tier: Tier, sampler: Omit<SamplerConfig, 'randomSeed'>, randomSeed: number): Candidate {
  return { id, tier, sampler: { ...sampler, randomSeed } };
}

function makeCases(): TeacherCase[] {
  const cleanGeometry = {
    assignmentSampleLimit: 4200,
    neighborCount: 18,
    minContactCount: 3,
    minWitnessCount: 2,
    minTransitionAgreement: 0.35,
    maxNormalizedProjectionRmse: 1.6,
    minPassingEdges: 35,
    minLargestComponentRatio: 0.1
  };
  const noisyGeometry = {
    assignmentSampleLimit: 4200,
    neighborCount: 22,
    minContactCount: 4,
    minWitnessCount: 2,
    minTransitionAgreement: 0.22,
    maxNormalizedProjectionRmse: 2.15,
    minPassingEdges: 35,
    minLargestComponentRatio: 0.1
  };
  const cleanBase = {
    querySamples: 1,
    gradientStep: 1e-5,
    slopeTolerance: 1e-5,
    interceptTolerance: 1e-5,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'region-vote' as const
  };
  const cancellationBase = {
    querySamples: 7,
    gradientStep: 0.08,
    slopeTolerance: 0.12,
    interceptTolerance: 0.28,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 32,
    aggregation: 'median' as const
  };
  const nearBase = {
    querySamples: 7,
    gradientStep: 0.04,
    slopeTolerance: 0.05,
    interceptTolerance: 0.12,
    minRegionSupport: 4,
    neighborCount: 27,
    predictionMode: 'region-vote' as const,
    localAffineNeighborCount: 16,
    aggregation: 'median' as const
  };
  const lowAmplitudeBase = {
    querySamples: 7,
    gradientStep: 0.055,
    slopeTolerance: 0.08,
    interceptTolerance: 0.18,
    minRegionSupport: 5,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 64,
    aggregation: 'median' as const
  };
  const weakBase = {
    querySamples: 7,
    gradientStep: 0.085,
    slopeTolerance: 0.09,
    interceptTolerance: 0.24,
    minRegionSupport: 4,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 48,
    aggregation: 'median' as const
  };
  return [
    {
      teacher: makeTeacher(29, 6, 'clean'),
      noisy: false,
      candidates: [
        makeCandidate('clean-low', 'low', { ...cleanBase, sampleCount: 14000 }, 360561),
        makeCandidate('clean-mid', 'mid', { ...cleanBase, sampleCount: 18000 }, 360561),
        makeCandidate('clean-high', 'high', { ...cleanBase, sampleCount: 26000 }, 360561)
      ],
      geometry: cleanGeometry,
      gates: { maxMae: 0.055, minR2: 0.99, minRegionCount: 56, maxSelectedRawOracleCalls: 1_600_000 }
    },
    {
      teacher: makeTeacher(83, 6, 'clean'),
      noisy: false,
      candidates: [
        makeCandidate('clean-low', 'low', { ...cleanBase, sampleCount: 14000 }, 360615),
        makeCandidate('clean-mid', 'mid', { ...cleanBase, sampleCount: 18000 }, 360615),
        makeCandidate('clean-high', 'high', { ...cleanBase, sampleCount: 26000 }, 360615)
      ],
      geometry: cleanGeometry,
      gates: { maxMae: 0.06, minR2: 0.99, minRegionCount: 56, maxSelectedRawOracleCalls: 1_600_000 }
    },
    {
      teacher: makeTeacher(717, 6, 'cancellation'),
      noisy: true,
      candidates: [
        makeCandidate('cancellation-low', 'low', { ...cancellationBase, sampleCount: 12000 }, 361249),
        makeCandidate('cancellation-mid', 'mid', { ...cancellationBase, sampleCount: 14000 }, 361249),
        makeCandidate('cancellation-high', 'high', { ...cancellationBase, sampleCount: 16000 }, 361249)
      ],
      geometry: noisyGeometry,
      gates: { maxMae: 0.095, minR2: 0.965, minRegionCount: 56, maxSelectedRawOracleCalls: 1_600_000 }
    },
    {
      teacher: makeTeacher(805, 6, 'near-parallel'),
      noisy: true,
      candidates: [
        makeCandidate('near-parallel-low', 'low', { ...nearBase, sampleCount: 13000 }, 360001),
        makeCandidate('near-parallel-mid', 'mid', { ...nearBase, sampleCount: 15000 }, 360001),
        makeCandidate('near-parallel-high', 'high', { ...nearBase, sampleCount: 17000 }, 360001)
      ],
      geometry: noisyGeometry,
      gates: { maxMae: 0.1, minR2: 0.96, minRegionCount: 56, maxSelectedRawOracleCalls: 1_600_000 }
    },
    {
      teacher: makeTeacher(634, 6, 'low-amplitude'),
      noisy: true,
      candidates: [
        makeCandidate('low-amplitude-low', 'low', { ...lowAmplitudeBase, sampleCount: 10000 }, 460001),
        makeCandidate('low-amplitude-mid', 'mid', { ...lowAmplitudeBase, sampleCount: 12000 }, 460001),
        makeCandidate('low-amplitude-high', 'high', { ...lowAmplitudeBase, sampleCount: 14000 }, 460001)
      ],
      geometry: noisyGeometry,
      gates: { maxMae: 0.075, minR2: 0.96, minRegionCount: 56, maxSelectedRawOracleCalls: 1_600_000 }
    },
    {
      teacher: makeTeacher(911, 6, 'weak-support'),
      noisy: true,
      candidates: [
        makeCandidate('weak-support-low', 'low', { ...weakBase, sampleCount: 11000 }, 360001),
        makeCandidate('weak-support-mid', 'mid', { ...weakBase, sampleCount: 13000 }, 360001),
        makeCandidate('weak-support-high', 'high', { ...weakBase, sampleCount: 15000 }, 360001)
      ],
      geometry: noisyGeometry,
      gates: { maxMae: 0.09, minR2: 0.96, minRegionCount: 56, maxSelectedRawOracleCalls: 1_600_000 }
    }
  ];
}

function scoreCandidate(candidate: ReturnType<typeof evaluateCandidate>): number {
  return candidate.metrics.r2 * 100
    - candidate.metrics.mae
    + Math.min(candidate.recovery.regionCount, 140) / 140
    + Math.min(candidate.geometry.edgeCount, 1200) / 1200;
}

function evaluateCandidate(testCase: TeacherCase, candidate: Candidate) {
  let rawOracleCalls = 0;
  const teacher = testCase.teacher;
  const baseOracle = testCase.noisy ? makeNoisyOracle(teacher) : (input: number[]) => predictTeacher(teacher, input);
  const countedOracle = (input: number[]): number => {
    rawOracleCalls++;
    return baseOracle(input);
  };
  const model = decompileDeepBlackBoxReluNdRegionSampler(countedOracle, {
    dimension: teacher.dimension,
    ranges: Array.from({ length: teacher.dimension }, () => [-2, 2] as [number, number]),
    sampleCount: candidate.sampler.sampleCount,
    gradientStep: candidate.sampler.gradientStep,
    slopeTolerance: candidate.sampler.slopeTolerance,
    interceptTolerance: candidate.sampler.interceptTolerance,
    minRegionSupport: candidate.sampler.minRegionSupport,
    querySamples: candidate.sampler.querySamples,
    aggregation: candidate.sampler.aggregation ?? 'mean',
    neighborCount: candidate.sampler.neighborCount,
    predictionMode: candidate.sampler.predictionMode,
    localAffineNeighborCount: candidate.sampler.localAffineNeighborCount,
    randomSeed: candidate.sampler.randomSeed,
    ridge: 1e-8
  });
  const geometry = buildDeepBlackBoxReluNdAdjacencyGeometry(model, testCase.geometry);
  const validationInputs = makeInputs(410000 + teacher.seed + teacher.dimension, teacher.dimension, 2600);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const criteria = {
    geometryPass: geometry.pass,
    edgeCountEnough: geometry.edgeCount >= testCase.geometry.minPassingEdges,
    largestComponentEnough: geometry.largestComponentRatio >= testCase.geometry.minLargestComponentRatio,
    projectionRmseBounded: geometry.maxNormalizedProjectionRmse <= testCase.geometry.maxNormalizedProjectionRmse,
    maeBelowGate: metrics.mae < testCase.gates.maxMae,
    r2AboveGate: metrics.r2 > testCase.gates.minR2,
    regionCountEnough: model.regionCount >= testCase.gates.minRegionCount,
    selectedOracleCallsBelowBudget: rawOracleCalls < testCase.gates.maxSelectedRawOracleCalls
  };
  const pass = Object.values(criteria).every(Boolean);
  return {
    candidateId: candidate.id,
    tier: candidate.tier,
    randomSeed: candidate.sampler.randomSeed,
    pass,
    criteria,
    score: 0,
    sampler: candidate.sampler,
    recovery: {
      recoveryKind: model.recoveryKind,
      regionCount: model.regionCount,
      firstLayerWidth: teacher.l1.length,
      deepOnlyRegionEstimate: model.regionCount - teacher.l1.length - 1,
      retainedSampleCount: model.retainedSampleCount,
      droppedSampleCount: model.droppedSampleCount,
      retainedSampleRatio: model.retainedSampleCount / model.sampleCount,
      predictionMode: model.predictionMode
    },
    geometry: {
      pass: geometry.pass,
      geometryKind: geometry.geometryKind,
      candidatePairCount: geometry.candidatePairCount,
      edgeCount: geometry.edgeCount,
      largestComponentSize: geometry.largestComponentSize,
      largestComponentRatio: geometry.largestComponentRatio,
      maxNormalizedProjectionRmse: geometry.maxNormalizedProjectionRmse,
      maxProjectionRmse: geometry.maxProjectionRmse,
      minSupportScore: geometry.minSupportScore
    },
    metrics,
    queryBudget: {
      rawOracleCalls,
      finiteDifferenceCallsPerSample: (1 + 2 * teacher.dimension) * candidate.sampler.querySamples,
      geometryOracleCalls: 0,
      validationPoints: validationInputs.length
    }
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const results = makeCases().map(testCase => {
  const attempts: Array<ReturnType<typeof evaluateCandidate> & { score: number }> = [];
  for (const candidate of testCase.candidates) {
    const evaluated = evaluateCandidate(testCase, candidate);
    const scored = { ...evaluated, score: scoreCandidate(evaluated) };
    attempts.push(scored);
    if (scored.pass) break;
  }
  const selected = attempts.find(candidate => candidate.pass)
    ?? [...attempts].sort((a, b) => b.score - a.score)[0];
  const rejected = attempts.filter(candidate => candidate.candidateId !== selected.candidateId);
  return {
    teacherId: testCase.teacher.id,
    seed: testCase.teacher.seed,
    dimension: testCase.teacher.dimension,
    profile: testCase.teacher.profile,
    noisy: testCase.noisy,
    pass: selected.pass,
    selection: {
      selectedCandidateId: selected.candidateId,
      selectedTier: selected.tier,
      selectedRandomSeed: selected.randomSeed,
      attemptedCandidateCount: attempts.length,
      rejectedCandidateCount: rejected.length,
      rejectedFailCount: rejected.filter(candidate => !candidate.pass).length,
      totalSearchRawOracleCalls: attempts.reduce((sum, candidate) => sum + candidate.queryBudget.rawOracleCalls, 0),
      selectedRawOracleCalls: selected.queryBudget.rawOracleCalls,
      policy: 'evaluate low/mid/high candidates in order and stop at the first candidate that passes all geometry/accuracy/region gates'
    },
    selected,
    rejectedCandidates: rejected.map(candidate => ({
      candidateId: candidate.candidateId,
      tier: candidate.tier,
      pass: candidate.pass,
      criteria: candidate.criteria,
      score: candidate.score,
      recovery: candidate.recovery,
      geometry: candidate.geometry,
      metrics: candidate.metrics,
      queryBudget: candidate.queryBudget
    }))
  };
});

const selectedResults = results.map(result => result.selected);
const totalSelectedRawOracleCalls = results.reduce((sum, result) => sum + result.selection.selectedRawOracleCalls, 0);
const totalSearchRawOracleCalls = results.reduce((sum, result) => sum + result.selection.totalSearchRawOracleCalls, 0);
const selectedReduction = 1 - totalSelectedRawOracleCalls / previousSweepSelectedRawOracleCalls;
const searchReduction = 1 - totalSearchRawOracleCalls / previousSweepSearchRawOracleCalls;
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  cleanTeacherCount: results.filter(result => !result.noisy).length,
  noisyTeacherCount: results.filter(result => result.noisy).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  profiles: [...new Set(results.map(result => result.profile))],
  selectedTiers: results.map(result => result.selection.selectedTier),
  lowTierSelections: results.filter(result => result.selection.selectedTier === 'low').length,
  midTierSelections: results.filter(result => result.selection.selectedTier === 'mid').length,
  highTierSelections: results.filter(result => result.selection.selectedTier === 'high').length,
  attemptedCandidateCount: results.reduce((sum, result) => sum + result.selection.attemptedCandidateCount, 0),
  rejectedCandidateCount: results.reduce((sum, result) => sum + result.selection.rejectedCandidateCount, 0),
  rejectedFailCount: results.reduce((sum, result) => sum + result.selection.rejectedFailCount, 0),
  minRegionCount: Math.min(...selectedResults.map(result => result.recovery.regionCount)),
  minEdgeCount: Math.min(...selectedResults.map(result => result.geometry.edgeCount)),
  minLargestComponentRatio: Math.min(...selectedResults.map(result => result.geometry.largestComponentRatio)),
  maxNormalizedProjectionRmse: Math.max(...selectedResults.map(result => result.geometry.maxNormalizedProjectionRmse)),
  maxMae: Math.max(...selectedResults.map(result => result.metrics.mae)),
  minR2: Math.min(...selectedResults.map(result => result.metrics.r2)),
  maxSelectedRawOracleCallsPerTeacher: Math.max(...results.map(result => result.selection.selectedRawOracleCalls)),
  totalSelectedRawOracleCalls,
  totalSearchRawOracleCalls,
  previousSweepSelectedRawOracleCalls,
  previousSweepSearchRawOracleCalls,
  selectedReduction,
  searchReduction
};

const gates = {
  pass: summary.pass,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  cleanAndNoisyCovered: summary.cleanTeacherCount >= 2 && summary.noisyTeacherCount >= 4,
  only6dTeachers: summary.dimensions.length === 1 && summary.dimensions[0] === 6,
  hardProfilesCovered: ['clean', 'near-parallel', 'cancellation', 'low-amplitude', 'weak-support']
    .every(profile => summary.profiles.includes(profile as Profile)),
  budgetLadderActuallyRejectedFailures: summary.rejectedFailCount >= 1,
  minRegionCountAbove55: summary.minRegionCount >= 56,
  minEdgesAtLeast35: summary.minEdgeCount >= 35,
  largestComponentRatioAtLeast10Percent: summary.minLargestComponentRatio >= 0.1,
  normalizedProjectionRmseAtMost2_15: summary.maxNormalizedProjectionRmse <= 2.15,
  cleanR2Above099: results.filter(result => !result.noisy).every(result => result.selected.metrics.r2 > 0.99),
  noisyR2Above096: results.filter(result => result.noisy).every(result => result.selected.metrics.r2 > 0.96),
  maxSelectedRawOracleCallsPerTeacherBelow1_6M: summary.maxSelectedRawOracleCallsPerTeacher < 1_600_000,
  selectedRawOracleReductionAtLeast30Percent: summary.selectedReduction >= 0.3,
  searchRawOracleCallsBelowPreviousSweepSearch: summary.totalSearchRawOracleCalls < previousSweepSearchRawOracleCalls
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: '6D query-budget ladder for recovered-adjacency geometry on deep black-box ReLU region samplers',
  blackBox: {
    type: '6D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(vector), not hidden activations, layer widths, teacher weights, or true region adjacency'
  },
  claimScope: {
    provesIfPasses: 'the 6D recovered-adjacency lane keeps the hard-profile accuracy/geometry gates while reducing selected raw oracle calls versus the prior 6D sweep',
    doesNotClaim: 'SOTA query efficiency, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, or arbitrary production API extraction'
  },
  gates,
  summary,
  results,
  nextStep: pass
    ? '6D query-budget ladder passed. Next valid step is promotion-gate integration or a second efficiency pass that targets the near-parallel/cancellation high-cost profiles.'
    : '6D query-budget ladder failed. Fix the failing budget/profile before claiming query-cost improvement.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# 6D ND Recovered-Adjacency Geometry Query-Budget Ladder',
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
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Selected tiers: ${summary.selectedTiers.join(', ')}`,
  `- Low/mid/high selections: ${summary.lowTierSelections}/${summary.midTierSelections}/${summary.highTierSelections}`,
  `- Rejected failed candidates: ${summary.rejectedFailCount}`,
  `- Min region count: ${summary.minRegionCount}`,
  `- Min adjacency edges: ${summary.minEdgeCount}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Max selected raw oracle calls per teacher: ${summary.maxSelectedRawOracleCallsPerTeacher}`,
  `- Previous selected raw oracle calls: ${summary.previousSweepSelectedRawOracleCalls}`,
  `- New selected raw oracle calls: ${summary.totalSelectedRawOracleCalls}`,
  `- Selected-call reduction: ${summary.selectedReduction}`,
  `- Previous search raw oracle calls: ${summary.previousSweepSearchRawOracleCalls}`,
  `- New search raw oracle calls: ${summary.totalSearchRawOracleCalls}`,
  `- Search-call reduction: ${summary.searchReduction}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Pass: ${result.pass}`,
    `- Profile: ${result.profile}`,
    `- Noisy oracle: ${result.noisy}`,
    `- Selected candidate: ${result.selection.selectedCandidateId}`,
    `- Selected tier: ${result.selection.selectedTier}`,
    `- Candidates attempted: ${result.selection.attemptedCandidateCount}`,
    `- Rejected failures: ${result.selection.rejectedFailCount}`,
    `- Region count: ${result.selected.recovery.regionCount}`,
    `- Geometry pass: ${result.selected.geometry.pass}`,
    `- Passing edges: ${result.selected.geometry.edgeCount}`,
    `- GCL MAE: ${result.selected.metrics.mae}`,
    `- GCL R2: ${result.selected.metrics.r2}`,
    `- Selected raw oracle calls: ${result.selection.selectedRawOracleCalls}`,
    `- Search raw oracle calls: ${result.selection.totalSearchRawOracleCalls}`,
    ''
  ].join('\n')),
  '## Next Step',
  '',
  report.nextStep
].join('\n').trimEnd();

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-nd-adjacency-geometry-6d-query-budget',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Research query-budget hardening for the 6D recovered-adjacency lane.',
    'Evaluates low/mid/high candidates in order and stops at the first passing budget.',
    'Keeps the hard-profile geometry/accuracy gates while requiring selected-call reduction against the previous 6D sweep.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
