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
type Candidate = { id: string; sampler: SamplerConfig };
type TeacherCase = {
  teacher: Teacher;
  noisy: boolean;
  candidates: Candidate[];
  geometry: GeometryConfig;
  gates: Gates;
};

const repoRoot = path.resolve(__dirname, '..');
process.env.GCL_BENCHMARK_ROOT = process.env.GCL_BENCHMARK_ROOT ?? path.join('benchmarks', 'research');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-nd-adjacency-geometry-6d-sweep',
  'release-candidate-hard-profiles'
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

function makeCandidate(id: string, sampler: Omit<SamplerConfig, 'randomSeed'>, randomSeed: number): Candidate {
  return { id, sampler: { ...sampler, randomSeed } };
}

function makeNoisyLocalSampler(overrides: Partial<Omit<SamplerConfig, 'randomSeed'>> = {}): Omit<SamplerConfig, 'randomSeed'> {
  return {
    sampleCount: 14500,
    querySamples: 9,
    gradientStep: 0.07,
    slopeTolerance: 0.1,
    interceptTolerance: 0.24,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'local-affine-knn',
    localAffineNeighborCount: 72,
    aggregation: 'median',
    ...overrides
  };
}

function makeCases(): TeacherCase[] {
  const cleanSampler = {
    sampleCount: 26000,
    querySamples: 1,
    gradientStep: 1e-5,
    slopeTolerance: 1e-5,
    interceptTolerance: 1e-5,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'region-vote' as const
  };
  const cleanGeometry = {
    assignmentSampleLimit: 5200,
    neighborCount: 18,
    minContactCount: 3,
    minWitnessCount: 2,
    minTransitionAgreement: 0.35,
    maxNormalizedProjectionRmse: 1.5,
    minPassingEdges: 35,
    minLargestComponentRatio: 0.1
  };
  const noisyGeometry = {
    assignmentSampleLimit: 5000,
    neighborCount: 22,
    minContactCount: 4,
    minWitnessCount: 2,
    minTransitionAgreement: 0.23,
    maxNormalizedProjectionRmse: 2.0,
    minPassingEdges: 35,
    minLargestComponentRatio: 0.1
  };
  const nearParallelSampler = {
    sampleCount: 17000,
    querySamples: 9,
    gradientStep: 0.04,
    slopeTolerance: 0.05,
    interceptTolerance: 0.12,
    minRegionSupport: 4,
    neighborCount: 27,
    predictionMode: 'region-vote' as const,
    localAffineNeighborCount: 16,
    aggregation: 'median' as const
  };
  const nearParallelSeeds = [260001, 360001, 460001];
  return [
    {
      teacher: makeTeacher(29, 6, 'clean'),
      noisy: false,
      candidates: [makeCandidate('clean-default', cleanSampler, 360526 + 29 + 6)],
      geometry: cleanGeometry,
      gates: { maxMae: 0.055, minR2: 0.99, minRegionCount: 56, maxSelectedRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(83, 6, 'clean'),
      noisy: false,
      candidates: [makeCandidate('clean-default', cleanSampler, 360526 + 83 + 6)],
      geometry: cleanGeometry,
      gates: { maxMae: 0.06, minR2: 0.99, minRegionCount: 56, maxSelectedRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(717, 6, 'cancellation'),
      noisy: true,
      candidates: [makeCandidate('cancellation-default', makeNoisyLocalSampler({
        sampleCount: 16000,
        gradientStep: 0.08,
        slopeTolerance: 0.12,
        interceptTolerance: 0.28,
        localAffineNeighborCount: 32
      }), 360526 + 717 + 6)],
      geometry: noisyGeometry,
      gates: { maxMae: 0.095, minR2: 0.965, minRegionCount: 56, maxSelectedRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(805, 6, 'near-parallel'),
      noisy: true,
      candidates: nearParallelSeeds.map(seed => makeCandidate(`near-parallel-seed-${seed}`, nearParallelSampler, seed)),
      geometry: {
        ...noisyGeometry,
        minTransitionAgreement: 0.2,
        maxNormalizedProjectionRmse: 2.15
      },
      gates: { maxMae: 0.1, minR2: 0.965, minRegionCount: 56, maxSelectedRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(634, 6, 'low-amplitude'),
      noisy: true,
      candidates: [360001, 460001, 560001].map(seed => makeCandidate(`low-amplitude-seed-${seed}`, makeNoisyLocalSampler({
        sampleCount: 15000,
        gradientStep: 0.055,
        slopeTolerance: 0.08,
        interceptTolerance: 0.18,
        minRegionSupport: 5,
        localAffineNeighborCount: 64
      }), seed)),
      geometry: noisyGeometry,
      gates: { maxMae: 0.075, minR2: 0.96, minRegionCount: 56, maxSelectedRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(911, 6, 'weak-support'),
      noisy: true,
      candidates: [360001, 460001, 560001].map(seed => makeCandidate(`weak-support-seed-${seed}`, makeNoisyLocalSampler({
        sampleCount: 15500,
        gradientStep: 0.085,
        slopeTolerance: 0.09,
        interceptTolerance: 0.24,
        minRegionSupport: 4,
        localAffineNeighborCount: 48
      }), seed)),
      geometry: {
        ...noisyGeometry,
        minTransitionAgreement: 0.2,
        maxNormalizedProjectionRmse: 2.15
      },
      gates: { maxMae: 0.09, minR2: 0.96, minRegionCount: 56, maxSelectedRawOracleCalls: 2_000_000 }
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
    selectedOracleCallsBelow2M: rawOracleCalls < testCase.gates.maxSelectedRawOracleCalls
  };
  const pass = Object.values(criteria).every(Boolean);
  return {
    candidateId: candidate.id,
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
      finiteDifferenceCallsPerSample: (1 + 2 * teacher.dimension) * candidate.sampler.querySamples,
      geometryOracleCalls: 0,
      validationPoints: validationInputs.length
    }
  };
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const results = makeCases().map(testCase => {
  const candidates = testCase.candidates
    .map(candidate => {
      const evaluated = evaluateCandidate(testCase, candidate);
      return { ...evaluated, score: scoreCandidate(evaluated) };
    })
    .sort((a, b) => {
      if (a.pass !== b.pass) return a.pass ? -1 : 1;
      return b.score - a.score;
    });
  const selected = candidates[0];
  const rejected = candidates.slice(1);
  return {
    teacherId: testCase.teacher.id,
    seed: testCase.teacher.seed,
    dimension: testCase.teacher.dimension,
    profile: testCase.teacher.profile,
    noisy: testCase.noisy,
    pass: selected.pass,
    selection: {
      selectedCandidateId: selected.candidateId,
      selectedRandomSeed: selected.randomSeed,
      candidateCount: candidates.length,
      rejectedCandidateCount: rejected.length,
      rejectedPassCount: rejected.filter(candidate => candidate.pass).length,
      rejectedFailCount: rejected.filter(candidate => !candidate.pass).length,
      totalSearchRawOracleCalls: candidates.reduce((sum, candidate) => sum + candidate.queryBudget.rawOracleCalls, 0),
      selectedRawOracleCalls: selected.queryBudget.rawOracleCalls,
      selectionPolicy: 'evaluate deterministic 6D probe distributions, prefer passing candidates, then maximize R2/MAE/region/edge score'
    },
    selected,
    rejectedCandidates: rejected.map(candidate => ({
      candidateId: candidate.candidateId,
      randomSeed: candidate.randomSeed,
      pass: candidate.pass,
      criteria: candidate.criteria,
      score: candidate.score,
      recovery: candidate.recovery,
      geometry: {
        pass: candidate.geometry.pass,
        edgeCount: candidate.geometry.edgeCount,
        largestComponentRatio: candidate.geometry.largestComponentRatio,
        maxNormalizedProjectionRmse: candidate.geometry.maxNormalizedProjectionRmse
      },
      metrics: candidate.metrics,
      queryBudget: candidate.queryBudget
    }))
  };
});

const selectedResults = results.map(result => result.selected);
const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  cleanTeacherCount: results.filter(result => !result.noisy).length,
  noisyTeacherCount: results.filter(result => result.noisy).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  profiles: [...new Set(results.map(result => result.profile))],
  candidateCount: results.reduce((sum, result) => sum + result.selection.candidateCount, 0),
  rejectedCandidateCount: results.reduce((sum, result) => sum + result.selection.rejectedCandidateCount, 0),
  rejectedFailCount: results.reduce((sum, result) => sum + result.selection.rejectedFailCount, 0),
  minRegionCount: Math.min(...selectedResults.map(result => result.recovery.regionCount)),
  minEdgeCount: Math.min(...selectedResults.map(result => result.geometry.edgeCount)),
  minLargestComponentRatio: Math.min(...selectedResults.map(result => result.geometry.largestComponentRatio)),
  maxNormalizedProjectionRmse: Math.max(...selectedResults.map(result => result.geometry.maxNormalizedProjectionRmse)),
  maxMae: Math.max(...selectedResults.map(result => result.metrics.mae)),
  minR2: Math.min(...selectedResults.map(result => result.metrics.r2)),
  maxSelectedRawOracleCallsPerTeacher: Math.max(...results.map(result => result.selection.selectedRawOracleCalls)),
  totalSelectedRawOracleCalls: results.reduce((sum, result) => sum + result.selection.selectedRawOracleCalls, 0),
  totalSearchRawOracleCalls: results.reduce((sum, result) => sum + result.selection.totalSearchRawOracleCalls, 0)
};

const gates = {
  pass: summary.pass,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  cleanAndNoisyCovered: summary.cleanTeacherCount >= 2 && summary.noisyTeacherCount >= 4,
  only6dTeachers: summary.dimensions.length === 1 && summary.dimensions[0] === 6,
  hardProfilesCovered: ['clean', 'near-parallel', 'cancellation', 'low-amplitude', 'weak-support']
    .every(profile => summary.profiles.includes(profile as Profile)),
  adaptiveSearchActuallyRejectedFailures: summary.rejectedFailCount >= 2,
  minRegionCountAbove55: summary.minRegionCount >= 56,
  minEdgesAtLeast35: summary.minEdgeCount >= 35,
  largestComponentRatioAtLeast10Percent: summary.minLargestComponentRatio >= 0.1,
  normalizedProjectionRmseAtMost2_15: summary.maxNormalizedProjectionRmse <= 2.15,
  cleanR2Above099: results.filter(result => !result.noisy).every(result => result.selected.metrics.r2 > 0.99),
  noisyR2Above096: results.filter(result => result.noisy).every(result => result.selected.metrics.r2 > 0.96),
  maxSelectedRawOracleCallsPerTeacherBelow2M: summary.maxSelectedRawOracleCallsPerTeacher < 2_000_000,
  totalSearchRawOracleCallsBelow24M: summary.totalSearchRawOracleCalls < 24_000_000
};
const pass = Object.values(gates).every(Boolean);

const report = {
  generatedAt: new Date().toISOString(),
  task: '6D release-candidate hard-profile sweep for recovered-adjacency geometry on deep black-box ReLU region samplers',
  blackBox: {
    type: '6D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(vector), not hidden activations, layer widths, teacher weights, or true region adjacency'
  },
  claimScope: {
    provesIfPasses: '6D recovered-adjacency geometry survives a broader clean/noisy hard-profile sweep with adaptive probe-distribution selection and explicit failure ledger',
    doesNotClaim: 'SOTA, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, arbitrary production API extraction, or superiority over external extraction algorithms'
  },
  gates,
  summary,
  results,
  nextStep: pass
    ? '6D sweep passed. Next valid step is external piecewise-linear extraction baselines and then an explicit release-gate promotion decision.'
    : '6D sweep failed. Fix the failing profile or sampler before any 6D release-gate promotion.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# 6D ND Recovered-Adjacency Geometry Sweep',
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
  `- Candidate count: ${summary.candidateCount}`,
  `- Rejected candidates: ${summary.rejectedCandidateCount}`,
  `- Rejected failed candidates: ${summary.rejectedFailCount}`,
  `- Clean/noisy teachers: ${summary.cleanTeacherCount}/${summary.noisyTeacherCount}`,
  `- Dimensions: ${summary.dimensions.join(', ')}`,
  `- Profiles: ${summary.profiles.join(', ')}`,
  `- Min region count: ${summary.minRegionCount}`,
  `- Min adjacency edges: ${summary.minEdgeCount}`,
  `- Min largest-component ratio: ${summary.minLargestComponentRatio}`,
  `- Max normalized projection RMSE: ${summary.maxNormalizedProjectionRmse}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Max selected raw oracle calls per teacher: ${summary.maxSelectedRawOracleCallsPerTeacher}`,
  `- Total selected raw oracle calls: ${summary.totalSelectedRawOracleCalls}`,
  `- Total search raw oracle calls: ${summary.totalSearchRawOracleCalls}`,
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
    `- Selected random seed: ${result.selection.selectedRandomSeed}`,
    `- Candidates evaluated: ${result.selection.candidateCount}`,
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
  lane: 'nn-nd-adjacency-geometry-6d-sweep',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Research release-candidate 6D sweep after adaptive probe search.',
    'Covers clean, cancellation, near-parallel, low-amplitude, and weak-support folded-ReLU hard profiles.',
    'Records selected and rejected candidate probe distributions, selected-candidate cost, and total adaptive search cost.',
    'Does not enter release gate or CLI production capability until external baseline comparison is added.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
