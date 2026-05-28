#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerBatch
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { buildDeepBlackBoxReluNdAdjacencyGeometry } from '../src/solvers/deep-blackbox-relu-nd-adjacency-geometry';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

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
type TeacherCase = {
  teacher: Teacher;
  noisy: boolean;
  sampler: {
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
  };
  geometry: {
    assignmentSampleLimit: number;
    neighborCount: number;
    minContactCount: number;
    minWitnessCount: number;
    minTransitionAgreement: number;
    maxNormalizedProjectionRmse: number;
    minPassingEdges: number;
    minLargestComponentRatio: number;
  };
  gates: {
    maxMae: number;
    minR2: number;
    minRegionCount: number;
    maxRawOracleCalls: number;
  };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-nd-adjacency-geometry-5d-sweep',
  'clean-noisy-hard-profiles'
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
        .map(value => value * (0.03 + index * 0.03));
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
  const firstLayerScale = profile === 'low-amplitude' ? [0.45, 0.9] : [0.6, 1.45];
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
        ? randomBetween(random, 0.45, 1.05)
        : randomBetween(random, 0.2, 1.0);
      return sign * amplitude * (profile === 'weak-support' && weightIndex === 3 ? 0.45 : 1);
    }),
    b: profile === 'weak-support' && unitIndex === 3
      ? randomBetween(random, 0.6, 1.0)
      : randomBetween(random, -0.55, 0.55)
  }));
  const outputScale = profile === 'low-amplitude' ? 0.65 : 1;
  return {
    id: `${profile}-5d-${seed}`,
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
  const random = makeRandom(91000 + teacher.seed + teacher.dimension);
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

function makeCases(): TeacherCase[] {
  const cleanSampler = {
    sampleCount: 18000,
    querySamples: 1,
    gradientStep: 1e-5,
    slopeTolerance: 1e-5,
    interceptTolerance: 1e-5,
    minRegionSupport: 6,
    neighborCount: 21,
    predictionMode: 'region-vote' as const
  };
  const cleanGeometry = {
    assignmentSampleLimit: 3600,
    neighborCount: 14,
    minContactCount: 3,
    minWitnessCount: 2,
    minTransitionAgreement: 0.4,
    maxNormalizedProjectionRmse: 1.25,
    minPassingEdges: 30,
    minLargestComponentRatio: 0.12
  };
  const noisySampler = {
    sampleCount: 9000,
    querySamples: 9,
    gradientStep: 0.085,
    slopeTolerance: 0.18,
    interceptTolerance: 0.36,
    minRegionSupport: 7,
    neighborCount: 23,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 96,
    aggregation: 'median' as const
  };
  const noisyGeometry = {
    assignmentSampleLimit: 3600,
    neighborCount: 18,
    minContactCount: 4,
    minWitnessCount: 2,
    minTransitionAgreement: 0.25,
    maxNormalizedProjectionRmse: 1.75,
    minPassingEdges: 30,
    minLargestComponentRatio: 0.12
  };
  return [
    {
      teacher: makeTeacher(29, 5, 'clean'),
      noisy: false,
      sampler: cleanSampler,
      geometry: cleanGeometry,
      gates: { maxMae: 0.035, minR2: 0.99, minRegionCount: 51, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(83, 5, 'clean'),
      noisy: false,
      sampler: cleanSampler,
      geometry: cleanGeometry,
      gates: { maxMae: 0.04, minR2: 0.99, minRegionCount: 51, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(634, 5, 'low-amplitude'),
      noisy: true,
      sampler: {
        ...noisySampler,
        sampleCount: 12000,
        slopeTolerance: 0.08,
        interceptTolerance: 0.2,
        minRegionSupport: 5,
        localAffineNeighborCount: 112
      },
      geometry: noisyGeometry,
      gates: { maxMae: 0.07, minR2: 0.97, minRegionCount: 51, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(717, 5, 'cancellation'),
      noisy: true,
      sampler: {
        ...noisySampler,
        sampleCount: 12000,
        slopeTolerance: 0.14,
        interceptTolerance: 0.3,
        localAffineNeighborCount: 128
      },
      geometry: noisyGeometry,
      gates: { maxMae: 0.08, minR2: 0.97, minRegionCount: 51, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(805, 5, 'near-parallel'),
      noisy: true,
      sampler: {
        ...noisySampler,
        sampleCount: 12000,
        slopeTolerance: 0.06,
        interceptTolerance: 0.18,
        minRegionSupport: 5,
        localAffineNeighborCount: 112
      },
      geometry: noisyGeometry,
      gates: { maxMae: 0.08, minR2: 0.97, minRegionCount: 51, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(911, 5, 'weak-support'),
      noisy: true,
      sampler: {
        ...noisySampler,
        sampleCount: 11000,
        querySamples: 9,
        localAffineNeighborCount: 112
      },
      geometry: {
        ...noisyGeometry,
        minTransitionAgreement: 0.2,
        maxNormalizedProjectionRmse: 1.95
      },
      gates: { maxMae: 0.09, minR2: 0.97, minRegionCount: 51, maxRawOracleCalls: 2_000_000 }
    }
  ];
}

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();

const results = makeCases().map(testCase => {
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
    sampleCount: testCase.sampler.sampleCount,
    gradientStep: testCase.sampler.gradientStep,
    slopeTolerance: testCase.sampler.slopeTolerance,
    interceptTolerance: testCase.sampler.interceptTolerance,
    minRegionSupport: testCase.sampler.minRegionSupport,
    querySamples: testCase.sampler.querySamples,
    aggregation: testCase.sampler.aggregation ?? 'mean',
    neighborCount: testCase.sampler.neighborCount,
    predictionMode: testCase.sampler.predictionMode,
    localAffineNeighborCount: testCase.sampler.localAffineNeighborCount,
    randomSeed: 260526 + teacher.seed + teacher.dimension,
    ridge: 1e-8
  });
  const geometry = buildDeepBlackBoxReluNdAdjacencyGeometry(model, testCase.geometry);
  const validationInputs = makeInputs(310000 + teacher.seed + teacher.dimension, teacher.dimension, 2400);
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
    regionCountAbove50: model.regionCount >= testCase.gates.minRegionCount,
    oracleCallsBelow2M: rawOracleCalls < testCase.gates.maxRawOracleCalls
  };
  const pass = Object.values(criteria).every(Boolean);

  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    dimension: teacher.dimension,
    profile: teacher.profile,
    noisy: testCase.noisy,
    pass,
    criteria,
    gates: testCase.gates,
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
      geometryOracleCalls: 0,
      validationPoints: validationInputs.length
    }
  };
});

const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  cleanTeacherCount: results.filter(result => !result.noisy).length,
  noisyTeacherCount: results.filter(result => result.noisy).length,
  dimensions: [...new Set(results.map(result => result.dimension))],
  profiles: [...new Set(results.map(result => result.profile))],
  minRegionCount: Math.min(...results.map(result => result.recovery.regionCount)),
  minEdgeCount: Math.min(...results.map(result => result.geometry.edgeCount)),
  minLargestComponentRatio: Math.min(...results.map(result => result.geometry.largestComponentRatio)),
  maxNormalizedProjectionRmse: Math.max(...results.map(result => result.geometry.maxNormalizedProjectionRmse)),
  maxMae: Math.max(...results.map(result => result.metrics.mae)),
  minR2: Math.min(...results.map(result => result.metrics.r2)),
  maxRawOracleCallsPerTeacher: Math.max(...results.map(result => result.queryBudget.rawOracleCalls)),
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0)
};

const gates = {
  pass: summary.pass,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  cleanAndNoisyCovered: summary.cleanTeacherCount >= 2 && summary.noisyTeacherCount >= 4,
  only5dTeachers: summary.dimensions.length === 1 && summary.dimensions[0] === 5,
  hardProfilesCovered: ['clean', 'near-parallel', 'cancellation', 'low-amplitude', 'weak-support']
    .every(profile => summary.profiles.includes(profile as Profile)),
  minRegionCountAbove50: summary.minRegionCount >= 51,
  minEdgesAtLeast30: summary.minEdgeCount >= 30,
  largestComponentRatioAtLeast12Percent: summary.minLargestComponentRatio >= 0.12,
  normalizedProjectionRmseAtMost1_95: summary.maxNormalizedProjectionRmse <= 1.95,
  cleanR2Above099: results.filter(result => !result.noisy).every(result => result.metrics.r2 > 0.99),
  noisyR2Above097: results.filter(result => result.noisy).every(result => result.metrics.r2 > 0.97),
  maxRawOracleCallsPerTeacherBelow2M: summary.maxRawOracleCallsPerTeacher < 2_000_000
};

const report = {
  generatedAt: new Date().toISOString(),
  task: '5D clean/noisy recovered-adjacency geometry sweep for deep black-box ReLU region samplers',
  blackBox: {
    type: '5D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(vector), not hidden activations, layer widths, teacher weights, or true region adjacency'
  },
  claimScope: {
    proves: 'benchmark-scoped 5D clean/noisy ND adjacency geometry recovered from affine-region assignments across multiple deterministic folded-ReLU hard profiles',
    doesNotClaim: '6D readiness, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, arbitrary production API extraction, or superiority over all model-extraction algorithms'
  },
  gates,
  summary,
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# 5D ND Recovered-Adjacency Geometry Sweep',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves: ${report.claimScope.proves}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  '',
  '## Summary',
  '',
  `- Pass: ${Object.values(gates).every(Boolean)}`,
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Clean/noisy teachers: ${summary.cleanTeacherCount}/${summary.noisyTeacherCount}`,
  `- Dimensions: ${summary.dimensions.join(', ')}`,
  `- Profiles: ${summary.profiles.join(', ')}`,
  `- Min region count: ${summary.minRegionCount}`,
  `- Min adjacency edges: ${summary.minEdgeCount}`,
  `- Min largest-component ratio: ${summary.minLargestComponentRatio}`,
  `- Max normalized projection RMSE: ${summary.maxNormalizedProjectionRmse}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
  `- Max raw oracle calls per teacher: ${summary.maxRawOracleCallsPerTeacher}`,
  `- Total raw oracle calls: ${summary.totalRawOracleCalls}`,
  '',
  '## Results',
  '',
  ...results.map(result => [
    `### ${result.teacherId}`,
    '',
    `- Pass: ${result.pass}`,
    `- Dimension: ${result.dimension}`,
    `- Profile: ${result.profile}`,
    `- Noisy oracle: ${result.noisy}`,
    `- Region count: ${result.recovery.regionCount}`,
    `- Deep-only region estimate: ${result.recovery.deepOnlyRegionEstimate}`,
    `- Retained sample ratio: ${result.recovery.retainedSampleRatio}`,
    `- Geometry pass: ${result.geometry.pass}`,
    `- Candidate pairs: ${result.geometry.candidatePairCount}`,
    `- Passing edges: ${result.geometry.edgeCount}`,
    `- Largest-component ratio: ${result.geometry.largestComponentRatio}`,
    `- Max normalized projection RMSE: ${result.geometry.maxNormalizedProjectionRmse}`,
    `- GCL MAE: ${result.metrics.mae}`,
    `- GCL R2: ${result.metrics.r2}`,
    `- Raw oracle calls: ${result.queryBudget.rawOracleCalls}`,
    ''
  ].join('\n'))
].join('\n').trimEnd();

const pass = Object.values(gates).every(Boolean);
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-nd-adjacency-geometry-5d-sweep',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    '5D release-candidate sweep after the single 5D research probe.',
    'Covers clean and noisy quantized folded-ReLU teachers across low-amplitude, cancellation, near-parallel, and weak-support profiles.',
    'Each teacher must keep R2/region-count/query-budget and recovered-adjacency geometry gates above threshold.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
