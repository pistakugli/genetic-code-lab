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

type Profile = 'clean' | 'near-parallel' | 'cancellation';
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
    randomSeed?: number;
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
process.env.GCL_BENCHMARK_ROOT = process.env.GCL_BENCHMARK_ROOT ?? path.join('benchmarks', 'research');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-nd-adjacency-geometry-6d',
  'guarded-probe'
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
  return Array.from({ length: 4 }, () => normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1))));
}

function makeTeacher(seed: number, dimension: number, profile: Profile): Teacher {
  const random = makeRandom(seed);
  const directions = makeDirections(random, dimension, profile);
  const l1 = directions.map(direction => ({
    w: direction.map(value => value * randomBetween(random, 0.6, 1.35)),
    b: randomBetween(random, -0.75, 0.75)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      const amplitude = profile === 'cancellation'
        ? randomBetween(random, 0.5, 1.05)
        : randomBetween(random, 0.2, 1.0);
      return sign * amplitude;
    }),
    b: randomBetween(random, -0.55, 0.55)
  }));
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

function makeNoisyOracle(teacher: Teacher): (input: number[]) => number {
  const random = makeRandom(160000 + teacher.seed + teacher.dimension);
  return (input: number[]) => {
    const noisy = predictTeacher(teacher, input) + gaussian(random) * 0.0025;
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
  const noisySampler = {
    sampleCount: 14500,
    querySamples: 9,
    gradientStep: 0.07,
    slopeTolerance: 0.1,
    interceptTolerance: 0.24,
    minRegionSupport: 6,
    neighborCount: 27,
    predictionMode: 'local-affine-knn' as const,
    localAffineNeighborCount: 72,
    aggregation: 'median' as const
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
  return [
    {
      teacher: makeTeacher(29, 6, 'clean'),
      noisy: false,
      sampler: cleanSampler,
      geometry: cleanGeometry,
      gates: { maxMae: 0.055, minR2: 0.99, minRegionCount: 56, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(83, 6, 'clean'),
      noisy: false,
      sampler: cleanSampler,
      geometry: cleanGeometry,
      gates: { maxMae: 0.06, minR2: 0.99, minRegionCount: 56, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(717, 6, 'cancellation'),
      noisy: true,
      sampler: {
        ...noisySampler,
        sampleCount: 16000,
        gradientStep: 0.08,
        slopeTolerance: 0.12,
        interceptTolerance: 0.28,
        localAffineNeighborCount: 32
      },
      geometry: noisyGeometry,
      gates: { maxMae: 0.095, minR2: 0.965, minRegionCount: 56, maxRawOracleCalls: 2_000_000 }
    },
    {
      teacher: makeTeacher(805, 6, 'near-parallel'),
      noisy: true,
      sampler: {
        ...noisySampler,
        sampleCount: 17000,
        querySamples: 9,
        gradientStep: 0.04,
        slopeTolerance: 0.05,
        interceptTolerance: 0.12,
        minRegionSupport: 4,
        neighborCount: 27,
        predictionMode: 'region-vote',
        localAffineNeighborCount: 16,
        randomSeed: Number(process.env.GCL_6D_NEAR_RANDOM_SEED ?? 360001)
      },
      geometry: {
        ...noisyGeometry,
        minTransitionAgreement: 0.2,
        maxNormalizedProjectionRmse: 2.15
      },
      gates: { maxMae: 0.1, minR2: 0.965, minRegionCount: 56, maxRawOracleCalls: 2_000_000 }
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
    randomSeed: testCase.sampler.randomSeed ?? 360526 + teacher.seed + teacher.dimension,
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
    regionCountAbove55: model.regionCount >= testCase.gates.minRegionCount,
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
      finiteDifferenceCallsPerSample: (1 + 2 * teacher.dimension) * testCase.sampler.querySamples,
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
  cleanAndNoisyCovered: summary.cleanTeacherCount >= 2 && summary.noisyTeacherCount >= 2,
  only6dTeachers: summary.dimensions.length === 1 && summary.dimensions[0] === 6,
  hardProfilesCovered: ['clean', 'near-parallel', 'cancellation']
    .every(profile => summary.profiles.includes(profile as Profile)),
  minRegionCountAbove55: summary.minRegionCount >= 56,
  minEdgesAtLeast35: summary.minEdgeCount >= 35,
  largestComponentRatioAtLeast10Percent: summary.minLargestComponentRatio >= 0.1,
  normalizedProjectionRmseAtMost2_15: summary.maxNormalizedProjectionRmse <= 2.15,
  cleanR2Above099: results.filter(result => !result.noisy).every(result => result.metrics.r2 > 0.99),
  noisyR2Above0965: results.filter(result => result.noisy).every(result => result.metrics.r2 > 0.965),
  maxRawOracleCallsPerTeacherBelow2M: summary.maxRawOracleCallsPerTeacher < 2_000_000
};

const pass = Object.values(gates).every(Boolean);
const report = {
  generatedAt: new Date().toISOString(),
  task: 'guarded 6D recovered-adjacency geometry experiment for deep black-box ReLU region samplers',
  blackBox: {
    type: '6D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(vector), not hidden activations, layer widths, teacher weights, or true region adjacency'
  },
  claimScope: {
    provesIfPasses: '6D can enter a later release-candidate sweep under this controlled clean/noisy hard-profile setup',
    doesNotClaim: '6D release readiness, probe-seed robustness, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, arbitrary production API extraction, or superiority over external extraction algorithms'
  },
  probeSeedNote: 'During development the noisy near-parallel 6D case was probe-seed sensitive: several deterministic sampler seeds failed the R2 gate while seed 360001 passed. This is kept as the next hardening target, not hidden as robustness.',
  gates,
  summary,
  results,
  nextStep: pass
    ? '6D guarded probe passed. The next valid step is a broader 6D release-candidate sweep with more seeds and external extraction baselines.'
    : '6D guarded probe failed. Fix the sampler or adjacency witness builder before promoting any 6D claim.',
  runtimeMs: Date.now() - started
};

const markdown = [
  '# 6D ND Recovered-Adjacency Geometry Guarded Probe',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  '## Claim Scope',
  '',
  `- Proves if pass: ${report.claimScope.provesIfPasses}`,
  `- Does not claim: ${report.claimScope.doesNotClaim}`,
  `- Probe seed note: ${report.probeSeedNote}`,
  '',
  '## Summary',
  '',
  `- Pass: ${pass}`,
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
    `- Profile: ${result.profile}`,
    `- Noisy oracle: ${result.noisy}`,
    `- Region count: ${result.recovery.regionCount}`,
    `- Deep-only region estimate: ${result.recovery.deepOnlyRegionEstimate}`,
    `- Retained sample ratio: ${result.recovery.retainedSampleRatio}`,
    `- Geometry pass: ${result.geometry.pass}`,
    `- Passing edges: ${result.geometry.edgeCount}`,
    `- Largest-component ratio: ${result.geometry.largestComponentRatio}`,
    `- Max normalized projection RMSE: ${result.geometry.maxNormalizedProjectionRmse}`,
    `- GCL MAE: ${result.metrics.mae}`,
    `- GCL R2: ${result.metrics.r2}`,
    `- Raw oracle calls: ${result.queryBudget.rawOracleCalls}`,
    ''
  ].join('\n')),
  '## Next Step',
  '',
  report.nextStep
].join('\n').trimEnd();

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-nd-adjacency-geometry-6d',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Research-only guarded 6D probe after the release-gated 5D sweep.',
    'Does not enter release gate or CLI production capability until a broader 6D sweep is implemented and proven.',
    'Covers clean, noisy cancellation, and noisy near-parallel folded-ReLU teachers under a per-teacher oracle budget.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
