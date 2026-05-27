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

type Profile = 'clean' | 'near-parallel' | 'cancellation' | 'low-amplitude';
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
  };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler',
  '3d-4d-clean-noisy'
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

function makeTeacher(seed: number, dimension: number, profile: Profile): Teacher {
  const random = makeRandom(seed);
  const directions: number[][] = [];
  if (profile === 'near-parallel') {
    const base = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
    for (let index = 0; index < 4; index++) {
      const perturbation = normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
        .map(value => value * (0.04 + index * 0.035));
      directions.push(normalize(base.map((value, axis) => value + perturbation[axis])));
    }
  } else {
    for (let index = 0; index < 4; index++) {
      directions.push(normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1))));
    }
  }

  const l1 = directions.map(direction => ({
    w: direction.map(value => value * randomBetween(
      random,
      profile === 'low-amplitude' ? 0.45 : 0.6,
      profile === 'low-amplitude' ? 0.9 : 1.45
    )),
    b: randomBetween(random, -0.75, 0.75)
  }));
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      const amplitude = profile === 'cancellation'
        ? randomBetween(random, 0.45, 1.05)
        : randomBetween(random, 0.2, 1.0);
      return sign * amplitude;
    }),
    b: randomBetween(random, -0.55, 0.55)
  }));
  const outputScale = profile === 'low-amplitude' ? 0.65 : 1;
  return {
    id: `${profile}-${dimension}d-${seed}`,
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
  return [
    {
      teacher: makeTeacher(29, 3, 'clean'),
      noisy: false,
      sampler: {
        sampleCount: 6500,
        querySamples: 1,
        gradientStep: 1e-5,
        slopeTolerance: 1e-5,
        interceptTolerance: 1e-5,
        minRegionSupport: 6,
        neighborCount: 11,
        predictionMode: 'region-vote'
      },
      geometry: {
        assignmentSampleLimit: 1800,
        neighborCount: 8,
        minContactCount: 3,
        minWitnessCount: 2,
        minTransitionAgreement: 0.5,
        maxNormalizedProjectionRmse: 0.75,
        minPassingEdges: 18,
        minLargestComponentRatio: 0.25
      },
      gates: { maxMae: 0.012, minR2: 0.999 }
    },
    {
      teacher: makeTeacher(29, 4, 'clean'),
      noisy: false,
      sampler: {
        sampleCount: 10000,
        querySamples: 1,
        gradientStep: 1e-5,
        slopeTolerance: 1e-5,
        interceptTolerance: 1e-5,
        minRegionSupport: 6,
        neighborCount: 15,
        predictionMode: 'region-vote'
      },
      geometry: {
        assignmentSampleLimit: 2200,
        neighborCount: 10,
        minContactCount: 3,
        minWitnessCount: 2,
        minTransitionAgreement: 0.45,
        maxNormalizedProjectionRmse: 0.9,
        minPassingEdges: 20,
        minLargestComponentRatio: 0.18
      },
      gates: { maxMae: 0.02, minR2: 0.998 }
    },
    {
      teacher: makeTeacher(634, 3, 'low-amplitude'),
      noisy: true,
      sampler: {
        sampleCount: 3600,
        querySamples: 15,
        gradientStep: 0.085,
        slopeTolerance: 0.22,
        interceptTolerance: 0.44,
        minRegionSupport: 6,
        neighborCount: 15,
        predictionMode: 'local-affine-knn',
        localAffineNeighborCount: 64,
        aggregation: 'median'
      },
      geometry: {
        assignmentSampleLimit: 1800,
        neighborCount: 12,
        minContactCount: 4,
        minWitnessCount: 2,
        minTransitionAgreement: 0.35,
        maxNormalizedProjectionRmse: 1.25,
        minPassingEdges: 8,
        minLargestComponentRatio: 0.12
      },
      gates: { maxMae: 0.035, minR2: 0.995 }
    },
    {
      teacher: makeTeacher(717, 4, 'cancellation'),
      noisy: true,
      sampler: {
        sampleCount: 3600,
        querySamples: 9,
        gradientStep: 0.09,
        slopeTolerance: 0.24,
        interceptTolerance: 0.48,
        minRegionSupport: 5,
        neighborCount: 15,
        predictionMode: 'region-vote',
        localAffineNeighborCount: 60,
        aggregation: 'median'
      },
      geometry: {
        assignmentSampleLimit: 2200,
        neighborCount: 14,
        minContactCount: 4,
        minWitnessCount: 2,
        minTransitionAgreement: 0.35,
        maxNormalizedProjectionRmse: 1.35,
        minPassingEdges: 8,
        minLargestComponentRatio: 0.1
      },
      gates: { maxMae: 0.055, minR2: 0.985 }
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
  const validationInputs = makeInputs(310000 + teacher.seed + teacher.dimension, teacher.dimension, 1800);
  const actual = validationInputs.map(input => predictTeacher(teacher, input));
  const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, validationInputs);
  const metrics = regressionMetrics(actual, predicted);
  const pass = geometry.pass
    && geometry.edgeCount >= testCase.geometry.minPassingEdges
    && geometry.largestComponentRatio >= testCase.geometry.minLargestComponentRatio
    && geometry.maxNormalizedProjectionRmse <= testCase.geometry.maxNormalizedProjectionRmse
    && metrics.mae < testCase.gates.maxMae
    && metrics.r2 > testCase.gates.minR2;

  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    dimension: teacher.dimension,
    profile: teacher.profile,
    noisy: testCase.noisy,
    pass,
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
  minEdgeCount: Math.min(...results.map(result => result.geometry.edgeCount)),
  minLargestComponentRatio: Math.min(...results.map(result => result.geometry.largestComponentRatio)),
  maxNormalizedProjectionRmse: Math.max(...results.map(result => result.geometry.maxNormalizedProjectionRmse)),
  maxMae: Math.max(...results.map(result => result.metrics.mae)),
  minR2: Math.min(...results.map(result => result.metrics.r2)),
  totalRawOracleCalls: results.reduce((sum, result) => sum + result.queryBudget.rawOracleCalls, 0)
};

const gates = {
  pass: summary.pass,
  allTeachersPassed: summary.passedTeachers === summary.teacherCount,
  cleanAndNoisyCovered: summary.cleanTeacherCount >= 2 && summary.noisyTeacherCount >= 2,
  dimensionsCovered: summary.dimensions.includes(3) && summary.dimensions.includes(4),
  minEdgesAtLeast8: summary.minEdgeCount >= 8,
  largestComponentRatioAtLeast10Percent: summary.minLargestComponentRatio >= 0.1,
  normalizedProjectionRmseAtMost1_35: summary.maxNormalizedProjectionRmse <= 1.35,
  maxMaeBelow0055: summary.maxMae < 0.055,
  minR2Above0985: summary.minR2 > 0.985
};

const report = {
  generatedAt: new Date().toISOString(),
  task: 'ND recovered-adjacency geometry for deep black-box ReLU region samplers',
  blackBox: {
    type: '3D/4D input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(vector), not hidden activations, layer widths, teacher weights, or true region adjacency'
  },
  claimScope: {
    proves: 'benchmark-scoped ND adjacency geometry recovered from affine-region assignments using cross-region contact witnesses and affine-equality hyperplanes',
    doesNotClaim: 'topology-perfect full ND cell complex recovery, unique hidden-layer identity, exact layer attribution, arbitrary production API extraction, or globally optimal query complexity'
  },
  gates,
  summary,
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# ND Recovered-Adjacency Geometry for Deep Black-Box ReLU Region Samplers',
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
  `- Min adjacency edges: ${summary.minEdgeCount}`,
  `- Min largest-component ratio: ${summary.minLargestComponentRatio}`,
  `- Max normalized projection RMSE: ${summary.maxNormalizedProjectionRmse}`,
  `- Max MAE: ${summary.maxMae}`,
  `- Min R2: ${summary.minR2}`,
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
].join('\n');

const pass = Object.values(gates).every(Boolean);
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, `${markdown}\n`);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler',
  status: pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Phase-4 ND recovered-adjacency geometry gate.',
    'Geometry is built after decompilation from affine-region assignments only; it uses no hidden teacher state and no extra oracle calls.',
    'Passing edges require cross-region contact witnesses and bounded affine-equality projection residuals.'
  ]
});

console.log(JSON.stringify({
  pass,
  report: path.relative(repoRoot, output.jsonOutputPath),
  summary,
  gates
}, null, 2));

if (!pass) process.exit(1);
