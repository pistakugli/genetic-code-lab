#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxRelu2DRegionGraph,
  decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined,
  predictDeepBlackBoxRelu2DRegionGraphBatch,
  predictDeepBlackBoxRelu2DRegionGraphCoarseModel
} from '../src/solvers/deep-blackbox-relu2d-region-graph';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type Layer1Unit = { w: [number, number]; b: number };
type LayerUnit = { w: number[]; b: number };
type Teacher = {
  id: string;
  seed: number;
  l1: Layer1Unit[];
  l2: LayerUnit[];
  out: { w: number[]; b: number; base: [number, number] };
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-boundary-refined-deep-blackbox-relu2d-region-graph', 'fixed-plus-randomized');

function relu(value: number): number {
  return Math.max(0, value);
}

function makeRandom(seed: number): () => number {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function norm2(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}

function makeRandomDeepTeacher(seed: number): Teacher {
  const random = makeRandom(seed);
  const l1: Layer1Unit[] = [];
  for (let index = 0; index < 4; index++) {
    for (let attempt = 0; attempt < 1000; attempt++) {
      const angle = randomBetween(random, -Math.PI, Math.PI);
      const scale = randomBetween(random, 0.65, 1.55);
      const normal: [number, number] = [Math.cos(angle) * scale, Math.sin(angle) * scale];
      const separated = l1.every(unit => {
        const dot = (unit.w[0] * normal[0] + unit.w[1] * normal[1])
          / Math.max(1e-12, norm2(unit.w[0], unit.w[1]) * norm2(normal[0], normal[1]));
        return Math.abs(dot) < 0.94;
      });
      if (!separated) continue;
      l1.push({ w: normal, b: randomBetween(random, -0.85, 0.85) });
      break;
    }
  }
  const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
    w: Array.from({ length: 4 }, (_, weightIndex) => {
      const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
      return sign * randomBetween(random, 0.18, 1.25);
    }),
    b: randomBetween(random, -0.75, 0.75)
  }));
  return {
    id: `random-deep-2d-${seed}`,
    seed,
    l1,
    l2,
    out: {
      w: Array.from({ length: 4 }, (_, index) => {
        const sign = index % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.32, 1.2);
      }),
      b: randomBetween(random, -0.18, 0.18),
      base: [randomBetween(random, -0.09, 0.09), randomBetween(random, -0.09, 0.09)]
    }
  };
}

function predictTeacher(teacher: Teacher, x: number, y: number): number {
  const h1 = teacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b));
  const h2 = teacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
  return teacher.out.b
    + teacher.out.base[0] * x
    + teacher.out.base[1] * y
    + teacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
}

function makeGrid(width: number, height: number, xOffset: number, yOffset: number): Array<[number, number]> {
  const inputs: Array<[number, number]> = [];
  for (let xIndex = 0; xIndex < width; xIndex++) {
    for (let yIndex = 0; yIndex < height; yIndex++) {
      inputs.push([
        -1.95 + (3.9 * (xIndex + xOffset)) / width,
        -1.95 + (3.9 * (yIndex + yOffset)) / height
      ]);
    }
  }
  return inputs;
}

const fixedTeacher: Teacher = {
  id: 'fixed-deep-2d-1',
  seed: 1,
  l1: [
    { w: [-1.4999765222891948, -1.105386636783729], b: 0.6134527729949475 },
    { w: [-1.358866152687805, 0.5365941501563363], b: 0.43031137364945504 },
    { w: [1.3040786877312498, -0.34949376839165924], b: 0.04659929242599681 },
    { w: [0.992896038100958, -1.3962836697663028], b: -1.0716920769509786 }
  ],
  l2: [
    { w: [0.08316054072525403, 0.479218274987506, -1.3784450799026013, -0.3264361786902288], b: -0.7796839732487537 },
    { w: [-0.23103927227765264, 0.5229635942009869, 0.24913459946320815, 1.2052221851472016], b: 0.6231004027864898 },
    { w: [0.07540057662446098, -1.142498307062777, 0.43097309398555494, -0.2352018015787023], b: 0.36214306974983124 },
    { w: [1.1488983252541147, 0.7341545108092524, -0.6651316187019753, -1.2670993628605263], b: 0.4249473920324326 }
  ],
  out: {
    w: [-0.4122378580386171, 0.3183325625195472, 0.615385166011178, 1.1784897252716957],
    b: -0.053864531827964385,
    base: [-0.05059222295004151, 0.0965100572411996]
  }
};

fs.mkdirSync(output.outputDir, { recursive: true });
const started = Date.now();
const teachers = [fixedTeacher, makeRandomDeepTeacher(29), makeRandomDeepTeacher(83)];
const inputs = makeGrid(91, 83, 0.37, 0.23);

const results = teachers.map(teacher => {
  let fixedOracleCalls = 0;
  const fixedOracle = (x: number, y: number): number => {
    fixedOracleCalls++;
    return predictTeacher(teacher, x, y);
  };
  const fixedModel = decompileDeepBlackBoxRelu2DRegionGraph(fixedOracle, {
    xRange: [-2, 2],
    yRange: [-2, 2],
    gridWidth: 101,
    gridHeight: 101,
    gradientStep: 1e-5,
    slopeTolerance: 1e-5,
    interceptTolerance: 1e-5
  });

  let refinedOracleCalls = 0;
  const refinedOracle = (x: number, y: number): number => {
    refinedOracleCalls++;
    return predictTeacher(teacher, x, y);
  };
  const refinedModel = decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined(
    refinedOracle,
    {
      xRange: [-2, 2],
      yRange: [-2, 2],
      gradientStep: 1e-5,
      slopeTolerance: 1e-5,
      interceptTolerance: 1e-5
    },
    {
      gridWidth: 57,
      gridHeight: 57,
      marginFactor: 0.5,
      validationGridWidth: 37,
      validationGridHeight: 31,
      maxRefinedValidationMae: 0.0003,
      minRefinedValidationR2: 0.99999
    }
  );

  const refinement = refinedModel.boundaryRefinement;
  if (!refinement) throw new Error('boundary-refined model missing boundaryRefinement report');

  const actual = inputs.map(([x, y]) => predictTeacher(teacher, x, y));
  const fixedPredicted = predictDeepBlackBoxRelu2DRegionGraphBatch(fixedModel, inputs);
  const coarsePredicted = inputs.map(([x, y]) => predictDeepBlackBoxRelu2DRegionGraphCoarseModel(refinedModel, x, y));
  const refinedPredicted = predictDeepBlackBoxRelu2DRegionGraphBatch(refinedModel, inputs);
  const fixedMetrics = regressionMetrics(actual, fixedPredicted);
  const coarseMetrics = regressionMetrics(actual, coarsePredicted);
  const refinedMetrics = regressionMetrics(actual, refinedPredicted);
  const oracleReductionVsFixed = 1 - refinement.totalOracleCalls / Math.max(1, fixedOracleCalls);
  const heldOutMaeImprovementRatio = coarseMetrics.mae / Math.max(1e-12, refinedMetrics.mae);
  const pass = refinement.pass
    && refinedOracleCalls === refinement.totalOracleCalls
    && oracleReductionVsFixed > 0.6
    && refinedMetrics.mae < 0.0003
    && refinedMetrics.r2 > 0.99999
    && heldOutMaeImprovementRatio > 5
    && refinedMetrics.mae <= Math.max(0.0003, fixedMetrics.mae * 1.25)
    && refinedModel.compactBoundarySegmentCount === refinedModel.adjacency.length;

  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    pass,
    fixed101: {
      gridWidth: fixedModel.gridWidth,
      gridHeight: fixedModel.gridHeight,
      rawOracleCalls: fixedOracleCalls,
      regionCount: fixedModel.regionCount,
      adjacencyEdgeCount: fixedModel.adjacency.length,
      compactBoundarySegmentCount: fixedModel.compactBoundarySegmentCount,
      metrics: fixedMetrics
    },
    coarse57: {
      gridWidth: refinedModel.gridWidth,
      gridHeight: refinedModel.gridHeight,
      regionCount: refinedModel.regionCount,
      adjacencyEdgeCount: refinedModel.adjacency.length,
      compactBoundarySegmentCount: refinedModel.compactBoundarySegmentCount,
      metrics: coarseMetrics
    },
    boundaryRefined57: {
      totalOracleCalls: refinement.totalOracleCalls,
      decompileOracleCalls: refinement.decompileOracleCalls,
      validationOracleCalls: refinement.validationOracleCalls,
      oracleReductionVsFixed,
      boundaryMargin: refinement.boundaryMargin,
      marginFactor: refinement.marginFactor,
      projectionSlack: refinement.projectionSlack,
      validationMae: refinement.refinedValidationMae,
      validationR2: refinement.refinedValidationR2,
      validationMaeImprovementRatio: refinement.maeImprovementRatio,
      heldOutMaeImprovementRatio,
      metrics: refinedMetrics,
      report: refinement
    }
  };
});

const summary = {
  pass: results.every(result => result.pass),
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  minOracleReductionVsFixed: Math.min(...results.map(result => result.boundaryRefined57.oracleReductionVsFixed)),
  totalFixed101OracleCalls: results.reduce((sum, result) => sum + result.fixed101.rawOracleCalls, 0),
  totalBoundaryRefinedOracleCalls: results.reduce((sum, result) => sum + result.boundaryRefined57.totalOracleCalls, 0),
  maxBoundaryRefinedMae: Math.max(...results.map(result => result.boundaryRefined57.metrics.mae)),
  minBoundaryRefinedR2: Math.min(...results.map(result => result.boundaryRefined57.metrics.r2)),
  minHeldOutMaeImprovementRatio: Math.min(...results.map(result => result.boundaryRefined57.heldOutMaeImprovementRatio)),
  maxBoundaryRefinedToFixedMaeRatio: Math.max(...results.map(result => (
    result.boundaryRefined57.metrics.mae / Math.max(1e-12, result.fixed101.metrics.mae)
  )))
};

const report = {
  generatedAt: new Date().toISOString(),
  task: 'boundary-refined local prediction for deep black-box ReLU 2D region graphs',
  blackBox: {
    type: '2-input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'benchmark-scoped boundary-aware local switching that improves a lower-cost 57x57 region graph using compact boundary geometry and held-out validation',
    doesNotClaim: 'unique hidden-layer recovery, exact layer attribution, arbitrary deep ND extraction, stochastic production API extraction, or topology-perfect true-region merging'
  },
  gates: {
    pass: summary.pass,
    allTeachersPassed: summary.passedTeachers === summary.teacherCount,
    oracleReductionAbove60Percent: summary.minOracleReductionVsFixed > 0.6,
    boundaryRefinedMaeBelow0003: summary.maxBoundaryRefinedMae < 0.0003,
    boundaryRefinedR2Above99999: summary.minBoundaryRefinedR2 > 0.99999,
    coarseToRefinedMaeImprovementAbove5x: summary.minHeldOutMaeImprovementRatio > 5
  },
  summary,
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Boundary-Refined Deep Black-Box ReLU 2D Region-Graph Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${summary.pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- GCL access: oracle calls only, `f(x,y)`',
  '- Claim: compact-boundary local switching improves a lower-cost 57x57 output-equivalent region graph',
  '- Not claimed: original hidden-layer identity, exact layer attribution, arbitrary deep ND extraction, or production API extraction',
  '',
  '## Summary',
  '',
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Minimum oracle reduction vs fixed 101x101: ${(summary.minOracleReductionVsFixed * 100).toFixed(3)}%`,
  `- Total fixed 101x101 oracle calls: ${summary.totalFixed101OracleCalls}`,
  `- Total boundary-refined oracle calls: ${summary.totalBoundaryRefinedOracleCalls}`,
  `- Max boundary-refined MAE: ${summary.maxBoundaryRefinedMae.toExponential(6)}`,
  `- Min boundary-refined R2: ${summary.minBoundaryRefinedR2.toFixed(12)}`,
  `- Minimum held-out coarse-to-refined MAE improvement: ${summary.minHeldOutMaeImprovementRatio.toFixed(6)}x`,
  '',
  '## Results',
  '',
  '| Teacher | Pass | Fixed Calls | Refined Calls | Reduction | Coarse MAE | Refined MAE | Improvement | Refined R2 |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
  ...results.map(result => `| ${result.teacherId} | ${result.pass ? 'yes' : 'no'} | ${result.fixed101.rawOracleCalls} | ${result.boundaryRefined57.totalOracleCalls} | ${(result.boundaryRefined57.oracleReductionVsFixed * 100).toFixed(3)}% | ${result.coarse57.metrics.mae.toExponential(6)} | ${result.boundaryRefined57.metrics.mae.toExponential(6)} | ${result.boundaryRefined57.heldOutMaeImprovementRatio.toFixed(6)}x | ${result.boundaryRefined57.metrics.r2.toFixed(12)} |`),
  ''
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, markdown);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-boundary-refined-deep-blackbox-relu2d-region-graph',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'This benchmark promotes compact boundary geometry from a passive artifact into an active local prediction mechanism.',
    'The solver uses a lower 57x57 probe grid plus held-out validation, then switches regions near compact boundaries instead of falling back to a full 101x101 grid.',
    'It still proves output-equivalent behavior for this benchmark family, not hidden-layer identity.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!summary.pass) process.exit(1);
