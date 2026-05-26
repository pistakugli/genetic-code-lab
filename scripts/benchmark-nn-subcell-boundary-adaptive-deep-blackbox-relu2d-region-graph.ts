#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  decompileDeepBlackBoxRelu2DRegionGraph,
  decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined,
  decompileDeepBlackBoxRelu2DRegionGraphSubcellBoundaryAdaptive,
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
const output = resolveBenchmarkOutput(repoRoot, 'nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph', 'fixed-plus-randomized');

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

  let boundary57OracleCalls = 0;
  const boundary57Model = decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined(
    (x, y) => {
      boundary57OracleCalls++;
      return predictTeacher(teacher, x, y);
    },
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

  let subcellOracleCalls = 0;
  const subcellModel = decompileDeepBlackBoxRelu2DRegionGraphSubcellBoundaryAdaptive(
    (x, y) => {
      subcellOracleCalls++;
      return predictTeacher(teacher, x, y);
    },
    {
      xRange: [-2, 2],
      yRange: [-2, 2],
      gradientStep: 1e-5,
      slopeTolerance: 1e-5,
      interceptTolerance: 1e-5
    },
    {
      candidateGridSizes: [37, 41, 49, 57],
      marginFactor: 0.5,
      validationGridWidth: 37,
      validationGridHeight: 31,
      maxValidationMae: 0.0003,
      minValidationR2: 0.99998
    }
  );

  const probe = subcellModel.subcellBoundaryProbe;
  if (!probe) throw new Error('subcell model missing subcellBoundaryProbe report');
  const actual = inputs.map(([x, y]) => predictTeacher(teacher, x, y));
  const fixedPredicted = predictDeepBlackBoxRelu2DRegionGraphBatch(fixedModel, inputs);
  const boundary57Predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(boundary57Model, inputs);
  const subcellPredicted = predictDeepBlackBoxRelu2DRegionGraphBatch(subcellModel, inputs);
  const subcellCoarsePredicted = inputs.map(([x, y]) => predictDeepBlackBoxRelu2DRegionGraphCoarseModel(subcellModel, x, y));
  const fixedMetrics = regressionMetrics(actual, fixedPredicted);
  const boundary57Metrics = regressionMetrics(actual, boundary57Predicted);
  const subcellMetrics = regressionMetrics(actual, subcellPredicted);
  const subcellCoarseMetrics = regressionMetrics(actual, subcellCoarsePredicted);
  const selected = probe.selectedAttempt;
  const oracleReductionVsFixed = 1 - probe.totalOracleCalls / Math.max(1, fixedOracleCalls);
  const oracleReductionVsBoundary57 = 1 - probe.totalOracleCalls / Math.max(1, boundary57Model.boundaryRefinement?.totalOracleCalls ?? boundary57OracleCalls);
  const heldOutMaeImprovementRatio = subcellCoarseMetrics.mae / Math.max(1e-12, subcellMetrics.mae);
  const pass = selected.pass
    && subcellOracleCalls === probe.totalOracleCalls
    && oracleReductionVsFixed > 0.6
    && subcellMetrics.mae < 0.0003
    && subcellMetrics.r2 > 0.99999
    && heldOutMaeImprovementRatio > 3
    && selected.gridWidth < 57
    && probe.attempts.every((attempt, index) => index === probe.attempts.length - 1 || !attempt.pass);

  return {
    teacherId: teacher.id,
    seed: teacher.seed,
    pass,
    fixed101: {
      rawOracleCalls: fixedOracleCalls,
      metrics: fixedMetrics
    },
    boundaryRefined57: {
      totalOracleCalls: boundary57Model.boundaryRefinement?.totalOracleCalls ?? boundary57OracleCalls,
      metrics: boundary57Metrics
    },
    subcellAdaptive: {
      selectedGridWidth: selected.gridWidth,
      selectedGridHeight: selected.gridHeight,
      totalOracleCalls: probe.totalOracleCalls,
      selectedAttemptOracleCalls: selected.totalOracleCalls,
      oracleReductionVsFixed,
      oracleReductionVsBoundary57,
      heldOutMaeImprovementRatio,
      attempts: probe.attempts,
      coarseMetrics: subcellCoarseMetrics,
      metrics: subcellMetrics
    }
  };
});

const teacherResultsPassed = results.every(result => result.pass);
const summary = {
  pass: false,
  teacherResultsPassed,
  teacherCount: results.length,
  passedTeachers: results.filter(result => result.pass).length,
  selectedGridWidths: results.map(result => result.subcellAdaptive.selectedGridWidth),
  minOracleReductionVsFixed: Math.min(...results.map(result => result.subcellAdaptive.oracleReductionVsFixed)),
  minOracleReductionVsBoundary57: Math.min(...results.map(result => result.subcellAdaptive.oracleReductionVsBoundary57)),
  totalFixed101OracleCalls: results.reduce((sum, result) => sum + result.fixed101.rawOracleCalls, 0),
  totalBoundary57OracleCalls: results.reduce((sum, result) => sum + result.boundaryRefined57.totalOracleCalls, 0),
  totalSubcellAdaptiveOracleCalls: results.reduce((sum, result) => sum + result.subcellAdaptive.totalOracleCalls, 0),
  maxSubcellAdaptiveMae: Math.max(...results.map(result => result.subcellAdaptive.metrics.mae)),
  minSubcellAdaptiveR2: Math.min(...results.map(result => result.subcellAdaptive.metrics.r2)),
  minHeldOutMaeImprovementRatio: Math.min(...results.map(result => result.subcellAdaptive.heldOutMaeImprovementRatio)),
  rejectedTooCoarseAttemptCount: results.reduce((sum, result) => (
    sum + result.subcellAdaptive.attempts.filter(attempt => !attempt.pass).length
  ), 0)
};
const totalOracleReductionVsFixed = 1 - summary.totalSubcellAdaptiveOracleCalls / Math.max(1, summary.totalFixed101OracleCalls);
const totalOracleReductionVsBoundary57 = 1 - summary.totalSubcellAdaptiveOracleCalls / Math.max(1, summary.totalBoundary57OracleCalls);
const overallPass = teacherResultsPassed
  && totalOracleReductionVsBoundary57 > 0.3
  && summary.rejectedTooCoarseAttemptCount > 0;
summary.pass = overallPass;

const report = {
  generatedAt: new Date().toISOString(),
  task: 'oracle-guided subcell boundary adaptive probing for deep black-box ReLU 2D region graphs',
  blackBox: {
    type: '2-input / 4-hidden / 4-hidden / 1-output folded two-layer ReLU MLP teachers',
    visibilityToGcl: 'oracle calls only: the decompiler receives f(x,y), not hidden activations, layer widths, or teacher weights'
  },
  claimScope: {
    proves: 'benchmark-scoped oracle-guided subcell validation that selects the lowest passing boundary-aware grid and records rejected too-coarse attempts',
    doesNotClaim: 'learned hidden-layer identity, exact layer attribution, topology-perfect boundary localization, arbitrary deep ND extraction, or production API extraction'
  },
  gates: {
    pass: overallPass,
    allTeachersPassed: summary.passedTeachers === summary.teacherCount,
    oracleReductionVsFixedAbove60Percent: summary.minOracleReductionVsFixed > 0.6,
    totalOracleReductionVsBoundary57Above30Percent: totalOracleReductionVsBoundary57 > 0.3,
    maxMaeBelow0003: summary.maxSubcellAdaptiveMae < 0.0003,
    minR2Above99999: summary.minSubcellAdaptiveR2 > 0.99999,
    selectedBelow57Grid: summary.selectedGridWidths.every(width => width < 57),
    recordsRejectedTooCoarseAttempt: summary.rejectedTooCoarseAttemptCount > 0
  },
  summary: {
    ...summary,
    totalOracleReductionVsFixed,
    totalOracleReductionVsBoundary57
  },
  results,
  runtimeMs: Date.now() - started
};

const markdown = [
  '# Subcell Boundary Adaptive Deep Black-Box ReLU 2D Region-Graph Benchmark',
  '',
  `Generated: ${report.generatedAt}`,
  '',
  `Pass: ${summary.pass ? 'yes' : 'no'}`,
  '',
  '## Scope',
  '',
  '- GCL access: oracle calls only, `f(x,y)`',
  '- Claim: subcell validation selects the lowest passing boundary-aware grid and records rejected too-coarse attempts',
  '- Not claimed: original hidden-layer identity, exact layer attribution, topology-perfect boundary localization, arbitrary deep ND extraction, or production API extraction',
  '',
  '## Summary',
  '',
  `- Teachers passed: ${summary.passedTeachers}/${summary.teacherCount}`,
  `- Selected grid widths: ${summary.selectedGridWidths.join(', ')}`,
  `- Rejected too-coarse attempts: ${summary.rejectedTooCoarseAttemptCount}`,
  `- Minimum per-teacher oracle reduction vs fixed 101x101: ${(summary.minOracleReductionVsFixed * 100).toFixed(3)}%`,
  `- Total oracle reduction vs fixed 101x101: ${(totalOracleReductionVsFixed * 100).toFixed(3)}%`,
  `- Total oracle reduction vs boundary-refined 57x57: ${(totalOracleReductionVsBoundary57 * 100).toFixed(3)}%`,
  `- Total fixed 101x101 oracle calls: ${summary.totalFixed101OracleCalls}`,
  `- Total boundary-refined 57x57 oracle calls: ${summary.totalBoundary57OracleCalls}`,
  `- Total subcell-adaptive oracle calls: ${summary.totalSubcellAdaptiveOracleCalls}`,
  `- Max subcell-adaptive MAE: ${summary.maxSubcellAdaptiveMae.toExponential(6)}`,
  `- Min subcell-adaptive R2: ${summary.minSubcellAdaptiveR2.toFixed(12)}`,
  `- Minimum held-out coarse-to-subcell MAE improvement: ${summary.minHeldOutMaeImprovementRatio.toFixed(6)}x`,
  '',
  '## Results',
  '',
  '| Teacher | Pass | Selected Grid | Attempts | Fixed Calls | Boundary57 Calls | Subcell Calls | Reduction vs Fixed | Reduction vs Boundary57 | Subcell MAE | Subcell R2 |',
  '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|',
  ...results.map(result => `| ${result.teacherId} | ${result.pass ? 'yes' : 'no'} | ${result.subcellAdaptive.selectedGridWidth} | ${result.subcellAdaptive.attempts.map(attempt => `${attempt.gridWidth}:${attempt.pass ? 'pass' : 'fail'}`).join(', ')} | ${result.fixed101.rawOracleCalls} | ${result.boundaryRefined57.totalOracleCalls} | ${result.subcellAdaptive.totalOracleCalls} | ${(result.subcellAdaptive.oracleReductionVsFixed * 100).toFixed(3)}% | ${(result.subcellAdaptive.oracleReductionVsBoundary57 * 100).toFixed(3)}% | ${result.subcellAdaptive.metrics.mae.toExponential(6)} | ${result.subcellAdaptive.metrics.r2.toFixed(12)} |`),
  ''
].join('\n');

fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, markdown);
writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph',
  status: overallPass ? 'pass' : 'fail',
  summary: {
    ...summary,
    totalOracleReductionVsFixed,
    totalOracleReductionVsBoundary57
  },
  notes: [
    'This benchmark records too-coarse candidate grids instead of silently promoting them.',
    'The solver uses oracle-guided subcell validation to choose the lowest passing boundary-aware grid.',
    'This is a query-budget and validation discipline gate, not a claim of exact hidden-layer recovery or topology-perfect boundary localization.'
  ]
});

console.log(JSON.stringify(report, null, 2));
if (!overallPass) process.exit(1);
