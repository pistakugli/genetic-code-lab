import {
  decompileDeepBlackBoxRelu2DRegionGraphAdaptive,
  decompileDeepBlackBoxRelu2DRegionGraph,
  decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined,
  decompileDeepBlackBoxRelu2DRegionGraphSubcellBoundaryAdaptive,
  predictDeepBlackBoxRelu2DRegionGraphCoarseModel,
  predictDeepBlackBoxRelu2DRegionGraphBatch
} from '../src/solvers/deep-blackbox-relu2d-region-graph';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('deep black-box ReLU 2D region graph decompiler', () => {
  type Layer1Unit = { w: [number, number]; b: number };
  type LayerUnit = { w: number[]; b: number };

  const teacher = {
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

  function makeRandomDeepTeacher(seed: number): {
    l1: Layer1Unit[];
    l2: LayerUnit[];
    out: { w: number[]; b: number; base: [number, number] };
  } {
    const random = makeRandom(seed);
    const l1: Layer1Unit[] = [];
    for (let index = 0; index < 4; index++) {
      for (let attempt = 0; attempt < 1000; attempt++) {
        const angle = randomBetween(random, -Math.PI, Math.PI);
        const scale = randomBetween(random, 0.65, 1.55);
        const normal: [number, number] = [Math.cos(angle) * scale, Math.sin(angle) * scale];
        const separated = l1.every(unit => {
          const dot = (unit.w[0] * normal[0] + unit.w[1] * normal[1]) / Math.max(1e-12, norm2(unit.w[0], unit.w[1]) * norm2(normal[0], normal[1]));
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

  function deepTeacher(x: number, y: number): number {
    const h1 = teacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b));
    const h2 = teacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
    return teacher.out.b
      + teacher.out.base[0] * x
      + teacher.out.base[1] * y
      + teacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
  }

  it('recovers a held-out accurate 2D affine-region adjacency graph for a folded two-layer ReLU teacher', () => {
    const model = decompileDeepBlackBoxRelu2DRegionGraph(deepTeacher, {
      xRange: [-2, 2],
      yRange: [-2, 2],
      gridWidth: 101,
      gridHeight: 101,
      gradientStep: 1e-5,
      slopeTolerance: 1e-5,
      interceptTolerance: 1e-5
    });
    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 81; i++) {
      for (let j = 0; j < 71; j++) {
        inputs.push([
          -1.95 + (3.9 * (i + 0.37)) / 81,
          -1.95 + (3.9 * (j + 0.23)) / 71
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => deepTeacher(x, y));
    const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.recoveryKind).toBe('output-equivalent-region-graph');
    expect(model.regionCount).toBeGreaterThanOrEqual(18);
    expect(model.regionCount).toBeGreaterThan(teacher.l1.length);
    expect(model.adjacency.length).toBeGreaterThanOrEqual(35);
    expect(model.kinkCellBoundaryCount).toBeGreaterThan(850);
    expect(model.compactBoundarySegmentCount).toBe(model.adjacency.length);
    expect(model.kinkCellBoundaryCount / model.compactBoundarySegmentCount).toBeGreaterThan(10);
    expect(Math.max(...model.boundarySegments.map(segment => segment.fitRmse))).toBeLessThan(0.08);
    expect(model.boundarySegments.every(segment => Math.abs(Math.hypot(...segment.normal) - 1) < 1e-9)).toBe(true);
    expect(model.boundarySegments.every(segment => segment.length > 0)).toBe(true);
    expect(metrics.mae).toBeLessThan(0.001);
    expect(metrics.r2).toBeGreaterThan(0.99999);
  });

  it('clips compact boundary segments to topology-preserving affine equality corridors', () => {
    const model = decompileDeepBlackBoxRelu2DRegionGraph(deepTeacher, {
      xRange: [-2, 2],
      yRange: [-2, 2],
      gridWidth: 57,
      gridHeight: 57,
      gradientStep: 1e-5,
      slopeTolerance: 1e-5,
      interceptTolerance: 1e-5
    });
    const topology = model.topologyGeometry;
    expect(topology).toBeDefined();
    expect(topology?.pass).toBe(true);
    expect(topology?.topologyPreserved).toBe(true);
    expect(topology?.boundarySegmentCount).toBe(model.adjacency.length);
    expect(topology?.fallbackSegmentCount).toBe(0);
    expect(topology?.lineClippedSegmentCount).toBe(model.boundarySegments.length);
    expect(topology?.maxEndpointLineResidual).toBeLessThan(1e-9);
    expect(topology?.maxNormalError).toBeLessThan(1e-9);
    expect(topology?.minSegmentLength).toBeGreaterThan(0);
    expect(model.boundarySegments.every(segment => segment.topologyPreserving)).toBe(true);
    expect(model.boundarySegments.every(segment => segment.lineClipped)).toBe(true);
    expect(model.boundarySegments.every(segment => (
      Math.max(
        Math.abs(segment.normal[0] * segment.topologyStart[0] + segment.normal[1] * segment.topologyStart[1] + segment.offset),
        Math.abs(segment.normal[0] * segment.topologyEnd[0] + segment.normal[1] * segment.topologyEnd[1] + segment.offset)
      ) < 1e-9
    ))).toBe(true);
  });

  it('recovers held-out accurate region graphs for deterministic randomized folded teachers', () => {
    for (const seed of [29, 83]) {
      const randomTeacher = makeRandomDeepTeacher(seed);
      const oracle = (x: number, y: number): number => {
        const h1 = randomTeacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b));
        const h2 = randomTeacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
        return randomTeacher.out.b
          + randomTeacher.out.base[0] * x
          + randomTeacher.out.base[1] * y
          + randomTeacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
      };
      const model = decompileDeepBlackBoxRelu2DRegionGraph(oracle, {
        xRange: [-2, 2],
        yRange: [-2, 2],
        gridWidth: 101,
        gridHeight: 101,
        gradientStep: 1e-5,
        slopeTolerance: 1e-5,
        interceptTolerance: 1e-5
      });
      const inputs: Array<[number, number]> = [];
      for (let i = 0; i < 53; i++) {
        for (let j = 0; j < 47; j++) {
          inputs.push([
            -1.95 + (3.9 * (i + 0.41)) / 53,
            -1.95 + (3.9 * (j + 0.29)) / 47
          ]);
        }
      }
      const actual = inputs.map(([x, y]) => oracle(x, y));
      const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, inputs);
      const metrics = regressionMetrics(actual, predicted);

      expect(model.regionCount).toBeGreaterThanOrEqual(18);
      expect(model.regionCount - randomTeacher.l1.length - 1).toBeGreaterThanOrEqual(10);
      expect(model.adjacency.length).toBeGreaterThanOrEqual(35);
      expect(model.kinkCellBoundaryCount).toBeGreaterThanOrEqual(800);
      expect(metrics.mae).toBeLessThan(0.001);
      expect(metrics.r2).toBeGreaterThan(0.99999);
    }
  });

  it('preserves explicitly configured large gradient steps for repeated-query noisy probing', () => {
    let calls = 0;
    const oracle = (x: number, y: number): number => {
      calls++;
      return deepTeacher(x, y);
    };
    const model = decompileDeepBlackBoxRelu2DRegionGraph(oracle, {
      xRange: [-2, 2],
      yRange: [-2, 2],
      gridWidth: 17,
      gridHeight: 17,
      gradientStep: 0.2,
      slopeTolerance: 0.2,
      interceptTolerance: 0.3,
      querySamples: 3,
      aggregation: 'median'
    });

    expect(model.gradientStep).toBe(0.2);
    expect(model.querySamples).toBe(3);
    expect(model.aggregation).toBe('median');
    expect(calls).toBe(17 * 17 * 5 * 3);
    expect(model.regionCount).toBeGreaterThan(1);
  });

  it('selects a lower-budget deep 2D probe grid when validation and boundary gates pass', () => {
    let calls = 0;
    const oracle = (x: number, y: number): number => {
      calls++;
      return deepTeacher(x, y);
    };
    const model = decompileDeepBlackBoxRelu2DRegionGraphAdaptive(
      oracle,
      {
        xRange: [-2, 2],
        yRange: [-2, 2],
        gradientStep: 1e-5,
        slopeTolerance: 1e-5,
        interceptTolerance: 1e-5
      },
      {
        candidateGridSizes: [89, 101],
        validationGridWidth: 37,
        validationGridHeight: 31,
        maxValidationMae: 0.001,
        minValidationR2: 0.99999,
        maxBoundaryFitRmse: 0.08,
        minCompressionRatio: 8
      }
    );
    const adaptive = model.adaptiveProbe;
    expect(adaptive).toBeDefined();
    expect(adaptive?.selectedAttempt.pass).toBe(true);
    expect(adaptive?.selectedAttempt.gridWidth).toBeLessThan(101);
    expect(adaptive?.totalOracleCalls).toBeLessThan(101 * 101 * 5 * 0.85);
    expect(calls).toBe(adaptive?.totalOracleCalls);

    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 61; i++) {
      for (let j = 0; j < 59; j++) {
        inputs.push([
          -1.95 + (3.9 * (i + 0.19)) / 61,
          -1.95 + (3.9 * (j + 0.31)) / 59
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => deepTeacher(x, y));
    const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.compactBoundarySegmentCount).toBe(model.adjacency.length);
    expect(metrics.mae).toBeLessThan(0.001);
    expect(metrics.r2).toBeGreaterThan(0.99999);
  });

  it('uses compact boundaries to locally refine lower-grid deep 2D predictions', () => {
    let calls = 0;
    const oracle = (x: number, y: number): number => {
      calls++;
      return deepTeacher(x, y);
    };
    const model = decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined(
      oracle,
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
    const refinement = model.boundaryRefinement;
    expect(refinement).toBeDefined();
    expect(refinement?.pass).toBe(true);
    expect(refinement?.totalOracleCalls).toBe(57 * 57 * 5 + 37 * 31);
    expect(calls).toBe(refinement?.totalOracleCalls);
    expect(refinement?.totalOracleCalls).toBeLessThan(101 * 101 * 5 * 0.4);
    expect(refinement?.maeImprovementRatio).toBeGreaterThan(5);
    expect(refinement?.refinedValidationMae).toBeLessThan(refinement?.coarseValidationMae ?? Number.POSITIVE_INFINITY);

    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 61; i++) {
      for (let j = 0; j < 59; j++) {
        inputs.push([
          -1.95 + (3.9 * (i + 0.19)) / 61,
          -1.95 + (3.9 * (j + 0.31)) / 59
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => deepTeacher(x, y));
    const coarsePredicted = inputs.map(([x, y]) => predictDeepBlackBoxRelu2DRegionGraphCoarseModel(model, x, y));
    const refinedPredicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, inputs);
    const coarseMetrics = regressionMetrics(actual, coarsePredicted);
    const refinedMetrics = regressionMetrics(actual, refinedPredicted);

    expect(refinedMetrics.mae).toBeLessThan(coarseMetrics.mae / 5);
    expect(refinedMetrics.mae).toBeLessThan(0.0003);
    expect(refinedMetrics.r2).toBeGreaterThan(0.99999);
  });

  it('rejects a too-coarse subcell boundary grid and selects the first passing lower-budget grid', () => {
    const randomTeacher = makeRandomDeepTeacher(83);
    const oracle = (x: number, y: number): number => {
      const h1 = randomTeacher.l1.map(unit => relu(unit.w[0] * x + unit.w[1] * y + unit.b));
      const h2 = randomTeacher.l2.map(unit => relu(unit.b + unit.w.reduce((sum, weight, index) => sum + weight * h1[index], 0)));
      return randomTeacher.out.b
        + randomTeacher.out.base[0] * x
        + randomTeacher.out.base[1] * y
        + randomTeacher.out.w.reduce((sum, weight, index) => sum + weight * h2[index], 0);
    };
    let calls = 0;
    const countedOracle = (x: number, y: number): number => {
      calls++;
      return oracle(x, y);
    };
    const model = decompileDeepBlackBoxRelu2DRegionGraphSubcellBoundaryAdaptive(
      countedOracle,
      {
        xRange: [-2, 2],
        yRange: [-2, 2],
        gradientStep: 1e-5,
        slopeTolerance: 1e-5,
        interceptTolerance: 1e-5
      },
      {
        candidateGridSizes: [37, 41, 57],
        marginFactor: 0.5,
        validationGridWidth: 37,
        validationGridHeight: 31,
        maxValidationMae: 0.0003,
        minValidationR2: 0.99999
      }
    );
    const probe = model.subcellBoundaryProbe;
    expect(probe).toBeDefined();
    expect(probe?.selectedAttempt.gridWidth).toBe(41);
    expect(probe?.attempts).toHaveLength(2);
    expect(probe?.attempts[0].gridWidth).toBe(37);
    expect(probe?.attempts[0].pass).toBe(false);
    expect(probe?.selectedAttempt.pass).toBe(true);
    expect(probe?.totalOracleCalls).toBe((37 * 37 * 5 + 37 * 31) + (41 * 41 * 5 + 37 * 31));
    expect(calls).toBe(probe?.totalOracleCalls);
    expect(model.boundaryRefinement?.pass).toBe(true);
    expect(model.boundaryRefinement?.totalOracleCalls).toBe(41 * 41 * 5 + 37 * 31);

    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 61; i++) {
      for (let j = 0; j < 59; j++) {
        inputs.push([
          -1.95 + (3.9 * (i + 0.19)) / 61,
          -1.95 + (3.9 * (j + 0.31)) / 59
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => oracle(x, y));
    const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(metrics.mae).toBeLessThan(0.0003);
    expect(metrics.r2).toBeGreaterThan(0.99999);
  });
});
