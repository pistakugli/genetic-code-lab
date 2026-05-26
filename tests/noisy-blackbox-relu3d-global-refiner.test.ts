import {
  decompileNoisyBlackBoxRelu3D,
  predictNoisyBlackBoxRelu3DBatch
} from '../src/solvers/noisy-blackbox-relu3d-decompiler';
import {
  refineNoisyBlackBoxRelu3DGlobal,
  predictNoisyBlackBoxRelu3DGlobalModel
} from '../src/solvers/noisy-blackbox-relu3d-global-refiner';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

function makeRandom(seed: number): () => number {
  let state = seed % 2147483647;
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

function teacher(x: number, y: number, z: number): number {
  return 0.12
    + 0.18 * x
    - 0.09 * y
    + 0.06 * z
    + 0.786 * Math.max(0, x + 0.344 * y + 0.135 * z + 0.351)
    - 0.54557 * Math.max(0, x - 0.21045 * y + 0.23107 * z - 0.47786)
    + 0.38307 * Math.max(0, x - 0.07408 * y + 0.05761 * z - 0.44405);
}

function makeNoisyOracle(seed: number): (x: number, y: number, z: number) => number {
  const random = makeRandom(seed);
  return (x: number, y: number, z: number) => {
    const noisy = teacher(x, y, z) + gaussian(random) * 0.005;
    return Math.round(noisy / 0.002) * 0.002;
  };
}

function makeGrid(): Array<[number, number, number]> {
  const inputs: Array<[number, number, number]> = [];
  for (let i = 0; i < 17; i++) {
    for (let j = 0; j < 11; j++) {
      for (let k = 0; k < 9; k++) {
        inputs.push([
          -2.75 + (5.5 * i) / 16,
          -1.25 + (2.5 * j) / 10,
          -1.05 + (2.1 * k) / 8
        ]);
      }
    }
  }
  return inputs;
}

describe('noisy black-box 3D ReLU global refiner', () => {
  it('recovers an output-equivalent surface when close opposite-sign kinks break slice-only stitching', () => {
    const sliceModel = decompileNoisyBlackBoxRelu3D(makeNoisyOracle(810000), {
      xRange: [-3, 3],
      yRange: [-1.35, 1.35],
      zRange: [-1.15, 1.15],
      maxTerms: 3,
      sliceCount: 7,
      ySliceCount: 11,
      samplesPerSlice: 301,
      querySamples: 21,
      minSegmentPoints: 16,
      clusterSlopeTolerance: 0.10,
      kinkLineTolerance: 0.11,
      hyperplaneTolerance: 0.12
    });
    const refined = refineNoisyBlackBoxRelu3DGlobal(makeNoisyOracle(910000), {
      xRange: [-3, 3],
      yRange: [-1.35, 1.35],
      zRange: [-1.15, 1.15],
      termCount: 3,
      initialModel: sliceModel,
      sampleCounts: [9, 7, 7],
      querySamples: 11,
      restarts: 48,
      coordinatePasses: 2,
      randomSeed: 120000,
      betaRange: [-1.8, 1.8]
    });

    const inputs = makeGrid();
    const actual = inputs.map(([x, y, z]) => teacher(x, y, z));
    const sliceMetrics = regressionMetrics(actual, predictNoisyBlackBoxRelu3DBatch(sliceModel, inputs));
    const refinedMetrics = regressionMetrics(
      actual,
      inputs.map(([x, y, z]) => predictNoisyBlackBoxRelu3DGlobalModel(refined.model, x, y, z))
    );

    expect(refined.model.terms).toHaveLength(3);
    expect(refinedMetrics.mae).toBeLessThan(0.018);
    expect(refinedMetrics.r2).toBeGreaterThan(0.998);
    expect(refinedMetrics.mae).toBeLessThan(sliceMetrics.mae * 0.35);
  });
});
