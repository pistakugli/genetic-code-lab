import {
  decompileNoisyBlackBoxRelu3D,
  predictNoisyBlackBoxRelu3DBatch
} from '../src/solvers/noisy-blackbox-relu3d-decompiler';
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

describe('noisy black-box 3D ReLU decompiler', () => {
  function teacher(x: number, y: number, z: number): number {
    return 0.18
      + 0.22 * x
      - 0.11 * y
      + 0.07 * z
      + 0.82 * Math.max(0, x + 0.25 * y - 0.20 * z + 1.10)
      - 0.48 * Math.max(0, x - 0.35 * y + 0.30 * z - 0.15)
      + 0.31 * Math.max(0, x + 0.10 * y + 0.22 * z - 1.20);
  }

  it('recovers useful 3D hyperplane geometry from noisy quantized f(x,y,z) probes', () => {
    const random = makeRandom(260527);
    const noisyOracle = (x: number, y: number, z: number) => {
      const noisy = teacher(x, y, z) + gaussian(random) * 0.006;
      return Math.round(noisy / 0.002) * 0.002;
    };
    const model = decompileNoisyBlackBoxRelu3D(noisyOracle, {
      xRange: [-3, 3],
      yRange: [-1.4, 1.4],
      zRange: [-1.2, 1.2],
      maxTerms: 3,
      sliceCount: 7,
      ySliceCount: 9,
      samplesPerSlice: 261,
      querySamples: 17,
      minSegmentPoints: 16,
      clusterSlopeTolerance: 0.10,
      kinkLineTolerance: 0.11,
      hyperplaneTolerance: 0.12
    });
    const inputs: Array<[number, number, number]> = [];
    for (let i = 0; i < 19; i++) {
      for (let j = 0; j < 13; j++) {
        for (let k = 0; k < 11; k++) {
          inputs.push([
            -2.75 + (5.5 * i) / 18,
            -1.25 + (2.5 * j) / 12,
            -1.05 + (2.1 * k) / 10
          ]);
        }
      }
    }
    const actual = inputs.map(([x, y, z]) => teacher(x, y, z));
    const predicted = predictNoisyBlackBoxRelu3DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const recovered = model.terms
      .map(term => ({ alphaY: term.alphaY, alphaZ: term.alphaZ, beta: term.beta, delta: term.deltaSlopeX }))
      .sort((a, b) => a.delta - b.delta);
    const expected = [
      { alphaY: 0.35, alphaZ: -0.30, beta: 0.15, delta: -0.48 },
      { alphaY: -0.10, alphaZ: -0.22, beta: 1.20, delta: 0.31 },
      { alphaY: -0.25, alphaZ: 0.20, beta: -1.10, delta: 0.82 }
    ].sort((a, b) => a.delta - b.delta);
    const maxError = expected.reduce((max, item, index) => Math.max(
      max,
      Math.abs(item.delta - recovered[index].delta),
      Math.abs(item.alphaY - recovered[index].alphaY),
      Math.abs(item.alphaZ - recovered[index].alphaZ),
      Math.abs(item.beta - recovered[index].beta)
    ), 0);

    expect(model.terms).toHaveLength(3);
    expect(maxError).toBeLessThan(0.12);
    expect(metrics.mae).toBeLessThan(0.08);
    expect(metrics.r2).toBeGreaterThan(0.995);
  });
});
