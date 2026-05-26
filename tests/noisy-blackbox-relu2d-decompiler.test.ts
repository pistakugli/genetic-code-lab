import {
  decompileNoisyBlackBoxRelu2D,
  predictNoisyBlackBoxRelu2DBatch
} from '../src/solvers/noisy-blackbox-relu2d-decompiler';
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

describe('noisy black-box 2D ReLU decompiler', () => {
  function teacher(x: number, y: number): number {
    return 0.2
      + 0.3 * x
      - 0.15 * y
      + 0.9 * Math.max(0, x + 0.35 * y + 1.25)
      - 0.55 * Math.max(0, x - 0.25 * y - 0.15)
      + 0.35 * Math.max(0, x + 0.15 * y - 1.25);
  }

  it('recovers useful kink-line geometry from noisy quantized f(x,y) probes', () => {
    const random = makeRandom(260524);
    const noisyOracle = (x: number, y: number) => {
      const noisy = teacher(x, y) + gaussian(random) * 0.01;
      return Math.round(noisy / 0.0025) * 0.0025;
    };
    const model = decompileNoisyBlackBoxRelu2D(noisyOracle, {
      xRange: [-3, 3],
      yRange: [-1.5, 1.5],
      maxTerms: 3,
      samplesPerSlice: 321,
      querySamples: 25,
      minSegmentPoints: 18,
      clusterSlopeTolerance: 0.08
    });
    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 41; i++) {
      for (let j = 0; j < 31; j++) {
        inputs.push([
          -2.8 + (5.6 * i) / 40,
          -1.35 + (2.7 * j) / 30
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => teacher(x, y));
    const predicted = predictNoisyBlackBoxRelu2DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const recovered = model.terms.map(term => term.deltaSlopeX).sort((a, b) => a - b);
    const expected = [-0.55, 0.35, 0.9];
    const maxDeltaError = expected.reduce((maxError, delta, index) =>
      Math.max(maxError, Math.abs(delta - recovered[index])), 0);

    expect(model.terms).toHaveLength(3);
    expect(maxDeltaError).toBeLessThan(0.08);
    expect(metrics.mae).toBeLessThan(0.04);
    expect(metrics.r2).toBeGreaterThan(0.998);
  });

  it('rejects bad slice detections when noisy kink lines cross', () => {
    const crossingTeacher = (x: number, y: number) =>
      0.2
      + 0.3 * x
      - 0.15 * y
      + 0.9 * Math.max(0, x + 0.4 * y + 1.2)
      - 0.55 * Math.max(0, x - 0.7 * y - 0.15)
      + 0.35 * Math.max(0, x + 0.2 * y - 1.1);
    const random = makeRandom(260524);
    const noisyOracle = (x: number, y: number) => {
      const noisy = crossingTeacher(x, y) + gaussian(random) * 0.01;
      return Math.round(noisy / 0.0025) * 0.0025;
    };
    const model = decompileNoisyBlackBoxRelu2D(noisyOracle, {
      xRange: [-3, 3],
      yRange: [-1.5, 1.5],
      maxTerms: 3,
      samplesPerSlice: 421,
      querySamples: 41,
      minSegmentPoints: 18,
      clusterSlopeTolerance: 0.08,
      kinkLineTolerance: 0.08
    });
    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 41; i++) {
      for (let j = 0; j < 31; j++) {
        inputs.push([
          -2.8 + (5.6 * i) / 40,
          -1.35 + (2.7 * j) / 30
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => crossingTeacher(x, y));
    const predicted = predictNoisyBlackBoxRelu2DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const recovered = model.terms
      .map(term => ({ alphaY: term.alphaY, beta: term.beta, delta: term.deltaSlopeX }))
      .sort((a, b) => a.delta - b.delta);
    const expected = [
      { alphaY: 0.7, beta: 0.15, delta: -0.55 },
      { alphaY: -0.2, beta: 1.1, delta: 0.35 },
      { alphaY: -0.4, beta: -1.2, delta: 0.9 }
    ].sort((a, b) => a.delta - b.delta);
    const maxError = expected.reduce((max, item, index) => Math.max(
      max,
      Math.abs(item.delta - recovered[index].delta),
      Math.abs(item.alphaY - recovered[index].alphaY),
      Math.abs(item.beta - recovered[index].beta)
    ), 0);

    expect(model.terms).toHaveLength(3);
    expect(maxError).toBeLessThan(0.08);
    expect(metrics.mae).toBeLessThan(0.04);
    expect(metrics.r2).toBeGreaterThan(0.998);
  });

  it('separates near-parallel noisy kink lines without merging them', () => {
    const nearParallelTeacher = (x: number, y: number) =>
      0.15
      + 0.28 * x
      - 0.12 * y
      + 0.85 * Math.max(0, x + 0.28 * y + 1.15)
      - 0.50 * Math.max(0, x + 0.24 * y - 0.05)
      + 0.32 * Math.max(0, x + 0.20 * y - 1.28);
    const random = makeRandom(260525);
    const noisyOracle = (x: number, y: number) => {
      const noisy = nearParallelTeacher(x, y) + gaussian(random) * 0.01;
      return Math.round(noisy / 0.0025) * 0.0025;
    };
    const model = decompileNoisyBlackBoxRelu2D(noisyOracle, {
      xRange: [-3, 3],
      yRange: [-1.5, 1.5],
      maxTerms: 3,
      samplesPerSlice: 421,
      querySamples: 41,
      minSegmentPoints: 18,
      clusterSlopeTolerance: 0.08,
      kinkLineTolerance: 0.08
    });
    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 41; i++) {
      for (let j = 0; j < 31; j++) {
        inputs.push([
          -2.8 + (5.6 * i) / 40,
          -1.35 + (2.7 * j) / 30
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => nearParallelTeacher(x, y));
    const predicted = predictNoisyBlackBoxRelu2DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const recovered = model.terms
      .map(term => ({ alphaY: term.alphaY, beta: term.beta, delta: term.deltaSlopeX }))
      .sort((a, b) => a.delta - b.delta);
    const expected = [
      { alphaY: -0.24, beta: 0.05, delta: -0.50 },
      { alphaY: -0.20, beta: 1.28, delta: 0.32 },
      { alphaY: -0.28, beta: -1.15, delta: 0.85 }
    ].sort((a, b) => a.delta - b.delta);
    const maxError = expected.reduce((max, item, index) => Math.max(
      max,
      Math.abs(item.delta - recovered[index].delta),
      Math.abs(item.alphaY - recovered[index].alphaY),
      Math.abs(item.beta - recovered[index].beta)
    ), 0);

    expect(model.terms).toHaveLength(3);
    expect(maxError).toBeLessThan(0.08);
    expect(metrics.mae).toBeLessThan(0.04);
    expect(metrics.r2).toBeGreaterThan(0.998);
  });

  it('survives kink-line collision on an observed slice boundary', () => {
    const collisionTeacher = (x: number, y: number) =>
      0.12
      + 0.25 * x
      - 0.10 * y
      + 0.75 * Math.max(0, x + 0.35 * y + 0.2)
      - 0.45 * Math.max(0, x - 0.35 * y + 0.2)
      + 0.28 * Math.max(0, x + 0.10 * y - 1.25);
    const random = makeRandom(260526);
    const noisyOracle = (x: number, y: number) => {
      const noisy = collisionTeacher(x, y) + gaussian(random) * 0.01;
      return Math.round(noisy / 0.0025) * 0.0025;
    };
    const model = decompileNoisyBlackBoxRelu2D(noisyOracle, {
      xRange: [-3, 3],
      yRange: [-1.5, 1.5],
      maxTerms: 3,
      samplesPerSlice: 421,
      querySamples: 41,
      minSegmentPoints: 18,
      clusterSlopeTolerance: 0.09,
      kinkLineTolerance: 0.08
    });
    const inputs: Array<[number, number]> = [];
    for (let i = 0; i < 41; i++) {
      for (let j = 0; j < 31; j++) {
        inputs.push([
          -2.8 + (5.6 * i) / 40,
          -1.35 + (2.7 * j) / 30
        ]);
      }
    }
    const actual = inputs.map(([x, y]) => collisionTeacher(x, y));
    const predicted = predictNoisyBlackBoxRelu2DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);
    const recovered = model.terms
      .map(term => ({ alphaY: term.alphaY, beta: term.beta, delta: term.deltaSlopeX }))
      .sort((a, b) => a.delta - b.delta);
    const expected = [
      { alphaY: 0.35, beta: -0.2, delta: -0.45 },
      { alphaY: -0.10, beta: 1.25, delta: 0.28 },
      { alphaY: -0.35, beta: -0.2, delta: 0.75 }
    ].sort((a, b) => a.delta - b.delta);
    const maxError = expected.reduce((max, item, index) => Math.max(
      max,
      Math.abs(item.delta - recovered[index].delta),
      Math.abs(item.alphaY - recovered[index].alphaY),
      Math.abs(item.beta - recovered[index].beta)
    ), 0);

    expect(model.terms).toHaveLength(3);
    expect(maxError).toBeLessThan(0.08);
    expect(metrics.mae).toBeLessThan(0.04);
    expect(metrics.r2).toBeGreaterThan(0.998);
  });
});
