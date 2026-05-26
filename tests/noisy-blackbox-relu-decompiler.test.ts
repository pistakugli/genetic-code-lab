import {
  decompileNoisyBlackBoxRelu1D,
  predictNoisyBlackBoxReluModel
} from '../src/solvers/noisy-blackbox-relu-decompiler';
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

describe('noisy black-box ReLU decompiler', () => {
  function teacher(x: number): number {
    return 0.25
      + 0.35 * x
      + 1.1 * Math.max(0, x + 1.4)
      - 0.7 * Math.max(0, x - 0.25)
      + 0.45 * Math.max(0, x - 1.2);
  }

  it('recovers a useful ReLU circuit from noisy and quantized oracle calls', () => {
    const random = makeRandom(20260523);
    const noisyOracle = (x: number) => {
      const noisy = teacher(x) + gaussian(random) * 0.01;
      return Math.round(noisy / 0.0025) * 0.0025;
    };
    const model = decompileNoisyBlackBoxRelu1D(noisyOracle, {
      min: -3,
      max: 3,
      maxTerms: 3,
      samples: 361,
      querySamples: 31,
      minSegmentPoints: 24
    });
    const xs = Array.from({ length: 501 }, (_, index) => -2.95 + (5.9 * index) / 500);
    const predicted = xs.map(x => predictNoisyBlackBoxReluModel(model, x));
    const actual = xs.map(teacher);
    const metrics = regressionMetrics(actual, predicted);
    const recovered = model.terms.map(term => term.kink).sort((a, b) => a - b);
    const expected = [-1.4, 0.25, 1.2];
    const maxKinkError = expected.reduce((maxError, kink, index) =>
      Math.max(maxError, Math.abs(kink - recovered[index])), 0);

    expect(model.terms).toHaveLength(3);
    expect(maxKinkError).toBeLessThan(0.04);
    expect(metrics.mae).toBeLessThan(0.02);
    expect(metrics.r2).toBeGreaterThan(0.999);
  });
});
