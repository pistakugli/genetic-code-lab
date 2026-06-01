import {
  decompileActiveSubspaceFfn,
  predictActiveSubspaceFfnBatch
} from '../src/solvers/active-subspace-ffn-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('active-subspace FFN decompiler', () => {
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

  function dot(a: number[], b: number[]): number {
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
  }

  function normalize(values: number[]): number[] {
    const length = Math.sqrt(Math.max(1e-12, dot(values, values)));
    return values.map(value => value / length);
  }

  function gelu(value: number): number {
    return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
  }

  function silu(value: number): number {
    return value / (1 + Math.exp(-value));
  }

  function makeDirection(seed: number, dimension: number): number[] {
    const random = makeRandom(seed);
    return normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)));
  }

  function makeGaussianDirection(seed: number, dimension: number): number[] {
    const random = makeRandom(seed);
    const gaussian = (): number => {
      const u1 = Math.max(1e-12, random());
      const u2 = random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    return normalize(Array.from({ length: dimension }, gaussian));
  }

  function makeOrthogonalPair(seed: number, dimension: number): [number[], number[]] {
    const first = makeDirection(seed, dimension);
    const random = makeRandom(seed + 77);
    const raw = Array.from({ length: dimension }, () => randomBetween(random, -1, 1));
    const projection = dot(raw, first);
    const second = normalize(raw.map((value, index) => value - projection * first[index]));
    return [first, second];
  }

  function makeInputs(seed: number, dimension: number, count: number): number[][] {
    const random = makeRandom(seed);
    return Array.from({ length: count }, () => (
      Array.from({ length: dimension }, () => randomBetween(random, -1.5, 1.5))
    ));
  }

  function subspaceProjectionNorm(vector: number[], basis: number[][]): number {
    return Math.sqrt(basis.reduce((sum, direction) => sum + dot(vector, direction) ** 2, 0));
  }

  it('discovers a high-dimensional single GELU active direction from oracle calls', () => {
    const dimension = 64;
    const direction = makeDirection(7711, dimension);
    const oracle = (input: number[]): number => 0.17 + 1.35 * gelu(1.45 * dot(direction, input) - 0.22);
    const model = decompileActiveSubspaceFfn(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
      activeRank: 1,
      gradientProbeCount: 10,
      trainingSampleCount: 1400,
      finiteDifferenceStep: 1e-4,
      neighborCount: 42,
      randomSeed: 9101
    });
    const inputs = makeInputs(9102, dimension, 800);
    const actual = inputs.map(oracle);
    const predicted = predictActiveSubspaceFfnBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.recoveryKind).toBe('active-subspace-ffn-surrogate');
    expect(model.activeRank).toBe(1);
    expect(Math.abs(dot(direction, model.activeDirections[0]))).toBeGreaterThan(0.995);
    expect(model.selectedGradientEnergyRatio).toBeGreaterThan(0.999);
    expect(model.meanGradientResidualRatio).toBeLessThan(0.02);
    expect(model.queryBudget.rawOracleCalls).toBe(10 * 2 * dimension + 1400);
    expect(metrics.mae).toBeLessThan(0.006);
    expect(metrics.r2).toBeGreaterThan(0.9995);
  });

  it('discovers a rank-2 SwiGLU active subspace and predicts held-out outputs', () => {
    const dimension = 96;
    const [valueDirection, gateDirection] = makeOrthogonalPair(8811, dimension);
    const oracle = (input: number[]): number => {
      const value = 0.85 * dot(valueDirection, input) - 0.08;
      const gate = 1.25 * dot(gateDirection, input) + 0.19;
      return -0.04 + 1.15 * value * silu(gate);
    };
    const model = decompileActiveSubspaceFfn(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
      activeRank: 2,
      gradientProbeCount: 16,
      trainingSampleCount: 2200,
      finiteDifferenceStep: 1e-4,
      neighborCount: 72,
      randomSeed: 9201
    });
    const inputs = makeInputs(9202, dimension, 1000);
    const actual = inputs.map(oracle);
    const predicted = predictActiveSubspaceFfnBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(subspaceProjectionNorm(valueDirection, model.activeDirections)).toBeGreaterThan(0.985);
    expect(subspaceProjectionNorm(gateDirection, model.activeDirections)).toBeGreaterThan(0.985);
    expect(model.selectedGradientEnergyRatio).toBeGreaterThan(0.995);
    expect(model.meanGradientResidualRatio).toBeLessThan(0.05);
    expect(model.queryBudget.rawOracleCalls).toBe(16 * 2 * dimension + 2200);
    expect(metrics.mae).toBeLessThan(0.018);
    expect(metrics.r2).toBeGreaterThan(0.997);
  });

  it('recovers a noisy rank-1 GELU active direction without increasing oracle calls', () => {
    const dimension = 128;
    const direction = makeGaussianDirection(9911, dimension);
    const trueOracle = (input: number[]): number => 0.5 + 1.2 * gelu(1.8 * dot(direction, input) - 0.3);
    const noiseRandom = makeRandom(9912);
    const noisyOracle = (input: number[]): number => {
      const u1 = Math.max(1e-12, noiseRandom());
      const u2 = noiseRandom();
      const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.01;
      return trueOracle(input) + noise;
    };
    const model = decompileActiveSubspaceFfn(noisyOracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
      activeRank: 1,
      gradientProbeCount: 10,
      trainingSampleCount: 1800,
      finiteDifferenceStep: 1e-3,
      neighborCount: 52,
      randomSeed: 9913
    });
    const inputs = makeInputs(9914, dimension, 1000);
    const actual = inputs.map(trueOracle);
    const predicted = predictActiveSubspaceFfnBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.directionCandidateScores).toHaveLength(2);
    expect(model.finiteDifferenceStep).toBeGreaterThan(0.5);
    expect(Math.abs(dot(direction, model.activeDirections[0]))).toBeGreaterThan(0.998);
    expect(model.queryBudget.rawOracleCalls).toBe(10 * 2 * dimension + 1800);
    expect(metrics.mae).toBeLessThan(0.055);
    expect(metrics.r2).toBeGreaterThan(0.99);
  });

  it('uses stable affine local fitting when high rank lacks quadratic support', () => {
    const dimension = 20;
    const activeRank = 16;
    const coefficients = Array.from({ length: activeRank }, (_unused, index) => (
      (index % 2 === 0 ? 1 : -1) * (0.15 + 0.02 * index)
    ));
    const oracle = (input: number[]): number => coefficients.reduce((sum, coefficient, index) => (
      sum + coefficient * gelu((0.25 + 0.01 * index) * input[index] + (index - 8) * 0.01)
    ), 0);
    const model = decompileActiveSubspaceFfn(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
      activeRank,
      gradientProbeCount: 40,
      trainingSampleCount: 2600,
      neighborCount: 80,
      randomSeed: 1234
    });
    const random = makeRandom(5678);
    const inputs = Array.from({ length: 1000 }, () => (
      Array.from({ length: dimension }, () => randomBetween(random, -1, 1))
    ));
    const metrics = regressionMetrics(inputs.map(oracle), predictActiveSubspaceFfnBatch(model, inputs));

    expect(model.localModel).toBe('local-affine-knn');
    expect(model.directionEstimator).toBe('finite-difference');
    expect(metrics.mae).toBeLessThan(0.02);
    expect(metrics.r2).toBeGreaterThan(0.98);
  });
});
