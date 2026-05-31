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
});
