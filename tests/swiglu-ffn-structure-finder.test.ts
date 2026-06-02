import {
  decompileSwigluFfnStructure,
  predictSwigluFfnStructureBatch
} from '../src/solvers/swiglu-ffn-structure-finder';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('SwiGLU FFN structure finder', () => {
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

  function dot(a: number[], b: number[]): number {
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
  }

  function normalize(values: number[]): number[] {
    const length = Math.sqrt(Math.max(1e-12, dot(values, values)));
    return values.map(value => value / length);
  }

  function subtractProjection(vector: number[], basis: number[][]): number[] {
    let output = [...vector];
    for (const direction of basis) {
      const projection = dot(output, direction);
      output = output.map((value, index) => value - projection * direction[index]);
    }
    return output;
  }

  function makeDirection(seed: number, dimension: number, previous: number[][] = []): number[] {
    const random = makeRandom(seed);
    return normalize(subtractProjection(
      Array.from({ length: dimension }, () => gaussian(random)),
      previous
    ));
  }

  function silu(value: number): number {
    return value / (1 + Math.exp(-value));
  }

  function makeInputs(seed: number, dimension: number, count: number): number[][] {
    const random = makeRandom(seed);
    return Array.from({ length: count }, () => (
      Array.from({ length: dimension }, () => -1.35 + 2.7 * random())
    ));
  }

  it('recovers a high-dimensional SwiGLU value/gate pair as explicit structure', () => {
    const dimension = 12;
    const valueDirection = makeDirection(1101, dimension);
    const gateDirection = makeDirection(1202, dimension, [valueDirection]);
    const oracle = (input: number[]): number => {
      const value = 1.08 * dot(valueDirection, input) - 0.16;
      const gate = 1.22 * dot(gateDirection, input) + 0.21;
      return 0.035 + 1.12 * value * silu(gate);
    };
    const model = decompileSwigluFfnStructure(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.35, 1.35] as [number, number]),
      pairCount: 1,
      activeRank: 2,
      gradientProbeCount: 14,
      trainingSampleCount: 1100,
      validationSampleCount: 260,
      randomSeed: 1303,
      restarts: 8,
      iterations: 520,
      learningRate: 0.026
    });
    const inputs = makeInputs(1404, dimension, 700);
    const metrics = regressionMetrics(inputs.map(oracle), predictSwigluFfnStructureBatch(model, inputs));
    const pair = model.pairs[0];
    const valueProjection = Math.abs(dot(valueDirection, normalize(pair.valueWeight)));
    const gateProjection = Math.abs(dot(gateDirection, normalize(pair.gateWeight)));

    expect(model.recoveryKind).toBe('swiglu-ffn-structure');
    expect(model.pairs).toHaveLength(1);
    expect(valueProjection).toBeGreaterThan(0.995);
    expect(gateProjection).toBeGreaterThan(0.995);
    expect(Math.abs(pair.gateWeightNorm - 1.22)).toBeLessThan(0.18);
    expect(Math.abs(pair.gateBias - 0.21)).toBeLessThan(0.18);
    expect(metrics.mae).toBeLessThan(0.018);
    expect(metrics.r2).toBeGreaterThan(0.999);
  });
});
