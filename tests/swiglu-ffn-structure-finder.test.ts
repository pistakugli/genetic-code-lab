import {
  decompileRankAdaptiveSwigluFfnStructure,
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

  it('recovers a noisy quantized high-dimensional SwiGLU value/gate pair with repeated median queries', () => {
    const dimension = 10;
    const valueDirection = makeDirection(2101, dimension);
    const gateDirection = makeDirection(2202, dimension, [valueDirection]);
    const cleanOracle = (input: number[]): number => {
      const value = 1.04 * dot(valueDirection, input) - 0.16;
      const gate = 1.22 * dot(gateDirection, input) + 0.2;
      return 0.035 + 1.08 * value * silu(gate);
    };
    const noiseRandom = makeRandom(2303);
    const noisyOracle = (input: number[]): number => {
      const noisy = cleanOracle(input) + gaussian(noiseRandom) * 0.002;
      return Math.round(noisy / 0.001) * 0.001;
    };
    const model = decompileSwigluFfnStructure(noisyOracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.35, 1.35] as [number, number]),
      pairCount: 1,
      activeRank: 2,
      gradientProbeCount: 18,
      trainingSampleCount: 1800,
      validationSampleCount: 360,
      randomSeed: 2404,
      restarts: 10,
      iterations: 720,
      learningRate: 0.024,
      querySamples: 9,
      aggregation: 'median'
    });
    const inputs = makeInputs(2505, dimension, 700);
    const metrics = regressionMetrics(inputs.map(cleanOracle), predictSwigluFfnStructureBatch(model, inputs));
    const pair = model.pairs[0];
    const valueProjection = Math.abs(dot(valueDirection, normalize(pair.valueWeight)));
    const gateProjection = Math.abs(dot(gateDirection, normalize(pair.gateWeight)));

    expect(model.querySamples).toBe(9);
    expect(model.aggregation).toBe('median');
    expect(valueProjection).toBeGreaterThan(0.99);
    expect(gateProjection).toBeGreaterThan(0.99);
    expect(metrics.mae).toBeLessThan(0.03);
    expect(metrics.r2).toBeGreaterThan(0.997);
  });

  it('canonicalizes shared-gate SwiGLU source channels into one effective pair', () => {
    const dimension = 10;
    const gateDirection = makeDirection(7501, dimension);
    const valueDirection = makeDirection(7502, dimension, [gateDirection]);
    const perturbation = makeDirection(7503, dimension, [gateDirection, valueDirection]);
    const secondValueDirection = normalize(valueDirection.map((value, index) => (
      -0.8 * value + 0.45 * perturbation[index]
    )));
    const sourceBlocks = [
      { direction: valueDirection, scale: 1.06, bias: -0.14, output: 1.0 },
      { direction: secondValueDirection, scale: 0.9, bias: 0.11, output: -0.72 }
    ];
    const aggregateValue = new Array(dimension).fill(0);
    for (const block of sourceBlocks) {
      for (let axis = 0; axis < dimension; axis++) {
        aggregateValue[axis] += block.output * block.scale * block.direction[axis];
      }
    }
    const aggregateDirection = normalize(aggregateValue);
    const oracle = (input: number[]): number => {
      const gate = 1.13 * dot(gateDirection, input) + 0.18;
      return 0.02 + sourceBlocks.reduce((sum, block) => (
        sum + block.output * (block.scale * dot(block.direction, input) + block.bias) * silu(gate)
      ), 0);
    };
    const model = decompileRankAdaptiveSwigluFfnStructure(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.35, 1.35] as [number, number]),
      maxPairCount: 3,
      minEffectivePairCount: 1,
      activeRank: 4,
      gradientProbeCount: 18,
      trainingSampleCount: 1800,
      validationSampleCount: 320,
      randomSeed: 7601,
      restarts: 10,
      iterations: 700,
      learningRate: 0.025,
      selectionRestarts: 1,
      pruneValidationToleranceRatio: 0.35,
      pruneValidationToleranceAbsolute: 0.012,
      pairRelativeStrengthThreshold: 0.018,
      pairAbsoluteStrengthThreshold: 0.0015
    });
    const inputs = makeInputs(7701, dimension, 600);
    const metrics = regressionMetrics(inputs.map(oracle), predictSwigluFfnStructureBatch(model, inputs));
    const pair = model.pairs[0];
    const valueProjection = Math.abs(dot(aggregateDirection, normalize(pair.valueWeight)));
    const gateProjection = Math.abs(dot(gateDirection, normalize(pair.gateWeight)));

    expect(model.recoveryKind).toBe('rank-adaptive-swiglu-ffn-structure');
    expect(model.rawPairCount).toBe(3);
    expect(model.effectivePairCount).toBe(1);
    expect(model.pairs).toHaveLength(1);
    expect(valueProjection).toBeGreaterThan(0.999);
    expect(gateProjection).toBeGreaterThan(0.999);
    expect(metrics.mae).toBeLessThan(0.004);
    expect(metrics.r2).toBeGreaterThan(0.9999);
  });
});
