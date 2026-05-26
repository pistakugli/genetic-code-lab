import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerBatch
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('deep black-box ReLU ND region sampler', () => {
  type Unit = { w: number[]; b: number };
  type Teacher = {
    dimension: number;
    l1: Unit[];
    l2: Unit[];
    out: { w: number[]; b: number; base: number[] };
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

  function dot(a: number[], b: number[]): number {
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
  }

  function normalize(values: number[]): number[] {
    const length = Math.sqrt(dot(values, values));
    return values.map(value => value / Math.max(1e-12, length));
  }

  function makeTeacher(seed: number, dimension: number): Teacher {
    const random = makeRandom(seed);
    const l1 = Array.from({ length: 4 }, () => ({
      w: normalize(Array.from({ length: dimension }, () => randomBetween(random, -1, 1)))
        .map(value => value * randomBetween(random, 0.6, 1.4)),
      b: randomBetween(random, -0.7, 0.7)
    }));
    const l2 = Array.from({ length: 4 }, (_, unitIndex) => ({
      w: Array.from({ length: 4 }, (_, weightIndex) => {
        const sign = (unitIndex + weightIndex) % 2 === 0 ? 1 : -1;
        return sign * randomBetween(random, 0.2, 1.0);
      }),
      b: randomBetween(random, -0.5, 0.5)
    }));
    return {
      dimension,
      l1,
      l2,
      out: {
        w: Array.from({ length: 4 }, (_, index) => {
          const sign = index % 2 === 0 ? 1 : -1;
          return sign * randomBetween(random, 0.3, 1.2);
        }),
        b: randomBetween(random, -0.1, 0.1),
        base: Array.from({ length: dimension }, () => randomBetween(random, -0.06, 0.06))
      }
    };
  }

  function predictTeacher(teacher: Teacher, input: number[]): number {
    const h1 = teacher.l1.map(unit => relu(dot(unit.w, input) + unit.b));
    const h2 = teacher.l2.map(unit => relu(dot(unit.w, h1) + unit.b));
    return teacher.out.b + dot(teacher.out.base, input) + dot(teacher.out.w, h2);
  }

  function makeValidationInputs(seed: number, dimension: number, count: number): number[][] {
    const random = makeRandom(seed);
    return Array.from({ length: count }, () => (
      Array.from({ length: dimension }, () => randomBetween(random, -1.75, 1.75))
    ));
  }

  it('recovers a held-out accurate 3D folded ReLU affine-region sampler from oracle calls only', () => {
    const teacher = makeTeacher(29, 3);
    const model = decompileDeepBlackBoxReluNdRegionSampler(
      input => predictTeacher(teacher, input),
      {
        dimension: 3,
        ranges: [[-2, 2], [-2, 2], [-2, 2]],
        sampleCount: 6500,
        gradientStep: 1e-5,
        slopeTolerance: 1e-5,
        interceptTolerance: 1e-5,
        minRegionSupport: 6,
        neighborCount: 11,
        randomSeed: 260526
      }
    );
    const inputs = makeValidationInputs(260527, 3, 2500);
    const actual = inputs.map(input => predictTeacher(teacher, input));
    const predicted = predictDeepBlackBoxReluNdRegionSamplerBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.recoveryKind).toBe('output-equivalent-nd-region-sampler');
    expect(model.dimension).toBe(3);
    expect(model.regionCount).toBeGreaterThan(teacher.l1.length + 20);
    expect(model.retainedSampleCount / model.sampleCount).toBeGreaterThan(0.98);
    expect(Math.max(...model.regions.map(region => region.fitRmse))).toBeLessThan(1e-8);
    expect(metrics.mae).toBeLessThan(0.008);
    expect(metrics.r2).toBeGreaterThan(0.9995);
  });
});
