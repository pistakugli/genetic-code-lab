import {
  decompileBlackBoxReluNdLineProbe,
  predictBlackBoxReluNdLineProbeBatch
} from '../src/solvers/blackbox-relu-nd-line-probe';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

describe('black-box ND ReLU line-probe decompiler', () => {
  const teacherTerms = [
    { normal: [0.92, -0.28, 0.18, 0.12], offset: 0.58, readout: 0.72 },
    { normal: [0.36, 0.81, -0.31, 0.18], offset: -0.24, readout: -0.53 },
    { normal: [-0.22, 0.16, 0.88, -0.38], offset: -0.92, readout: 0.41 },
    { normal: [0.14, -0.42, 0.25, 0.86], offset: 0.18, readout: -0.34 }
  ];

  function teacher(input: number[]): number {
    return 0.17
      + dot([0.12, -0.08, 0.05, -0.03], input)
      + teacherTerms.reduce((sum, term) => sum + term.readout * Math.max(0, dot(term.normal, input) + term.offset), 0);
  }

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

  it('recovers 4D shallow ReLU hyperplanes from random-direction oracle line probes', () => {
    const ranges: Array<[number, number]> = [
      [-2.6, 2.6],
      [-2.3, 2.3],
      [-2.1, 2.1],
      [-1.9, 1.9]
    ];
    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 4,
      ranges,
      termCount: teacherTerms.length,
      lineCount: 160,
      samplesPerLine: 701,
      randomSeed: 260529,
      ransacIterations: 1800,
      inlierTolerance: 2e-5,
      minSupport: 32,
      slopeTolerance: 1e-8,
      kinkMergeTolerance: 1e-5
    });
    const inputs: number[][] = [];
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 11; j++) {
        for (let k = 0; k < 9; k++) {
          for (let m = 0; m < 7; m++) {
            inputs.push([
              -2.4 + (4.8 * i) / 12,
              -2.0 + (4.0 * j) / 10,
              -1.8 + (3.6 * k) / 8,
              -1.6 + (3.2 * m) / 6
            ]);
          }
        }
      }
    }
    const actual = inputs.map(teacher);
    const predicted = predictBlackBoxReluNdLineProbeBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.dimension).toBe(4);
    expect(model.terms).toHaveLength(4);
    expect(model.kinkPointCount).toBeGreaterThan(180);
    expect(model.terms.every(term => term.support >= 32)).toBe(true);
    expect(model.expression).toContain('x3');
    expect(metrics.mae).toBeLessThan(1e-8);
    expect(metrics.r2).toBeGreaterThan(0.999999999);
  });

  it('recovers 4D shallow ReLU hyperplanes from repeated noisy quantized line probes', () => {
    const ranges: Array<[number, number]> = [
      [-2.6, 2.6],
      [-2.3, 2.3],
      [-2.1, 2.1],
      [-1.9, 1.9]
    ];
    const random = makeRandom(260530);
    const noisyOracle = (input: number[]): number => {
      const noisy = teacher(input) + gaussian(random) * 0.0015;
      return Math.round(noisy / 0.0005) * 0.0005;
    };
    const model = decompileBlackBoxReluNdLineProbe(noisyOracle, {
      dimension: 4,
      ranges,
      termCount: teacherTerms.length,
      lineCount: 150,
      samplesPerLine: 221,
      querySamples: 9,
      aggregation: 'median',
      minSegmentPoints: 14,
      minAbsLineDeltaSlope: 0.06,
      randomSeed: 260531,
      ransacIterations: 2400,
      inlierTolerance: 0.035,
      minSupport: 32,
      ridge: 1e-8
    });
    const inputs: number[][] = [];
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 7; j++) {
        for (let k = 0; k < 7; k++) {
          for (let m = 0; m < 5; m++) {
            inputs.push([
              -2.35 + (4.7 * i) / 8,
              -2.0 + (4.0 * j) / 6,
              -1.8 + (3.6 * k) / 6,
              -1.55 + (3.1 * m) / 4
            ]);
          }
        }
      }
    }
    const actual = inputs.map(teacher);
    const predicted = predictBlackBoxReluNdLineProbeBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.terms).toHaveLength(4);
    expect(model.kinkPointCount).toBeGreaterThan(120);
    expect(model.terms.every(term => term.support >= 32)).toBe(true);
    expect(metrics.mae).toBeLessThan(0.025);
    expect(metrics.r2).toBeGreaterThan(0.998);
  });
});
