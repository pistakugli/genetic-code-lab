import {
  decompileBlackBoxRelu2D,
  predictBlackBoxRelu2DBatch
} from '../src/solvers/blackbox-relu2d-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('black-box 2D ReLU decompiler', () => {
  function teacher(x: number, y: number): number {
    return 0.2
      + 0.3 * x
      - 0.15 * y
      + 0.9 * Math.max(0, x + 0.4 * y + 1.2)
      - 0.55 * Math.max(0, x - 0.7 * y - 0.15)
      + 0.35 * Math.max(0, x + 0.2 * y - 1.1);
  }

  it('recovers non-vertical ReLU kink lines from horizontal black-box probes', () => {
    const model = decompileBlackBoxRelu2D(teacher, {
      xRange: [-3, 3],
      yRange: [-1.5, 1.5],
      samplesPerSlice: 1201,
      slopeTolerance: 1e-8,
      clusterSlopeTolerance: 1e-7
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
    const predicted = predictBlackBoxRelu2DBatch(model, inputs);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.terms).toHaveLength(3);
    expect(model.terms.map(term => term.deltaSlopeX)).toEqual(expect.arrayContaining([
      expect.closeTo(0.9, 8),
      expect.closeTo(-0.55, 8),
      expect.closeTo(0.35, 8)
    ]));
    expect(model.expression).toContain('max(0');
    expect(metrics.mae).toBeLessThan(1e-10);
    expect(metrics.r2).toBeGreaterThan(0.999999999);
  });
});
