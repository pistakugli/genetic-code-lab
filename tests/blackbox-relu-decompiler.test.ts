import {
  decompileBlackBoxRelu1D,
  predictBlackBoxReluBatch
} from '../src/solvers/blackbox-relu-decompiler';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('black-box ReLU decompiler', () => {
  function teacher(x: number): number {
    return 0.25
      + 0.35 * x
      + 1.1 * Math.max(0, x + 1.4)
      - 0.7 * Math.max(0, x - 0.25)
      + 0.45 * Math.max(0, x - 1.2);
  }

  it('recovers an equivalent piecewise ReLU circuit from input-output queries only', () => {
    const model = decompileBlackBoxRelu1D(teacher, {
      min: -3,
      max: 3,
      samples: 1201,
      slopeTolerance: 1e-8,
      minPlateauIntervals: 8
    });
    const xs = Array.from({ length: 301 }, (_, index) => -2.95 + (5.9 * index) / 300);
    const predicted = predictBlackBoxReluBatch(model, xs);
    const actual = xs.map(teacher);
    const metrics = regressionMetrics(actual, predicted);

    expect(model.terms).toHaveLength(3);
    expect(model.terms.map(term => term.kink)).toEqual(expect.arrayContaining([
      expect.closeTo(-1.4, 8),
      expect.closeTo(0.25, 8),
      expect.closeTo(1.2, 8)
    ]));
    expect(model.expression).toContain('max(0');
    expect(metrics.mae).toBeLessThan(1e-10);
    expect(metrics.r2).toBeGreaterThan(0.999999999);
  });
});
