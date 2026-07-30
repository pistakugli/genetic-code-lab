import { metrics } from '../src/solvers/gelu-ffn-structure-finder';
import { regressionQuality } from '../src/solvers/deep-blackbox-relu2d-region-graph';

/**
 * These metric helpers returned R2 = 1 whenever the target had zero variance, regardless
 * of what was predicted. R2 is genuinely undefined on a constant target, but the model is
 * still either right or wrong about that constant, and several gates in this repo compare
 * `metrics.r2` straight against a minimum bar. A degenerate validation slice therefore
 * scored a perfect 1 and auto-passed the R2 check no matter how bad the prediction was.
 *
 * The rule now: reproduce the constant and score 1, fail to reproduce it and score 0,
 * which is what "no better than predicting the mean" means when the mean is the whole
 * signal. Both helpers must behave identically, so both are pinned here.
 */
describe('R2 on a zero-variance target', () => {
  const constantTarget = [2.5, 2.5, 2.5, 2.5];

  test('a model that reproduces the constant still scores 1', () => {
    expect(metrics(constantTarget, [2.5, 2.5, 2.5, 2.5]).r2).toBe(1);
    expect(regressionQuality(constantTarget, [2.5, 2.5, 2.5, 2.5]).r2).toBe(1);
  });

  test('a model that misses the constant no longer scores a perfect 1', () => {
    const wrong = [0, 0, 0, 0];

    expect(metrics(constantTarget, wrong).r2).toBe(0);
    expect(regressionQuality(constantTarget, wrong).r2).toBe(0);
  });

  test('a badly wrong model cannot clear a 0.999 bar on a degenerate slice', () => {
    // This is the concrete failure the fix closes: before it, both sides of this
    // comparison were 1 and every gate reading r2 waved the slice through.
    const wildlyWrong = [100, -100, 100, -100].map(() => 999);

    expect(metrics(constantTarget, wildlyWrong).r2).toBeLessThan(0.999);
    expect(regressionQuality(constantTarget, wildlyWrong).r2).toBeLessThan(0.999);
  });

  test('an all-zero target behaves the same way', () => {
    const zeros = [0, 0, 0, 0];

    expect(metrics(zeros, [0, 0, 0, 0]).r2).toBe(1);
    expect(metrics(zeros, [0.5, 0.5, 0.5, 0.5]).r2).toBe(0);
    expect(regressionQuality(zeros, [0, 0, 0, 0]).r2).toBe(1);
    expect(regressionQuality(zeros, [0.5, 0.5, 0.5, 0.5]).r2).toBe(0);
  });

  test('the rule is scale-free, so a quiet constant is judged like a loud one', () => {
    const quiet = [1e-6, 1e-6, 1e-6, 1e-6];

    expect(metrics(quiet, [1e-6, 1e-6, 1e-6, 1e-6]).r2).toBe(1);
    expect(metrics(quiet, [2e-6, 2e-6, 2e-6, 2e-6]).r2).toBe(0);
  });

  test('a non-degenerate target is unaffected', () => {
    const varied = [1, 2, 3, 4];

    expect(metrics(varied, [1, 2, 3, 4]).r2).toBeCloseTo(1, 12);
    expect(regressionQuality(varied, [1, 2, 3, 4]).r2).toBeCloseTo(1, 12);
    expect(metrics(varied, [4, 3, 2, 1]).r2).toBeLessThan(0);
  });
});
