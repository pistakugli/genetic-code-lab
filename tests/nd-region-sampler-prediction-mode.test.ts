import {
  decompileDeepBlackBoxReluNdRegionSampler,
  predictDeepBlackBoxReluNdRegionSamplerModel
} from '../src/solvers/deep-blackbox-relu-nd-region-sampler';
import { setSeed } from '../src/utils/helpers';

/**
 * predictLocalAffineKnn degrades to a region vote in three places without saying so: fewer
 * neighbours than dimension + 1, a singular local system, and a non-finite result. The model
 * kept advertising predictionMode 'local-affine-knn' regardless, and buildExpression prints
 * that mode into the expression string readers take at face value, so a model that mostly
 * voted looked exactly like one that fitted.
 *
 * The error metrics were never wrong, being computed against whatever was actually
 * predicted. What was missing was the distinction, which the model now carries as
 * predictionModeFallbacks and the adaptive-budget gate binds on.
 *
 * These tests pin that the record is real: it is populated during the solve without anyone
 * opting in, and it actually rises when the declared mode cannot be used. A counter that can
 * only ever read zero would be no better than the silence it replaces.
 */
describe('ND region sampler prediction-mode fallback record', () => {
  const dimension = 3;
  const teacher = (input: number[]): number => {
    const a = Math.max(0, 0.8 * input[0] - 0.5 * input[1] + 0.2);
    const b = Math.max(0, -0.4 * input[0] + 0.7 * input[2] - 0.1);
    return 1.3 * a - 0.9 * b + 0.05;
  };

  const build = (predictionMode: 'region-vote' | 'local-affine-knn') => {
    setSeed(20260801);
    return decompileDeepBlackBoxReluNdRegionSampler(teacher, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
      sampleCount: 320,
      predictionMode
    });
  };

  test('the record is populated during the solve, with no opt-in', () => {
    const model = build('local-affine-knn');

    expect(model.predictionModeFallbacks.predictionCount).toBe(model.assignments.length);
    expect(model.predictionModeFallbacks.total).toBe(
      model.predictionModeFallbacks.insufficientNeighbors + model.predictionModeFallbacks.singularOrNonFinite
    );
    expect(model.predictionModeFallbacks.appliedRate).toBeGreaterThanOrEqual(0);
    expect(model.predictionModeFallbacks.appliedRate).toBeLessThanOrEqual(1);
  });

  test('a well-supported local-affine model reports its declared mode as applied', () => {
    const model = build('local-affine-knn');

    expect(model.predictionMode).toBe('local-affine-knn');
    expect(model.predictionModeFallbacks.appliedRate).toBe(1);
  });

  test('the counter actually rises when there are too few neighbours for an affine fit', () => {
    // Requiring more neighbours than the model retained forces the insufficient-neighbour
    // path on every point, which is the degradation predictionMode used to hide entirely.
    const model = build('local-affine-knn');
    const starved = { ...model, localAffineNeighborCount: 1 };
    const counts = { insufficientNeighbors: 0, singularOrNonFinite: 0, total: 0 };

    for (const assignment of model.assignments.slice(0, 12)) {
      predictDeepBlackBoxReluNdRegionSamplerModel(starved, assignment.input, counts);
    }

    expect(counts.insufficientNeighbors).toBe(12);
    expect(counts.total).toBe(12);
  });

  test('a region-vote model reports a full applied rate, since it declares no affine fit', () => {
    const model = build('region-vote');

    expect(model.predictionMode).toBe('region-vote');
    expect(model.predictionModeFallbacks.appliedRate).toBe(1);
    expect(model.predictionModeFallbacks.total).toBe(0);
  });

  test('measuring does not change what the model predicts', () => {
    const model = build('local-affine-knn');
    const probes = model.assignments.slice(0, 6).map(assignment => assignment.input);

    const before = probes.map(input => predictDeepBlackBoxReluNdRegionSamplerModel(model, input));
    const counts = { insufficientNeighbors: 0, singularOrNonFinite: 0, total: 0 };
    probes.forEach(input => predictDeepBlackBoxReluNdRegionSamplerModel(model, input, counts));
    const after = probes.map(input => predictDeepBlackBoxReluNdRegionSamplerModel(model, input));

    expect(after).toEqual(before);
  });
});
