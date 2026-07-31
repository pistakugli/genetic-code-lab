import {
  decompileActiveSubspaceFfn,
  predictActiveSubspaceFfnModel,
  auditActiveSubspaceFfnFallbacks
} from '../src/solvers/active-subspace-ffn-decompiler';
import { setSeed } from '../src/utils/helpers';

/**
 * `model.localModel` states the intended local method, but predictFromSamples degrades
 * silently in two places: too few neighbours returns the global mean, and a singular local
 * system falls through to a kernel mean. The reported error metrics were always honest,
 * being recomputed against whatever was actually predicted, but nothing distinguished a
 * model that used its declared method throughout from one that mostly did not.
 *
 * auditActiveSubspaceFfnFallbacks measures that without changing any prediction. These
 * tests pin both that it reports a clean run as clean and that it actually catches a
 * degraded one, since a counter that can only ever report zero would be no better than the
 * silence it replaces.
 */
describe('active-subspace local-model fallback audit', () => {
  const dimension = 6;
  const teacher = (input: number[]): number => {
    const projection = input[0] * 0.8 - input[1] * 0.4 + input[2] * 0.2;
    return Math.tanh(projection) + 0.1 * projection * projection;
  };

  const buildModel = () => {
    setSeed(31337);
    return decompileActiveSubspaceFfn(teacher, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
      activeRank: 1
    });
  };

  test('predictions are unchanged by auditing, so the audit only measures', () => {
    const model = buildModel();
    const inputs = [
      Array.from({ length: dimension }, (_, i) => 0.1 * (i + 1)),
      Array.from({ length: dimension }, (_, i) => -0.05 * (i + 1))
    ];

    const before = inputs.map(input => predictActiveSubspaceFfnModel(model, input));
    auditActiveSubspaceFfnFallbacks(model, inputs);
    const after = inputs.map(input => predictActiveSubspaceFfnModel(model, input));

    expect(after).toEqual(before);
  });

  test('a well-supported model reports its declared method as applied', () => {
    const model = buildModel();
    const inputs = Array.from({ length: 24 }, (_, index) =>
      Array.from({ length: dimension }, (_, i) => Math.sin(index * 0.3 + i) * 0.5)
    );

    const audit = auditActiveSubspaceFfnFallbacks(model, inputs);

    expect(audit.predictionCount).toBe(inputs.length);
    expect(audit.appliedRate).toBeGreaterThan(0.5);
    expect(audit.total).toBe(audit.insufficientNeighbors + audit.singularLocalSystem);
  });

  test('the counter actually fires when there are too few neighbours to support the method', () => {
    // Starving the model of samples forces the insufficient-neighbour path, which is the
    // degradation the model's declared localModel used to hide completely.
    const model = buildModel();
    const starved = { ...model, samples: model.samples.slice(0, 1), activeRank: 8 };
    const inputs = Array.from({ length: 5 }, (_, index) =>
      Array.from({ length: dimension }, () => 0.1 * (index + 1))
    );

    const audit = auditActiveSubspaceFfnFallbacks(starved, inputs);

    expect(audit.insufficientNeighbors).toBe(inputs.length);
    expect(audit.total).toBe(inputs.length);
    expect(audit.appliedRate).toBe(0);
  });

  test('an empty input set is reported as fully applied rather than dividing by zero', () => {
    const audit = auditActiveSubspaceFfnFallbacks(buildModel(), []);

    expect(audit.predictionCount).toBe(0);
    expect(audit.appliedRate).toBe(1);
    expect(audit.total).toBe(0);
  });
});
