import { applyThresholdCourtFinalizer } from '../src/solvers/threshold-court-finalizer';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { PopulationBuffer, OpCode } from '../src/core/pop-buffer';

/**
 * The threshold court finaliser reads the exact inputs a program got wrong, appends literal
 * guard clauses for those masks, and re-scores against the same testCases. An audit could
 * not tell from the code alone whether that is exact Boolean synthesis or test-case
 * memorisation reported as accuracy, and the answer turns entirely on one property of the
 * caller: whether the cases are the complete truth table.
 *
 * They are. The release-gated path runs verify:default-threshold-policy ->
 * benchmark-majority-seeds.ts -> createMajorityProblem, which enumerates all 1 << n
 * assignments, so the specification is total and there is no held-out data to overfit.
 *
 * That made the safety an unstated assumption about the caller, which is what these tests
 * remove. The finaliser now refuses anything that is not exhaustive, so a future switch to
 * sampled cases fails loudly instead of quietly becoming overfitting that still reports a
 * high number.
 */
describe('threshold court finaliser exhaustiveness invariant', () => {
  const program = () => ({
    opcodes: Uint32Array.from([PopulationBuffer.pack(OpCode.XN, 0)]),
    constants: new Float64Array([1]),
    fitness: 0.99
  });

  test('the release-gated lane really does supply a complete truth table', () => {
    for (const n of [4, 6, 9]) {
      const problem = createMajorityProblem(n, Math.ceil(n / 2));
      expect(problem.testCases.length).toBe(1 << n);
      const seen = new Set(problem.testCases.map(testCase => testCase.input.join('')));
      expect(seen.size).toBe(1 << n);
    }
  });

  test('an exhaustive set is accepted', () => {
    const problem = createMajorityProblem(4, 2);

    expect(() => applyThresholdCourtFinalizer(problem.testCases, 2, program())).not.toThrow();
  });

  test('a sampled set is refused rather than silently memorised', () => {
    const problem = createMajorityProblem(4, 2);
    const sampled = problem.testCases.slice(0, 10);

    expect(() => applyThresholdCourtFinalizer(sampled, 2, program()))
      .toThrow(/exhaustive truth table: got 10 cases for 4 inputs, expected 16/);
  });

  test('a duplicated assignment padded to the right count is refused', () => {
    const problem = createMajorityProblem(4, 2);
    const padded = [...problem.testCases.slice(0, 15), problem.testCases[0]];

    expect(padded.length).toBe(16);
    expect(() => applyThresholdCourtFinalizer(padded, 2, program()))
      .toThrow(/each assignment to appear once/);
  });

  test('non-binary inputs are refused', () => {
    const problem = createMajorityProblem(4, 2);
    const corrupted = problem.testCases.map((testCase, index) =>
      index === 0 ? { ...testCase, input: [0.5, 0, 0, 0] } : testCase
    );

    expect(() => applyThresholdCourtFinalizer(corrupted, 2, program())).toThrow(/binary inputs/);
  });

  test('an empty set is refused', () => {
    expect(() => applyThresholdCourtFinalizer([], 2, program())).toThrow(/non-empty truth table/);
  });

  test('the invariant is only reached once the program clears the base fitness bar', () => {
    // Below minBaseFitness the finaliser returns untouched, so a weak program on a sampled
    // set must not throw; the guard belongs to the patching path, not to every call.
    const weak = { ...program(), fitness: 0.1 };

    expect(() => applyThresholdCourtFinalizer([], 2, weak)).not.toThrow();
  });
});
