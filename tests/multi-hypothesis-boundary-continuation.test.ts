import {
  decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation,
  predictBlackBoxReluNdMultiHypothesisBoundaryContinuation,
  MbcdRankPolicy,
  MultiHypothesisBoundaryContinuationConfig
} from '../src/solvers/multi-hypothesis-boundary-continuation';

function countedOracle(oracle: (input: number[]) => number): {
  oracle: (input: number[]) => number;
  calls: () => number;
} {
  let count = 0;
  return {
    oracle: input => {
      count++;
      return oracle(input);
    },
    calls: () => count
  };
}

function config(overrides: Partial<MultiHypothesisBoundaryContinuationConfig> = {}) {
  return {
    dimension: 2,
    ranges: [[-1, 1], [-1, 1]] as Array<[number, number]>,
    termCount: 2,
    oracleBudget: 500,
    maxGradientProbes: 32,
    randomSeed: 90317,
    ...overrides
  };
}

describe('Multi-Hypothesis Boundary Continuation and Disambiguation (MBCD)', () => {
  it.each([0, 1, 4])('makes no partial gradient stencil when budget %i is below its cost', budget => {
    const counted = countedOracle(input => Math.max(0, input[0] - 0.1));
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, config({
      oracleBudget: budget
    }));

    expect(counted.calls()).toBe(0);
    expect(model.diagnostics.totalOracleCalls).toBe(0);
    expect(model.diagnostics.gradientProbes).toBe(0);
    expect(model.diagnostics.budgetExhausted).toBe(true);
    expect(model.diagnostics.reconstructionComplete).toBe(false);
  });

  it.each([5, 11])('counts all real oracle calls for budget %i without overrun', budget => {
    const counted = countedOracle(input => 0.25 * input[0] - 0.1 * input[1]);
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, config({
      oracleBudget: budget
    }));

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(counted.calls()).toBeLessThanOrEqual(budget);
    expect(counted.calls()).toBe(model.diagnostics.gradientProbes * 5);
    expect(model.diagnostics.oracleCallsPerGradientProbe).toBe(5);
    expect(model.diagnostics.budgetExhausted).toBe(true);
    expect(model.diagnostics.reconstructionComplete).toBe(false);
  });

  it('retains unresolved active-rank alternatives instead of declaring a rank from constant-gradient evidence', () => {
    const counted = countedOracle(() => 0.5);
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, config({
      termCount: 2,
      oracleBudget: 40,
      maxGradientProbes: 8
    }));

    expect(model.diagnostics.plausibleActiveRanks).toContain(0);
    expect(model.diagnostics.plausibleActiveRanks).toContain(1);
    expect(model.diagnostics.rankResolved).toBe(false);
    expect(model.diagnostics.reasons).toContain('competing_active_rank_hypotheses_remain');
    expect(model.reconstructionComplete).toBe(false);
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
  });

  it.each<MbcdRankPolicy>(['retain-alternatives', 'single-best'])(
    'reports the configured rank policy %s without inventing oracle evidence',
    rankPolicy => {
      const counted = countedOracle(input => input[0] + 0.25 * input[1]);
      const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, config({
        rankPolicy,
        oracleBudget: 55,
        maxGradientProbes: 10
      }));

      expect(model.diagnostics.rankPolicy).toBe(rankPolicy);
      expect(model.diagnostics.totalOracleCalls).toBe(counted.calls());
      if (rankPolicy === 'retain-alternatives') {
        expect(model.diagnostics.rankHypotheses.length).toBeGreaterThanOrEqual(1);
      } else {
        expect(model.diagnostics.rankHypotheses).toHaveLength(1);
      }
    }
  );

  it.each([
    ['contradiction-seeking', 'contradictionQueries'],
    ['prediction-disagreement', 'predictionDisagreementQueries']
  ] as const)('executes and separately counts the %s acquisition policy', (queryPolicy, counter) => {
    const counted = countedOracle(input =>
      Math.max(0, input[0] - 0.1) + 0.8 * Math.max(0, input[1] + 0.2));
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, config({
      queryPolicy,
      oracleBudget: 500,
      maxGradientProbes: 48
    }));

    expect(model.diagnostics.queryPolicy).toBe(queryPolicy);
    expect(model.diagnostics.acquisitionQueries).toBeGreaterThan(0);
    expect(model.diagnostics[counter]).toBeGreaterThan(0);
    expect(model.diagnostics[counter]).toBeLessThanOrEqual(model.diagnostics.acquisitionQueries);
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(counted.calls()).toBeLessThanOrEqual(model.diagnostics.oracleBudget);
    expect(model.diagnostics.boundaryValidationQueries).toBeLessThanOrEqual(
      model.diagnostics.gradientProbes
    );
  });

  it('keeps detection, normal, offset, uniqueness, and completion as separate claims', () => {
    const counted = countedOracle(input =>
      Math.max(0, 0.8 * input[0] + 0.6 * input[1] - 0.05));
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, config({
      termCount: 1,
      oracleBudget: 20,
      maxGradientProbes: 4
    }));

    expect(model.diagnostics.boundaryDetectionCount).toBeGreaterThanOrEqual(
      model.diagnostics.acceptedBoundaryCount
    );
    expect(model.diagnostics.normalIdentifiedCount).toBeGreaterThanOrEqual(
      model.diagnostics.acceptedBoundaryCount
    );
    expect(model.diagnostics.offsetIdentifiedCount).toBeGreaterThanOrEqual(
      model.diagnostics.acceptedBoundaryCount
    );
    expect(model.diagnostics.uniqueHypothesisCount).toBeGreaterThanOrEqual(
      model.diagnostics.acceptedBoundaryCount
    );
    expect(model.diagnostics.reconstructionComplete).toBe(false);
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
  });

  it('uses the oracle-derived affine-intercept jump to preserve the boundary offset sign', () => {
    const counted = countedOracle(input => Math.max(0, input[0] - 0.2));
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(counted.oracle, {
      dimension: 1,
      ranges: [[-1, 1]],
      termCount: 1,
      oracleBudget: 180,
      maxGradientProbes: 50,
      randomSeed: 712
    });

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(model.terms.some(term => Math.abs(term.normal[0]) > 0.99
      && Math.abs(term.offset + 0.2) < 0.03)).toBe(true);
  });

  it('returns a finite prediction and rejects input dimension mismatches', () => {
    const teacher = (input: number[]) =>
      0.2 * input[0] + Math.max(0, 0.6 * input[0] + 0.8 * input[1] - 0.02);
    const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(teacher, config({
      termCount: 1,
      oracleBudget: 120,
      maxGradientProbes: 20
    }));

    expect(Number.isFinite(
      predictBlackBoxReluNdMultiHypothesisBoundaryContinuation(model, [0.2, -0.3])
    )).toBe(true);
    expect(() => predictBlackBoxReluNdMultiHypothesisBoundaryContinuation(model, [0, 1, 2]))
      .toThrow('MBCD prediction dimension mismatch');
  });
});
