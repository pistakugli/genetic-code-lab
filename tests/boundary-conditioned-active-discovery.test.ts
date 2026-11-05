import {
  decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery,
  predictBlackBoxReluNdBoundaryConditionedActiveDiscovery,
  BcadAcquisitionStrategy,
  BcadRankStrategy
} from '../src/solvers/boundary-conditioned-active-discovery';

function dot(first: number[], second: number[]): number {
  return first.reduce((sum, value, axis) => sum + value * second[axis], 0);
}

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

function config(overrides: Partial<Parameters<typeof decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery>[1]> = {}) {
  return {
    dimension: 4,
    ranges: Array.from({ length: 4 }, () => [-1, 1] as [number, number]),
    termCount: 1,
    oracleBudget: 600,
    maxGradientProbes: 40,
    randomSeed: 90210,
    ...overrides
  };
}

describe('Boundary-Conditioned Active Discovery (BCAD)', () => {
  it.each([0, 1, 8, 9, 10])('counts actual oracle calls within budget %i', budget => {
    const counted = countedOracle(input => Math.max(0, input[0] + 0.1));
    const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(counted.oracle, config({
      oracleBudget: budget
    }));

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(counted.calls()).toBeLessThanOrEqual(budget);
    expect(counted.calls() % 9).toBe(0);
    if (budget < 9) {
      expect(counted.calls()).toBe(0);
      expect(model.diagnostics.budgetExhausted).toBe(true);
      expect(model.diagnostics.reconstructionComplete).toBe(false);
    }
  });

  it.each<BcadAcquisitionStrategy>([
    'boundary-uncertainty',
    'prediction-disagreement',
    'gradient-jump'
  ])('runs the %s acquisition strategy and reports its probes', acquisitionStrategy => {
    const counted = countedOracle(input =>
      0.1 * input[2] + 1.2 * Math.max(0, 0.6 * input[0] + 0.8 * input[1] - 0.025));
    const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(counted.oracle, config({
      acquisitionStrategy,
      oracleBudget: 1500,
      maxGradientProbes: 35
    }));

    expect(model.diagnostics.acquisitionStrategy).toBe(acquisitionStrategy);
    expect(model.diagnostics.acquisitionQueries[acquisitionStrategy])
      .toBe(model.diagnostics.gradientProbes - Math.min(2, model.diagnostics.gradientProbes));
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(counted.calls()).toBeLessThanOrEqual(1500);
    if (acquisitionStrategy === 'boundary-uncertainty') {
      expect(model.diagnostics.boundaryConditionedQueries).toBeGreaterThan(0);
    }
  });

  it('keeps near-parallel boundary positions as separate hypotheses', () => {
    const counted = countedOracle(input =>
      0.9 * Math.max(0, input[0] - 0.12)
      + 0.7 * Math.max(0, input[0] + 0.11));
    const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(counted.oracle, config({
      termCount: 2,
      oracleBudget: 1800,
      maxGradientProbes: 96,
      randomSeed: 4431
    }));

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(model.diagnostics.boundaryHypotheses).toBeGreaterThanOrEqual(2);
    expect(model.diagnostics.maximumBoundaryHypotheses).toBeGreaterThan(2);
    expect(model.diagnostics.boundaryValidationQueries).toBeGreaterThan(0);
    expect(model.diagnostics.boundaryValidationNegativeSupport.some(support => support > 0)).toBe(true);
    expect(model.diagnostics.boundaryValidationPositiveSupport.some(support => support > 0)).toBe(true);
    expect(model.diagnostics.candidatesRecovered).toBe(2);
    expect(model.diagnostics.reconstructionComplete).toBe(true);
  });

  it.each<BcadRankStrategy>([
    'fixed-initial',
    'stable-adaptive',
    'delayed-reduction',
    'full-dimensional'
  ])('records rank policy %s without using hidden rank labels', rankStrategy => {
    const counted = countedOracle(input => input.reduce((sum, value, axis) =>
      sum + (axis + 1) * Math.max(0, value + (axis - 1.5) * 0.04), 0));
    const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(counted.oracle, config({
      termCount: 4,
      rankStrategy,
      oracleBudget: 900,
      maxGradientProbes: 24
    }));

    expect(model.diagnostics.rankStrategy).toBe(rankStrategy);
    expect(model.diagnostics.estimatedActiveRank).toBeGreaterThanOrEqual(0);
    expect(model.diagnostics.estimatedActiveRank).toBeLessThanOrEqual(model.dimension);
    expect(model.diagnostics.rankMisestimated).toBeNull();
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(counted.calls()).toBeLessThanOrEqual(900);
    if (rankStrategy === 'full-dimensional') {
      expect(model.diagnostics.effectiveQueryRank).toBe(model.dimension);
    }
  });

  it('does not claim complete reconstruction when boundary alternatives remain unresolved', () => {
    const counted = countedOracle(input =>
      0.5 * Math.max(0, input[0] + 0.1)
      + 0.7 * Math.max(0, input[1] - 0.2)
      + 0.3 * Math.max(0, input[2] + input[3]));
    const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(counted.oracle, config({
      termCount: 3,
      oracleBudget: 500,
      maxGradientProbes: 12
    }));

    expect(model.diagnostics.reconstructionComplete).toBe(false);
    expect(model.diagnostics.reasons).toContain('competing_boundary_hypotheses_remain_unresolved');
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
  });

  it('predicts a recovered model and validates the input dimension', () => {
    const teacher = (input: number[]): number =>
      0.25 + 0.1 * input[2] + 1.4 * Math.max(0, 0.6 * input[0] + 0.8 * input[1] - 0.04);
    const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(teacher, config({
      oracleBudget: 1200,
      maxGradientProbes: 40
    }));
    const probe = [0.4, -0.3, 0.2, 0.1];

    expect(Number.isFinite(predictBlackBoxReluNdBoundaryConditionedActiveDiscovery(model, probe))).toBe(true);
    expect(() => predictBlackBoxReluNdBoundaryConditionedActiveDiscovery(model, [0, 1])).toThrow(
      'BCAD prediction dimension mismatch'
    );
  });
});
