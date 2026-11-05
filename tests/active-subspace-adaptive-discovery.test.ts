import {
  decompileBlackBoxReluNdActiveSubspaceAdaptive,
  predictBlackBoxReluNdActiveSubspaceAdaptive
} from '../src/solvers/active-subspace-adaptive-discovery';

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
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

describe('Active-Subspace Adaptive Discovery (ASAD)', () => {
  it('recovers an isolated low-rank boundary through disagreement-guided queries', () => {
    const normal = [0.6, 0.8, 0, 0, 0, 0];
    const teacher = (input: number[]): number =>
      0.15 + dot([0.1, -0.07, 0.03, 0.02, -0.04, 0.06], input)
        + 1.4 * Math.max(0, dot(normal, input) - 0.12);
    const counted = countedOracle(teacher);
    const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
      dimension: 6,
      ranges: Array.from({ length: 6 }, () => [-1, 1]),
      termCount: 1,
      oracleBudget: 2500,
      randomSeed: 2601
    });

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(counted.calls()).toBeLessThanOrEqual(2500);
    expect(model.terms).toHaveLength(1);
    expect(Math.abs(dot(model.terms[0].normal, normal))).toBeGreaterThan(0.999);
    expect(model.terms[0].offset).toBeCloseTo(-0.12, 3);
    expect(model.diagnostics.activeSubspaceRank).toBe(1);
    expect(model.diagnostics.disagreementQueries).toBeGreaterThan(0);
    expect(model.diagnostics.reconstructionComplete).toBe(true);
    expect(model.diagnostics.gradientProbes).toBeGreaterThanOrEqual(
      1 + model.dimension + model.terms.length + model.diagnostics.requiredFitValidationObservations
    );
    expect(model.diagnostics.fitValidationObservations)
      .toBeGreaterThanOrEqual(model.diagnostics.requiredFitValidationObservations);
    expect(model.diagnostics.fitResidualRelative ?? Infinity).toBeLessThan(1e-5);

    const probes = [[-0.8, 0.2, 0, 0.1, 0.3, -0.1], [0.1, 0.5, -0.4, 0.1, 0.2, 0.3]];
    const errors = probes.map(input =>
      Math.abs(predictBlackBoxReluNdActiveSubspaceAdaptive(model, input) - teacher(input)));
    expect(Math.max(...errors)).toBeLessThan(1e-5);
  });

  it.each([0, 1])('never exceeds an oracle budget of %i', budget => {
    const counted = countedOracle(input => Math.max(0, input[0] + 0.1));
    const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
      dimension: 6,
      ranges: Array.from({ length: 6 }, () => [-1, 1]),
      termCount: 1,
      oracleBudget: budget,
      randomSeed: 2602
    });

    expect(counted.calls()).toBeLessThanOrEqual(budget);
    expect(model.diagnostics.totalOracleCalls).toBe(counted.calls());
    expect(model.diagnostics.budgetExhausted).toBe(true);
    expect(model.diagnostics.reconstructionComplete).toBe(false);
    expect(model.terms.every(term => term.readout === 0)).toBe(true);
  });

  it('switches to global exploration when observed hypotheses do not separate reliably', () => {
    const counted = countedOracle(input => input.reduce((sum, value) => sum + value * value, 0));
    const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
      dimension: 6,
      ranges: Array.from({ length: 6 }, () => [-1, 1]),
      termCount: 1,
      oracleBudget: 1200,
      maxGradientProbes: 18,
      randomSeed: 2603
    });

    expect(model.diagnostics.globalExplorationQueries).toBeGreaterThan(1);
    expect(model.diagnostics.strategySwitches).toBeGreaterThan(0);
    expect(model.diagnostics.reconstructionComplete).toBe(false);
    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
  });

  it('does not accept a composite gradient jump as a single hyperplane', () => {
    const teacher = (input: number[]): number =>
      0.8 * Math.max(0, 0.6 * input[0] + 0.8 * input[1] - 0.12)
      + 0.5 * Math.max(0, 0.3 * input[0] - 0.2 * input[1] + 0.9 * input[2] + 0.03);
    const counted = countedOracle(teacher);
    const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
      dimension: 6,
      ranges: Array.from({ length: 6 }, () => [-1, 1]),
      termCount: 1,
      oracleBudget: 2500,
      maxGradientProbes: 60,
      randomSeed: 2603
    });

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(model.diagnostics.candidatesRecovered).toBe(0);
    expect(model.diagnostics.candidateUncertainty).toBe(1);
    expect(model.diagnostics.reconstructionComplete).toBe(false);
  });

  it('does not claim a complete reconstruction when a localized smooth contaminant violates the affine model', () => {
    const normal = [0.6, 0.8, 0, 0, 0, 0];
    const teacher = (input: number[]): number => {
      const squaredDistance = input.reduce((sum, value, axis) =>
        sum + (value - (axis % 2 === 0 ? 0.15 : -0.1)) ** 2, 0);
      return 0.2 + 0.04 * input[0] - 0.03 * input[2]
        + 0.9 * Math.max(0, dot(normal, input) - 0.08)
        + 0.25 * Math.exp(-squaredDistance / (2 * 0.28 ** 2));
    };
    const counted = countedOracle(teacher);
    const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
      dimension: 6,
      ranges: Array.from({ length: 6 }, () => [-1, 1]),
      termCount: 1,
      oracleBudget: 2500,
      randomSeed: 2605
    });

    expect(counted.calls()).toBe(model.diagnostics.totalOracleCalls);
    expect(model.diagnostics.reconstructionComplete).toBe(false);
    expect(model.diagnostics.reasons).toContain('local_affine_model_residual_exceeded_confidence_gate');
  });

  it('estimates full-dimensional gradient variation without claiming a complete fit from weak evidence', () => {
    const dimension = 6;
    const counted = countedOracle(input =>
      input.reduce((sum, value, axis) => sum + (axis + 1) * Math.max(0, value + (axis - 2.5) * 0.03), 0));
    const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1, 1]),
      termCount: dimension,
      oracleBudget: 2500,
      randomSeed: 2604
    });

    expect(model.diagnostics.activeSubspaceRank).toBeGreaterThanOrEqual(2);
    expect(model.diagnostics.activeSubspaceRank).toBeLessThanOrEqual(dimension);
    expect(model.diagnostics.candidatesRecovered).toBeLessThanOrEqual(dimension);
    expect(model.diagnostics.totalOracleCalls).toBe(counted.calls());
    expect(model.diagnostics.reconstructionComplete).toBe(false);
  });
});
