import {
  decompileBlackBoxReluNdLineProbe,
  predictBlackBoxReluNdLineProbeBatch,
  probeKinkNormalStar
} from '../src/solvers/blackbox-relu-nd-line-probe';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function countingOracle(oracle: (input: number[]) => number): { oracle: (input: number[]) => number; calls: () => number } {
  let count = 0;
  return {
    oracle: input => {
      count++;
      return oracle(input);
    },
    calls: () => count
  };
}

describe('LSPNR Star-Probe Stage A in blackbox-relu-nd-line-probe', () => {
  it('recovers isolated 3D hyperplane using star-probe stage A with high precision', () => {
    const normal = [0.8, -0.6, 0.0];
    const offset = 0.3;
    const readout = 1.25;
    const teacher = (x: number[]): number => 0.5 + 0.2 * x[0] + readout * Math.max(0, dot(normal, x) + offset);

    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 3,
      ranges: [[-2, 2], [-2, 2], [-2, 2]],
      termCount: 1,
      stageAMethod: 'star-probe',
      lineCount: 20,
      samplesPerLine: 121,
      randomSeed: 42
    });

    expect(model.terms).toHaveLength(1);
    const recoveredNormal = model.terms[0].normal;
    // Check cosine similarity
    const cosSim = Math.abs(dot(recoveredNormal, [0.8, -0.6, 0.0]));
    expect(cosSim).toBeGreaterThan(0.999);
    expect(Math.abs(model.terms[0].readout - readout)).toBeLessThan(1e-4);
  });

  it('recovers multi-neuron model using hybrid star-probe stage A with RANSAC fallback', () => {
    const teacherTerms = [
      { normal: [1, 0, 0, 0], offset: 0.2, readout: 1.1 },
      { normal: [0, 1, 0, 0], offset: -0.3, readout: -0.9 },
      { normal: [0, 0, 1, 0], offset: 0.4, readout: 0.8 }
    ];
    const teacher = (x: number[]): number => (
      0.1 + teacherTerms.reduce((sum, t) => sum + t.readout * Math.max(0, dot(t.normal, x) + t.offset), 0)
    );

    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 4,
      ranges: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]],
      termCount: 3,
      stageAMethod: 'hybrid',
      lineCount: 60,
      samplesPerLine: 201,
      randomSeed: 101
    });

    expect(model.terms).toHaveLength(3);
    const testPoints = [
      [0.5, -0.2, 0.1, 0.0],
      [-0.4, 0.6, -0.5, 0.2],
      [0.0, 0.0, 0.0, 0.0]
    ];
    const actual = testPoints.map(teacher);
    const pred = predictBlackBoxReluNdLineProbeBatch(model, testPoints);
    const metrics = regressionMetrics(actual, pred);
    expect(metrics.mae).toBeLessThan(1e-4);
  });

  it('correctly handles duplicate collinear neurons as a single aggregate hyperplane', () => {
    // Two neurons with same normal and bias, readouts 0.7 and 0.8 -> net readout 1.5
    const normal = [0.6, 0.8];
    const offset = -0.2;
    const teacher = (x: number[]): number => (
      0.0 + 0.7 * Math.max(0, dot(normal, x) + offset) + 0.8 * Math.max(0, dot(normal, x) + offset)
    );

    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 1,
      stageAMethod: 'star-probe',
      lineCount: 15,
      samplesPerLine: 101,
      randomSeed: 202
    });

    expect(model.terms).toHaveLength(1);
    expect(Math.abs(dot(model.terms[0].normal, normal))).toBeGreaterThan(0.999);
    // Net readout must equal 0.7 + 0.8 = 1.5
    expect(Math.abs(model.terms[0].readout - 1.5)).toBeLessThan(1e-3);
  });

  it('rejects kink candidates with sub-threshold amplitude in star probe', () => {
    // Tiny amplitude neuron
    const normal = [1, 0];
    const offset = 0.0;
    const readout = 0.01; // below default minJump 0.05
    const teacher = (x: number[]): number => readout * Math.max(0, dot(normal, x) + offset);

    const kink = { point: [0, 0], deltaSlope: 0.01 };
    const cand = probeKinkNormalStar(teacher, kink, [1, 0], 0.015, 1e-4, 0.05, 1e-4);
    // Should be rejected because jumpNorm (0.01) < minJump (0.05)
    expect(cand).toBeNull();
  });

  it('rejects contaminated multi-boundary kinks failing affine consistency', () => {
    // Sharp non-linear quadratic function mimicking multi-boundary interference
    const nonLinearOracle = (x: number[]): number => x[0] * x[0] + x[1] * x[1];
    const kink = { point: [0.1, 0.1], deltaSlope: 0.5 };
    const cand = probeKinkNormalStar(nonLinearOracle, kink, [1, 0], 0.015, 1e-3, 0.05, 1e-6);
    expect(cand).toBeNull();
  });

  it('rejects multi-boundary contaminated kink violating directional projection consistency', () => {
    // Two orthogonal planes: H1 (x0=0) and H2 (x1=0)
    // Oracle has two active components across the corner
    const twoPlaneOracle = (x: number[]): number => Math.max(0, x[0]) + Math.max(0, x[1]);
    // A kink at [0, 0] with an artificial mismatch in 1D deltaSlope (0.2 vs true jump ~ 1.41)
    const contaminatedKink = { point: [0, 0], deltaSlope: 0.2 };
    // Probe along diagonal direction [1/sqrt(2), 1/sqrt(2)]
    const diag = [1 / Math.SQRT2, 1 / Math.SQRT2];
    const cand = probeKinkNormalStar(twoPlaneOracle, contaminatedKink, diag, 0.015, 1e-4, 0.05, 1e-4);
    // Projection consistency should reject this contaminated jump
    expect(cand).toBeNull();
  });

  it('prevents RANSAC fallback masking on intersecting boundaries in hybrid mode', () => {
    // Model with two orthogonal hyperplanes intersecting at the origin
    const teacher = (x: number[]): number => (
      0.1 + 1.2 * Math.max(0, x[0]) + 0.9 * Math.max(0, x[1])
    );

    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 2,
      stageAMethod: 'hybrid',
      lineCount: 40,
      samplesPerLine: 121,
      randomSeed: 303
    });

    expect(model.terms).toHaveLength(2);
    // Both orthogonal normals must be recovered (one along x0, one along x1)
    const normals = model.terms.map(t => t.normal);
    const hasX0 = normals.some(n => Math.abs(dot(n, [1, 0])) > 0.98);
    const hasX1 = normals.some(n => Math.abs(dot(n, [0, 1])) > 0.98);
    expect(hasX0).toBe(true);
    expect(hasX1).toBe(true);
  });

  it('adaptive mode selects star-probe for clean isolated model and populates diagnostics', () => {
    const normal = [0.8, -0.6, 0.0];
    const offset = 0.3;
    const readout = 1.25;
    const teacher = (x: number[]): number => 0.5 + 0.2 * x[0] + readout * Math.max(0, dot(normal, x) + offset);

    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 3,
      ranges: [[-2, 2], [-2, 2], [-2, 2]],
      termCount: 1,
      stageAMethod: 'adaptive',
      lineCount: 20,
      samplesPerLine: 121,
      randomSeed: 404
    });

    expect(model.terms).toHaveLength(1);
    expect(model.adaptiveDiagnostics).toBeDefined();
    expect(model.adaptiveDiagnostics?.methodSelected).toBe('star-probe');
    expect(model.adaptiveDiagnostics?.ransacFallbackTriggered).toBe(false);
    expect(model.adaptiveDiagnostics?.reasons).toContain('all_terms_recovered_via_star_probe_with_verified_support');
    expect(model.adaptiveDiagnostics?.totalOracleCalls).toBeGreaterThan(0);
  });

  it('adaptive mode triggers RANSAC fallback on sub-threshold amplitude with explicit diagnostics', () => {
    // Model with 1 regular neuron and 1 small amplitude neuron (readout 0.02)
    const teacher = (x: number[]): number => (
      0.1 + 1.1 * Math.max(0, x[0] - 0.2) + 0.02 * Math.max(0, x[1] + 0.2)
    );

    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 2,
      stageAMethod: 'adaptive',
      lineCount: 50,
      samplesPerLine: 151,
      randomSeed: 505
    });

    expect(model.terms).toHaveLength(2);
    expect(model.adaptiveDiagnostics).toBeDefined();
    expect(model.adaptiveDiagnostics?.ransacFallbackTriggered).toBe(true);
    expect(model.adaptiveDiagnostics?.reasons.some(r => r.includes('fallback_to_ransac'))).toBe(true);
  });

  it('adaptive mode respects oracle budget cap', () => {
    const teacher = (x: number[]): number => Math.max(0, x[0]);
    // The tight cap allows collection to begin but interrupts later oracle work.
    const model = decompileBlackBoxReluNdLineProbe(teacher, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 1,
      stageAMethod: 'adaptive',
      lineCount: 15,
      samplesPerLine: 101,
      minSupport: 3,
      oracleBudget: 1520, // lines use 15 * 101 = 1515, leaving only 5 queries (star probe needs 10)
      randomSeed: 606
    });

    expect(model.terms).toHaveLength(1);
    expect(model.adaptiveDiagnostics).toBeDefined();
    expect(model.adaptiveDiagnostics?.reasons).toContain('budget_limit_triggered_fallback_to_ransac');
  });

  it.each([0, 1])('enforces a total oracle budget of %i from the first call', budget => {
    const counted = countingOracle((x: number[]) => Math.max(0, x[0]));
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 1,
      stageAMethod: 'adaptive',
      oracleBudget: budget,
      lineCount: 12,
      samplesPerLine: 96,
      randomSeed: 701
    });

    expect(counted.calls()).toBeLessThanOrEqual(budget);
    expect(model.adaptiveDiagnostics?.totalOracleCalls).toBe(counted.calls());
    expect(model.adaptiveDiagnostics?.budgetConsumed).toBe(counted.calls());
    expect(model.adaptiveDiagnostics?.budgetLimit).toBe(budget);
    expect(model.adaptiveDiagnostics?.budgetExhausted).toBe(true);
    expect(model.reconstructionComplete).toBe(false);
  });

  it('stops initial kink collection at the global budget and reports the measured calls', () => {
    const budget = 23;
    const counted = countingOracle((x: number[]) => Math.max(0, x[0] + 0.2));
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 1,
      stageAMethod: 'adaptive',
      oracleBudget: budget,
      lineCount: 20,
      samplesPerLine: 96,
      randomSeed: 702
    });

    expect(counted.calls()).toBe(budget);
    expect(model.adaptiveDiagnostics?.totalOracleCalls).toBe(counted.calls());
    expect(model.adaptiveDiagnostics?.oracleCallsByPhase['kink-collection']).toBe(budget);
    expect(model.adaptiveDiagnostics?.stoppedPhase).toBe('kink-collection');
    expect(model.reconstructionComplete).toBe(false);
  });

  it('preserves kink candidates from completed lines when collection exhausts the budget', () => {
    const budget = 96 * 8 + 23;
    const counted = countingOracle((x: number[]) => 10 * Math.max(0, x[0] + 0.1));
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 1,
      stageAMethod: 'adaptive',
      oracleBudget: budget,
      lineCount: 16,
      samplesPerLine: 96,
      adaptiveMaxStarProbes: 0,
      randomSeed: 709
    });

    expect(counted.calls()).toBe(budget);
    expect(model.kinkPointCount).toBeGreaterThan(0);
    expect(model.adaptiveDiagnostics?.oracleCallsByPhase['kink-collection']).toBe(budget);
    expect(model.adaptiveDiagnostics?.totalOracleCalls).toBe(counted.calls());
    expect(model.reconstructionComplete).toBe(false);
  });

  it('stops a partially executed star-probe stencil without exceeding the budget', () => {
    const baseConfig = {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]] as Array<[number, number]>,
      termCount: 1,
      stageAMethod: 'adaptive' as const,
      lineCount: 24,
      samplesPerLine: 101,
      minSupport: 3,
      adaptiveNormalConsistencyRatio: 0,
      randomSeed: 703
    };
    const teacher = (x: number[]): number => Math.max(0, x[0] + 0.1);
    const reference = decompileBlackBoxReluNdLineProbe(teacher, {
      ...baseConfig,
      oracleBudget: 100_000,
      adaptiveMaxStarProbes: 1
    });
    const collectionCalls = reference.adaptiveDiagnostics?.oracleCallsByPhase['kink-collection'] ?? 0;
    const budget = collectionCalls + 2;
    const counted = countingOracle(teacher);
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      ...baseConfig,
      oracleBudget: budget,
      adaptiveMaxStarProbes: 1
    });

    expect(counted.calls()).toBe(budget);
    expect(model.adaptiveDiagnostics?.stoppedPhase).toBe('star-probe');
    expect(model.adaptiveDiagnostics?.starProbesExecuted).toBe(1);
    expect(model.adaptiveDiagnostics?.starProbeOracleCalls).toBe(2);
    expect(model.adaptiveDiagnostics?.totalOracleCalls).toBe(counted.calls());
    expect(model.reconstructionComplete).toBe(false);
  });

  it('runs RANSAC fallback without bypassing the budget and marks the unfitted result incomplete', () => {
    const config = {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]] as Array<[number, number]>,
      termCount: 1,
      stageAMethod: 'adaptive' as const,
      lineCount: 24,
      samplesPerLine: 101,
      minSupport: 3,
      randomSeed: 704,
      adaptiveMaxStarProbes: 0
    };
    const teacher = (x: number[]): number => Math.max(0, x[0] - 0.2);
    const reference = decompileBlackBoxReluNdLineProbe(teacher, {
      ...config,
      oracleBudget: 100_000
    });
    const budget = reference.adaptiveDiagnostics?.oracleCallsByPhase['kink-collection'] ?? 0;
    const counted = countingOracle(teacher);
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      ...config,
      oracleBudget: budget
    });

    expect(counted.calls()).toBe(budget);
    expect(model.adaptiveDiagnostics?.ransacFallbackTriggered).toBe(true);
    expect(model.adaptiveDiagnostics?.starProbesExecuted).toBe(0);
    expect(model.adaptiveDiagnostics?.starProbeOracleCalls).toBe(0);
    expect(model.adaptiveDiagnostics?.stoppedPhase).toBe('fit-sampling');
    expect(model.adaptiveDiagnostics?.totalOracleCalls).toBe(counted.calls());
    expect(model.reconstructionComplete).toBe(false);
  });

  it('stops during final fit sampling and reconciles every measured oracle call', () => {
    const config = {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]] as Array<[number, number]>,
      termCount: 1,
      stageAMethod: 'ransac' as const,
      lineCount: 24,
      samplesPerLine: 101,
      minSupport: 3,
      randomSeed: 705
    };
    const teacher = (x: number[]): number => 0.25 + 0.4 * x[1] + 1.2 * Math.max(0, x[0] + 0.15);
    const reference = decompileBlackBoxReluNdLineProbe(teacher, {
      ...config,
      oracleBudget: 100_000
    });
    const collectionCalls = reference.adaptiveDiagnostics?.oracleCallsByPhase['kink-collection'] ?? 0;
    const budget = collectionCalls + 3;
    const counted = countingOracle(teacher);
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      ...config,
      oracleBudget: budget
    });
    const diagnostics = model.adaptiveDiagnostics;

    expect(counted.calls()).toBe(budget);
    expect(diagnostics?.stoppedPhase).toBe('fit-sampling');
    expect(diagnostics?.oracleCallsByPhase['kink-collection']).toBe(collectionCalls);
    expect(diagnostics?.oracleCallsByPhase['fit-sampling']).toBe(3);
    expect(Object.values(diagnostics?.oracleCallsByPhase ?? {}).reduce((sum, calls) => sum + calls, 0))
      .toBe(counted.calls());
    expect(diagnostics?.totalOracleCalls).toBe(counted.calls());
    expect(diagnostics?.candidatesRecovered).toBe(model.terms.length);
    expect(model.reconstructionComplete).toBe(false);
  });

  it('counts all oracle calls by phase and matches the mock oracle total', () => {
    const counted = countingOracle((x: number[]) => (
      0.1 + Math.max(0, x[0] - 0.2) + 0.8 * Math.max(0, x[1] + 0.3)
    ));
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 2,
      stageAMethod: 'adaptive',
      oracleBudget: 100_000,
      lineCount: 40,
      samplesPerLine: 121,
      minSupport: 3,
      randomSeed: 706
    });
    const diagnostics = model.adaptiveDiagnostics;

    expect(counted.calls()).toBe(diagnostics?.totalOracleCalls);
    expect(counted.calls()).toBe(diagnostics?.budgetConsumed);
    expect(Object.values(diagnostics?.oracleCallsByPhase ?? {}).reduce((sum, calls) => sum + calls, 0))
      .toBe(counted.calls());
    expect(diagnostics?.starProbeOracleCalls).toBe(diagnostics?.oracleCallsByPhase['star-probe']);
  });

  it('executes no star-probe candidates when adaptiveMaxStarProbes is zero', () => {
    const counted = countingOracle((x: number[]) => Math.max(0, x[0]) + 0.7 * Math.max(0, x[1] - 0.1));
    const model = decompileBlackBoxReluNdLineProbe(counted.oracle, {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]],
      termCount: 2,
      stageAMethod: 'adaptive',
      adaptiveMaxStarProbes: 0,
      adaptiveNormalConsistencyRatio: 0,
      oracleBudget: 100_000,
      lineCount: 40,
      samplesPerLine: 121,
      minSupport: 3,
      randomSeed: 707
    });

    expect(model.adaptiveDiagnostics?.starProbesExecuted).toBe(0);
    expect(model.adaptiveDiagnostics?.starProbeOracleCalls).toBe(0);
    expect(model.adaptiveDiagnostics?.maxStarProbes).toBe(0);
  });

  it('limits primary star probes and lets a higher limit execute more when budget permits', () => {
    const teacher = (x: number[]): number => (
      0.1 + 1.1 * Math.max(0, x[0] - 0.2) + 0.8 * Math.max(0, x[1] + 0.15)
    );
    const config = {
      dimension: 2,
      ranges: [[-1.5, 1.5], [-1.5, 1.5]] as Array<[number, number]>,
      termCount: 2,
      stageAMethod: 'adaptive' as const,
      adaptiveNormalConsistencyRatio: 0,
      oracleBudget: 100_000,
      lineCount: 40,
      samplesPerLine: 121,
      minSupport: 3,
      randomSeed: 708
    };
    const oneProbe = decompileBlackBoxReluNdLineProbe(teacher, {
      ...config,
      adaptiveMaxStarProbes: 1
    });
    const moreProbes = decompileBlackBoxReluNdLineProbe(teacher, {
      ...config,
      adaptiveMaxStarProbes: 8
    });

    expect(oneProbe.adaptiveDiagnostics?.starProbesExecuted).toBe(1);
    expect(oneProbe.adaptiveDiagnostics?.maxStarProbes).toBe(1);
    expect(oneProbe.adaptiveDiagnostics?.starProbeOracleCalls).toBeGreaterThan(
      oneProbe.adaptiveDiagnostics?.starProbesExecuted ?? 0
    );
    expect(moreProbes.adaptiveDiagnostics?.starProbesExecuted).toBeGreaterThan(1);
    expect(moreProbes.adaptiveDiagnostics?.starProbesExecuted).toBeLessThanOrEqual(8);
    expect(moreProbes.adaptiveDiagnostics?.budgetExhausted).toBe(false);
  });
});
