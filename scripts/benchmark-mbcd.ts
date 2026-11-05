import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import {
  decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation,
  predictBlackBoxReluNdMultiHypothesisBoundaryContinuation,
  MbcdQueryPolicy,
  MbcdRankPolicy,
  MbcdValidationPolicy
} from '../src/solvers/multi-hypothesis-boundary-continuation';
import {
  decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery,
  predictBlackBoxReluNdBoundaryConditionedActiveDiscovery
} from '../src/solvers/boundary-conditioned-active-discovery';
import {
  decompileBlackBoxReluNdActiveSubspaceAdaptive,
  predictBlackBoxReluNdActiveSubspaceAdaptive
} from '../src/solvers/active-subspace-adaptive-discovery';
import {
  BlackBoxReluNdModel,
  decompileBlackBoxReluNdLineProbe,
  predictBlackBoxReluNdLineProbeModel
} from '../src/solvers/blackbox-relu-nd-line-probe';
import {
  createTeacher,
  makeScenario,
  mean,
  randomTestPoints,
  SCENARIOS as ASAD_SCENARIOS,
  Scenario,
  ScenarioName,
  solverBudget
} from './benchmark-asad';
import {
  ASAD_MAE_ABSOLUTE_FLOOR,
  maeValuesAreComparable
} from './asad-benchmark-metrics';

type MbcdMethod =
  | 'asad'
  | 'ransac'
  | 'star-probe'
  | 'adaptive'
  | 'bcad'
  | 'mbcd-full'
  | 'mbcd-single-rank'
  | 'mbcd-no-contradiction'
  | 'mbcd-no-independent-validation';

type MbcdRunRecord = {
  schemaVersion: 1;
  seed: number;
  dimension: number;
  scenario: ScenarioName;
  method: MbcdMethod;
  solverSeed: number;
  scenarioModelHash: string;
  evaluationPointsHash: string;
  status: 'complete' | 'incomplete' | 'threw';
  oracleBudget: number;
  actualOracleCalls: number;
  diagnosticOracleCalls: number | null;
  oracleCountReconciled: boolean;
  oracleBudgetExceeded: boolean;
  runtimeMs: number;
  mae: number | null;
  meanAngularErrorDeg: number | null;
  meanOffsetError: number | null;
  correctHyperplanes: number;
  falseAcceptedHyperplanes: number;
  missedHyperplanes: number;
  recoveredHyperplanes: number;
  activeRankEstimate: number | null;
  expectedActiveRank: number;
  activeRankMisestimated: boolean | null;
  reconstructionComplete: boolean;
  mbcdDiagnostics?: {
    queryPolicy: MbcdQueryPolicy;
    rankPolicy: MbcdRankPolicy;
    validationPolicy: MbcdValidationPolicy;
    totalOracleCalls: number;
    gradientProbes: number;
    rankResolved: boolean;
    plausibleActiveRanks: number[];
    rankUncertainty: number;
    boundaryHypotheses: number;
    boundaryDetectionCount: number;
    normalIdentifiedCount: number;
    offsetIdentifiedCount: number;
    uniqueHypothesisCount: number;
    acceptedBoundaryCount: number;
    unresolvedHypothesisCount: number;
    contradictionQueries: number;
    predictionDisagreementQueries: number;
    acquisitionQueries: number;
    boundaryValidationQueries: number;
    fitResidualRelative: number | null;
    reconstructionComplete: boolean;
  };
  error?: string;
};

type SummaryRow = {
  dimension: number;
  scenario: ScenarioName;
  method: MbcdMethod;
  runs: number;
  scoredRuns: number;
  completeRuns: number;
  incompleteRuns: number;
  thrownRuns: number;
  reconstructionCompleteRate: number;
  meanMae: number | null;
  meanOracleCalls: number;
  meanNormalErrorDeg: number | null;
  meanOffsetError: number | null;
  meanCorrectHyperplanes: number;
  meanFalseAcceptedHyperplanes: number;
  falseAcceptedRunRate: number | null;
  meanUnresolvedHypotheses: number | null;
  activeRankMisestimateRate: number | null;
  meanRuntimeMs: number;
};

type CellAssessment = {
  dimension: number;
  scenario: ScenarioName;
  bestBaseline: MbcdMethod | null;
  mbcdMae: number | null;
  bestBaselineMae: number | null;
  mbcdOracleCalls: number | null;
  bestComparableBaselineCalls: number | null;
  oracleCallReduction: number | null;
  maeReduction: number | null;
  mbcdReconstructionCompleteRate: number | null;
  bestBaselineReconstructionCompleteRate: number | null;
  mbcdFalseAcceptedRunRate: number | null;
  baselineFalseAcceptedRunRate: number | null;
  falseAcceptedIncrease: number | null;
  performancePass: boolean;
  falseAcceptancePass: boolean;
  qualifies: boolean;
  reason: string;
};

type CampaignSummary = {
  runId: string;
  seedsPerCell: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  methods: MbcdMethod[];
  evaluationPoints: number;
  totalRuns: number;
  totalOracleBudgetOverruns: number;
  totalDiagnosticMismatches: number;
  statusTotals: { complete: number; incomplete: number; threw: number };
  rawFile: string;
  summary: SummaryRow[];
  verdict: {
    verdict: 'promising' | 'partially-supported' | 'not-supported';
    primaryMethod: 'mbcd-full';
    primaryScenarioWins: ScenarioName[];
    requiredPrimaryScenarioWins: number;
    seedsPerCell: number;
    keyCells: CellAssessment[];
    budgetOverruns: number;
    falseAcceptanceTolerance: number;
    maeComparisonTolerance: number;
    absoluteMaeFloor: number;
  };
};

const DIMENSIONS = [16, 32];
const FOCUS_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank'
];
const METHODS: MbcdMethod[] = [
  'asad',
  'ransac',
  'star-probe',
  'adaptive',
  'bcad',
  'mbcd-full',
  'mbcd-single-rank',
  'mbcd-no-contradiction',
  'mbcd-no-independent-validation'
];
const EXISTING_METHODS: MbcdMethod[] = ['asad', 'ransac', 'star-probe', 'adaptive', 'bcad'];
const PRIMARY_METHOD = 'mbcd-full';
const PRIMARY_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank'
];
const EVALUATION_POINTS = 192;
const MAX_GRADIENT_PROBES = 80;
const FALSE_ACCEPTANCE_TOLERANCE = 0.01;
const MAE_COMPARISON_TOLERANCE = 0.05;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(args: string[]): {
  seeds: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  output: string;
} {
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index >= 0 ? args[index + 1] : undefined;
  };
  const seeds = Number(value('--seeds') ?? 100);
  if (!Number.isSafeInteger(seeds) || seeds < 1) {
    throw new Error('--seeds must be a positive safe integer');
  }
  const dimensions = (value('--dimensions') ?? DIMENSIONS.join(',')).split(',').map(Number);
  if (!dimensions.length || dimensions.some(dimension => !DIMENSIONS.includes(dimension))) {
    throw new Error(`--dimensions must be selected from ${DIMENSIONS.join(',')}`);
  }
  const scenarios = (value('--scenarios') ?? FOCUS_SCENARIOS.join(',')).split(',') as ScenarioName[];
  if (!scenarios.length || scenarios.some(scenario => !FOCUS_SCENARIOS.includes(scenario))) {
    throw new Error(`--scenarios must be selected from ${FOCUS_SCENARIOS.join(',')}`);
  }
  return {
    seeds,
    dimensions: [...new Set(dimensions)],
    scenarios: [...new Set(scenarios)],
    output: value('--output') ?? 'scratch/results/mbcd'
  };
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function angleDegrees(first: number[], second: number[]): number {
  const firstNorm = Math.sqrt(first.reduce((sum, value) => sum + value * value, 0));
  const secondNorm = Math.sqrt(second.reduce((sum, value) => sum + value * value, 0));
  if (firstNorm <= 1e-14 || secondNorm <= 1e-14) return 90;
  const similarity = Math.min(1, Math.max(-1,
    Math.abs(first.reduce((sum, value, axis) => sum + value * second[axis], 0) / (firstNorm * secondNorm))));
  return Math.acos(similarity) * 180 / Math.PI;
}

function offsetError(
  predicted: { normal: number[]; offset: number },
  truth: { normal: number[]; offset: number }
): number {
  const orientation = predicted.normal.reduce((sum, value, axis) =>
    sum + value * truth.normal[axis], 0) < 0 ? -1 : 1;
  const predictedNorm = Math.sqrt(predicted.normal.reduce((sum, value) => sum + value * value, 0));
  return Math.abs(orientation * predicted.offset / Math.max(1e-14, predictedNorm) - truth.offset);
}

function evaluateGeometry(
  predicted: Array<{ normal: number[]; offset: number }>,
  truth: Array<{ normal: number[]; offset: number }>
): {
  correct: number;
  falseAccepted: number;
  missed: number;
  meanAngle: number | null;
  meanOffset: number | null;
} {
  const pairs: Array<{ predictedIndex: number; truthIndex: number; angle: number; offset: number }> = [];
  for (let predictedIndex = 0; predictedIndex < predicted.length; predictedIndex++) {
    for (let truthIndex = 0; truthIndex < truth.length; truthIndex++) {
      pairs.push({
        predictedIndex,
        truthIndex,
        angle: angleDegrees(predicted[predictedIndex].normal, truth[truthIndex].normal),
        offset: offsetError(predicted[predictedIndex], truth[truthIndex])
      });
    }
  }
  pairs.sort((first, second) =>
    first.angle + 100 * first.offset - (second.angle + 100 * second.offset));
  const usedPredicted = new Set<number>();
  const usedTruth = new Set<number>();
  const matchedAngles: number[] = [];
  const matchedOffsets: number[] = [];
  for (const pair of pairs) {
    if (pair.angle > 10 || pair.offset > 0.1) continue;
    if (usedPredicted.has(pair.predictedIndex) || usedTruth.has(pair.truthIndex)) continue;
    usedPredicted.add(pair.predictedIndex);
    usedTruth.add(pair.truthIndex);
    matchedAngles.push(pair.angle);
    matchedOffsets.push(pair.offset);
  }
  return {
    correct: usedTruth.size,
    falseAccepted: predicted.length - usedPredicted.size,
    missed: truth.length - usedTruth.size,
    meanAngle: truth.length
        ? (matchedAngles.reduce((sum, value) => sum + value, 0)
          + 90 * (truth.length - matchedAngles.length)) / truth.length
        : null,
    meanOffset: matchedOffsets.length ? mean(matchedOffsets) : null
  };
}

function mbcdArm(method: MbcdMethod): {
  queryPolicy: MbcdQueryPolicy;
  rankPolicy: MbcdRankPolicy;
  validationPolicy: MbcdValidationPolicy;
} | null {
  switch (method) {
    case 'mbcd-full':
      return {
        queryPolicy: 'contradiction-seeking',
        rankPolicy: 'retain-alternatives',
        validationPolicy: 'independent'
      };
    case 'mbcd-single-rank':
      return {
        queryPolicy: 'contradiction-seeking',
        rankPolicy: 'single-best',
        validationPolicy: 'independent'
      };
    case 'mbcd-no-contradiction':
      return {
        queryPolicy: 'prediction-disagreement',
        rankPolicy: 'retain-alternatives',
        validationPolicy: 'independent'
      };
    case 'mbcd-no-independent-validation':
      return {
        queryPolicy: 'contradiction-seeking',
        rankPolicy: 'retain-alternatives',
        validationPolicy: 'support-only'
      };
    default:
      return null;
  }
}

function runOne(
  method: MbcdMethod,
  scenario: Scenario,
  dimension: number,
  seed: number
): MbcdRunRecord {
  const teacher = createTeacher(scenario);
  let actualOracleCalls = 0;
  const countedOracle = (input: number[]): number => {
    actualOracleCalls++;
    return teacher(input);
  };
  const { budget, lineCount } = solverBudget(dimension, scenario.terms.length);
  const solverSeed = seed * 104729
    + dimension * 13007
    + ASAD_SCENARIOS.indexOf(scenario.name) * 8191;
  const evaluationPoints = randomTestPoints(seed, dimension);
  const evaluationPointsHash = hash(evaluationPoints);
  const scenarioModelHash = hash(scenario);
  const start = performance.now();
  let reconstructionComplete = false;
  let diagnosticOracleCalls: number | null = null;
  let activeRankEstimate: number | null = null;
  let predictor: ((input: number[]) => number) | null = null;
  let recoveredPlanes: Array<{ normal: number[]; offset: number }> = [];
  let mbcdDiagnostics: MbcdRunRecord['mbcdDiagnostics'];
  let error: string | undefined;
  try {
    if (method === 'mbcd-full' || method === 'mbcd-single-rank'
      || method === 'mbcd-no-contradiction' || method === 'mbcd-no-independent-validation') {
      const arm = mbcdArm(method);
      if (!arm) throw new Error(`No MBCD configuration exists for ${method}`);
      const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: solverSeed,
        maxGradientProbes: MAX_GRADIENT_PROBES,
        ...arm
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      activeRankEstimate = model.diagnostics.estimatedActiveRank;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdMultiHypothesisBoundaryContinuation(model, input);
      mbcdDiagnostics = {
        queryPolicy: model.diagnostics.queryPolicy,
        rankPolicy: model.diagnostics.rankPolicy,
        validationPolicy: model.diagnostics.validationPolicy,
        totalOracleCalls: model.diagnostics.totalOracleCalls,
        gradientProbes: model.diagnostics.gradientProbes,
        rankResolved: model.diagnostics.rankResolved,
        plausibleActiveRanks: model.diagnostics.plausibleActiveRanks,
        rankUncertainty: model.diagnostics.rankUncertainty,
        boundaryHypotheses: model.diagnostics.boundaryHypotheses,
        boundaryDetectionCount: model.diagnostics.boundaryDetectionCount,
        normalIdentifiedCount: model.diagnostics.normalIdentifiedCount,
        offsetIdentifiedCount: model.diagnostics.offsetIdentifiedCount,
        uniqueHypothesisCount: model.diagnostics.uniqueHypothesisCount,
        acceptedBoundaryCount: model.diagnostics.acceptedBoundaryCount,
        unresolvedHypothesisCount: model.diagnostics.unresolvedHypothesisCount,
        contradictionQueries: model.diagnostics.contradictionQueries,
        predictionDisagreementQueries: model.diagnostics.predictionDisagreementQueries,
        acquisitionQueries: model.diagnostics.acquisitionQueries,
        boundaryValidationQueries: model.diagnostics.boundaryValidationQueries,
        fitResidualRelative: model.diagnostics.fitResidualRelative,
        reconstructionComplete: model.diagnostics.reconstructionComplete
      };
    } else if (method === 'bcad') {
      const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: solverSeed,
        maxGradientProbes: 96
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdBoundaryConditionedActiveDiscovery(model, input);
    } else if (method === 'asad') {
      const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: solverSeed,
        maxGradientProbes: 96
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      activeRankEstimate = model.diagnostics.activeSubspaceRank;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdActiveSubspaceAdaptive(model, input);
    } else {
      const model: BlackBoxReluNdModel = decompileBlackBoxReluNdLineProbe(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        stageAMethod: method,
        lineCount,
        samplesPerLine: 96,
        ransacIterations: 64,
        minSupport: Math.max(dimension + 1, Math.floor(lineCount * 0.15)),
        randomSeed: solverSeed,
        oracleBudget: budget
      });
      diagnosticOracleCalls = model.adaptiveDiagnostics?.totalOracleCalls ?? null;
      reconstructionComplete = model.reconstructionComplete ?? true;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdLineProbeModel(model, input);
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }
  const runtimeMs = performance.now() - start;
  const predictions = predictor ? evaluationPoints.map(predictor) : null;
  const mae = predictions === null
    ? null
    : mean(evaluationPoints.map((point, index) => Math.abs(teacher(point) - predictions[index])));
  const geometry = evaluateGeometry(recoveredPlanes, scenario.terms);
  const status = error
    ? 'threw'
    : reconstructionComplete
      ? 'complete'
      : 'incomplete';
  const expectedActiveRank = scenario.expectedActiveRank;
  return {
    schemaVersion: 1,
    seed,
    dimension,
    scenario: scenario.name,
    method,
    solverSeed,
    scenarioModelHash,
    evaluationPointsHash,
    status,
    oracleBudget: budget,
    actualOracleCalls,
    diagnosticOracleCalls,
    oracleCountReconciled: diagnosticOracleCalls === null
      ? status === 'threw'
      : diagnosticOracleCalls === actualOracleCalls,
    oracleBudgetExceeded: actualOracleCalls > budget,
    runtimeMs,
    mae,
    meanAngularErrorDeg: geometry.meanAngle,
    meanOffsetError: geometry.meanOffset,
    correctHyperplanes: geometry.correct,
    falseAcceptedHyperplanes: geometry.falseAccepted,
    missedHyperplanes: geometry.missed,
    recoveredHyperplanes: recoveredPlanes.length,
    activeRankEstimate,
    expectedActiveRank,
    activeRankMisestimated: activeRankEstimate === null
      ? null
      : activeRankEstimate !== expectedActiveRank,
    reconstructionComplete,
    ...(mbcdDiagnostics ? { mbcdDiagnostics } : {}),
    ...(error ? { error } : {})
  };
}

function summarize(records: MbcdRunRecord[]): SummaryRow[] {
  const groups = new Map<string, MbcdRunRecord[]>();
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    const group = groups.get(key) ?? [];
    group.push(record);
    groups.set(key, group);
  }
  return [...groups.values()].map(group => {
    const first = group[0];
    const scored = group.filter(record => record.mae !== null);
    const completeRuns = group.filter(record => record.status === 'complete').length;
    const thrownRuns = group.filter(record => record.status === 'threw').length;
    const falseAcceptedRunRate = scored.length
      ? scored.filter(record => record.falseAcceptedHyperplanes > 0).length / scored.length
      : null;
    const unresolved = group.filter(record => record.mbcdDiagnostics)
      .map(record => record.mbcdDiagnostics!.unresolvedHypothesisCount);
    const rankErrors = group.filter(record => record.activeRankMisestimated !== null);
    const rankMisestimateRate = rankErrors.length
      ? rankErrors.filter(record => record.activeRankMisestimated).length / rankErrors.length
      : null;
    const normalErrors = scored
      .map(record => record.meanAngularErrorDeg)
      .filter((value): value is number => value !== null);
    const offsetErrors = scored
      .map(record => record.meanOffsetError)
      .filter((value): value is number => value !== null);
    return {
      dimension: first.dimension,
      scenario: first.scenario,
      method: first.method,
      runs: group.length,
      scoredRuns: scored.length,
      completeRuns,
      incompleteRuns: group.length - completeRuns - thrownRuns,
      thrownRuns,
      reconstructionCompleteRate: completeRuns / group.length,
      meanMae: scored.length ? mean(scored.map(record => record.mae!)) : null,
      meanOracleCalls: mean(group.map(record => record.actualOracleCalls)),
      meanNormalErrorDeg: normalErrors.length ? mean(normalErrors) : null,
      meanOffsetError: offsetErrors.length ? mean(offsetErrors) : null,
      meanCorrectHyperplanes: mean(scored.map(record => record.correctHyperplanes)),
      meanFalseAcceptedHyperplanes: mean(scored.map(record => record.falseAcceptedHyperplanes)),
      falseAcceptedRunRate,
      meanUnresolvedHypotheses: unresolved.length ? mean(unresolved) : null,
      activeRankMisestimateRate: rankMisestimateRate,
      meanRuntimeMs: mean(group.map(record => record.runtimeMs))
    };
  }).sort((first, second) =>
    first.dimension - second.dimension
    || FOCUS_SCENARIOS.indexOf(first.scenario) - FOCUS_SCENARIOS.indexOf(second.scenario)
    || METHODS.indexOf(first.method) - METHODS.indexOf(second.method));
}

function assessCell(rows: SummaryRow[], dimension: number, scenario: ScenarioName): CellAssessment {
  const cellRows = rows.filter(row => row.dimension === dimension && row.scenario === scenario);
  const primary = cellRows.find(row => row.method === PRIMARY_METHOD);
  const baselines = cellRows.filter(row =>
    EXISTING_METHODS.includes(row.method) && row.meanMae !== null && row.scoredRuns > 0);
  const rankedBaselines = [...baselines].sort((first, second) => first.meanMae! - second.meanMae!);
  const best = rankedBaselines[0] ?? null;
  if (!primary || primary.meanMae === null || !best || best.meanMae === null) {
    return {
      dimension,
      scenario,
      bestBaseline: best?.method ?? null,
      mbcdMae: primary?.meanMae ?? null,
      bestBaselineMae: best?.meanMae ?? null,
      mbcdOracleCalls: primary?.meanOracleCalls ?? null,
      bestComparableBaselineCalls: null,
      oracleCallReduction: null,
      maeReduction: null,
      mbcdReconstructionCompleteRate: primary?.reconstructionCompleteRate ?? null,
      bestBaselineReconstructionCompleteRate: best?.reconstructionCompleteRate ?? null,
      mbcdFalseAcceptedRunRate: primary?.falseAcceptedRunRate ?? null,
      baselineFalseAcceptedRunRate: best?.falseAcceptedRunRate ?? null,
      falseAcceptedIncrease: null,
      performancePass: false,
      falseAcceptancePass: false,
      qualifies: false,
      reason: 'missing_scored_primary_method_or_baseline'
    };
  }
  const comparable = baselines.filter(row => maeValuesAreComparable(row.meanMae!, best.meanMae!));
  const bestComparableCalls = comparable.length
    ? Math.min(...comparable.map(row => row.meanOracleCalls))
    : null;
  const oracleCallReduction = bestComparableCalls !== null && bestComparableCalls > 0
    ? 1 - primary.meanOracleCalls / bestComparableCalls
    : null;
  const maeReduction = best.meanMae > ASAD_MAE_ABSOLUTE_FLOOR
    ? 1 - primary.meanMae / best.meanMae
    : null;
  const maeImprovementPass = maeReduction !== null && maeReduction >= 0.25;
  const efficiencyPass = oracleCallReduction !== null
    && oracleCallReduction >= 0.3
    && primary.meanMae <= best.meanMae * (1 + MAE_COMPARISON_TOLERANCE)
      + ASAD_MAE_ABSOLUTE_FLOOR;
  const referenceFalseRates = comparable
    .map(row => row.falseAcceptedRunRate)
    .filter((value): value is number => value !== null);
  const baselineFalseRate = referenceFalseRates.length
    ? Math.min(...referenceFalseRates)
    : null;
  const falseAcceptedIncrease = primary.falseAcceptedRunRate !== null && baselineFalseRate !== null
    ? primary.falseAcceptedRunRate - baselineFalseRate
    : null;
  const falseAcceptancePass = falseAcceptedIncrease !== null
    && falseAcceptedIncrease <= FALSE_ACCEPTANCE_TOLERANCE;
  const performancePass = maeImprovementPass || efficiencyPass;
  const qualifies = performancePass && falseAcceptancePass;
  const reason = qualifies
    ? maeImprovementPass
      ? 'mae_improved_by_at_least_25_percent'
      : 'at_least_30_percent_fewer_calls_with_mae_within_5_percent'
    : !performancePass
      ? 'neither_predeclared_performance_threshold_met'
      : 'false_accepted_run_rate_increased_by_more_than_1_percentage_point';
  return {
    dimension,
    scenario,
    bestBaseline: best.method,
    mbcdMae: primary.meanMae,
    bestBaselineMae: best.meanMae,
    mbcdOracleCalls: primary.meanOracleCalls,
    bestComparableBaselineCalls: bestComparableCalls,
    oracleCallReduction,
    maeReduction,
    mbcdReconstructionCompleteRate: primary.reconstructionCompleteRate,
    bestBaselineReconstructionCompleteRate: best.reconstructionCompleteRate,
    mbcdFalseAcceptedRunRate: primary.falseAcceptedRunRate,
    baselineFalseAcceptedRunRate: baselineFalseRate,
    falseAcceptedIncrease,
    performancePass,
    falseAcceptancePass,
    qualifies,
    reason
  };
}

function renderReport(summary: CampaignSummary): string {
  const primaryCells = summary.verdict.keyCells;
  const rows = summary.summary;
  const total = summary.statusTotals;
  const methodRows = rows;
  const metricRows = methodRows.map(row =>
    `| ${row.dimension} | ${row.scenario} | ${row.method} | ${row.scoredRuns}/${row.runs} | ${format(row.meanMae)} | ${format(row.meanOracleCalls)} | ${formatPercent(row.reconstructionCompleteRate)} | ${format(row.meanNormalErrorDeg)} | ${format(row.meanOffsetError)} | ${formatPercent(row.falseAcceptedRunRate)} | ${format(row.meanUnresolvedHypotheses)} | ${formatPercent(row.activeRankMisestimateRate)} | ${format(row.meanRuntimeMs)} | ${row.thrownRuns} |`
  ).join('\n');
  return `# Multi-Hypothesis Boundary Continuation and Disambiguation (MBCD): Focused Experiment

**Run:** ${summary.runId}  
**Verdict:** ${summary.verdict.verdict}  
**Dimensions:** ${summary.dimensions.join(', ')}  
**Scenarios:** ${summary.scenarios.join(', ')}  
**Methods/ablations:** ${summary.methods.join(', ')}  
**Seeds per cell:** ${summary.seedsPerCell}  
**Raw observations:** [${summary.rawFile}](../../scratch/results/mbcd/${summary.rawFile})

## Research question and implementation

MBCD keeps multiple oracle-derived active-rank and boundary hypotheses rather than promoting the current best candidate into a settled explanation. A candidate boundary tracks detection, normal identification, offset identification, uniqueness, and later validation separately. The contradiction-seeking policy queries where candidate planes disagree about which side contains a point. The query is one full finite-difference gradient stencil, costing exactly $2d+1$ oracle calls; no partial stencil starts when the remaining budget is smaller. Model fitting and scoring make no additional oracle calls.

The solver receives only the oracle, dimension, ranges, requested term count, seed, and budget. Teacher metadata and the shared evaluation set are used only by the benchmark after reconstruction. Existing ASAD, BCAD, RANSAC, Star-Probe, and Adaptive implementations are called without modification.

## Paired design and predeclared criteria

Each seed/dimension/scenario uses the same deterministic teacher, solver seed, solverBudget result, and 192 independently generated evaluation points for every method. The five focus scenarios are near-parallel facets, multiple crossings, small-amplitude jumps, contaminated local probes, and orthogonal full-rank controls at 16D and 32D. MBCD ablations remove rank alternatives, contradiction-seeking acquisition, or independent boundary validation one component at a time.

A primary cell qualifies if full MBCD achieves at least 25% lower MAE than the best scored existing method, or uses at least 30% fewer calls than an existing method within 5% MAE of the best baseline while staying within 5% of that best MAE. Its false-accepted-run rate must be no more than one percentage point above the lowest such comparable baseline rate. At least two of the three primary scenario classes must qualify at both dimensions. There are ${summary.seedsPerCell} seeds per cell and the actual oracle-call total must never exceed the common budget. The verdict is not derived from test pass status.

## Primary comparison

| D | Scenario | Best existing method | MBCD MAE | Baseline MAE | Call reduction | MAE reduction | MBCD complete rate | False-accepted increase reference | Qualifies | Reason |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
${primaryCells.map(cell =>
  `| ${cell.dimension} | ${cell.scenario} | ${cell.bestBaseline ?? 'n/a'} | ${format(cell.mbcdMae)} | ${format(cell.bestBaselineMae)} | ${formatPercent(cell.oracleCallReduction)} | ${formatPercent(cell.maeReduction)} | ${formatPercent(cell.mbcdReconstructionCompleteRate)} | ${formatPercent(cell.falseAcceptedIncrease)} | ${cell.qualifies ? 'yes' : 'no'} | ${cell.reason} |`
).join('\n')}

**Primary scenario classes qualifying at both dimensions:** ${summary.verdict.primaryScenarioWins.length}/${PRIMARY_SCENARIOS.length} (${summary.verdict.primaryScenarioWins.join(', ') || 'none'}).  
**Actual oracle-budget overruns:** ${summary.verdict.budgetOverruns}.  
**Run statuses:** ${total.complete} complete, ${total.incomplete} incomplete, ${total.threw} threw.

## Method metrics (all dimensions and focus scenarios)

| D | Scenario | Method | Scored/runs | MAE | Oracle calls | Complete rate | Normal error (deg) | Offset error | False-accepted run rate | Mean unresolved hypotheses | Rank misestimate rate | Runtime (ms) | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${metricRows}

## Integrity and limitations

The independent verifier recomputes paired fixture hashes, evaluation-set hashes, seeds, budgets, actual-to-diagnostic oracle counts, completion rates, false-accepted rates, and the formal verdict from raw records. A method that throws remains unscored rather than receiving a synthetic MAE. Mean normal and offset error are calculated only for geometrically matched planes; misses and false accepts are reported separately. MBCD's reconstruction-complete rate is shown so a low MAE on incomplete outputs cannot be mistaken for reliable geometric recovery.
`;
}

function format(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? 'n/a'
    : Number(value.toPrecision(5)).toString();
}

function formatPercent(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? 'n/a'
    : `${(100 * value).toFixed(2)}%`;
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const projectRoot = path.resolve(__dirname, '..');
  const outputDirectory = path.resolve(projectRoot, options.output);
  fs.mkdirSync(outputDirectory, { recursive: true });
  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const rawFile = `mbcd-${runId}-raw.jsonl`;
  const rawPath = path.join(outputDirectory, rawFile);
  const records: MbcdRunRecord[] = [];
  const totalRuns = options.seeds * options.dimensions.length * options.scenarios.length * METHODS.length;
  let completedRuns = 0;
  for (const dimension of options.dimensions) {
    for (const scenarioName of options.scenarios) {
      for (let seed = 1; seed <= options.seeds; seed++) {
        const scenario = makeScenario(scenarioName, dimension, seed);
        for (const method of METHODS) {
          const record = runOne(method, scenario, dimension, seed);
          records.push(record);
          fs.appendFileSync(rawPath, `${JSON.stringify(record)}\n`);
          completedRuns++;
          if (completedRuns % 100 === 0 || completedRuns === totalRuns) {
            process.stdout.write(`MBCD_PROGRESS ${completedRuns}/${totalRuns} dimension=${dimension} scenario=${scenarioName} seed=${seed} method=${method}\n`);
          }
        }
      }
    }
  }
  const summaryRows = summarize(records);
  const keyCells = options.dimensions.flatMap(dimension =>
    options.scenarios.filter(scenario => PRIMARY_SCENARIOS.includes(scenario))
      .map(scenario => assessCell(summaryRows, dimension, scenario)));
  const primaryScenarioWins = PRIMARY_SCENARIOS.filter(scenario =>
    options.dimensions.includes(16)
    && options.dimensions.includes(32)
    && [16, 32].every(dimension =>
      keyCells.some(cell => cell.dimension === dimension
        && cell.scenario === scenario && cell.qualifies)));
  const totalOracleBudgetOverruns = records.filter(record => record.oracleBudgetExceeded).length;
  const totalDiagnosticMismatches = records.filter(record =>
    record.diagnosticOracleCalls !== null && !record.oracleCountReconciled).length;
  const verdict: CampaignSummary['verdict']['verdict'] = totalOracleBudgetOverruns > 0
    ? 'not-supported'
    : primaryScenarioWins.length >= 2
      ? 'promising'
      : primaryScenarioWins.length === 1
        ? 'partially-supported'
        : 'not-supported';
  const summary: CampaignSummary = {
    runId,
    seedsPerCell: options.seeds,
    dimensions: options.dimensions,
    scenarios: options.scenarios,
    methods: METHODS,
    evaluationPoints: EVALUATION_POINTS,
    totalRuns: records.length,
    totalOracleBudgetOverruns,
    totalDiagnosticMismatches,
    statusTotals: {
      complete: records.filter(record => record.status === 'complete').length,
      incomplete: records.filter(record => record.status === 'incomplete').length,
      threw: records.filter(record => record.status === 'threw').length
    },
    rawFile,
    summary: summaryRows,
    verdict: {
      verdict,
      primaryMethod: PRIMARY_METHOD,
      primaryScenarioWins,
      requiredPrimaryScenarioWins: 2,
      seedsPerCell: options.seeds,
      keyCells,
      budgetOverruns: totalOracleBudgetOverruns,
      falseAcceptanceTolerance: FALSE_ACCEPTANCE_TOLERANCE,
      maeComparisonTolerance: MAE_COMPARISON_TOLERANCE,
      absoluteMaeFloor: ASAD_MAE_ABSOLUTE_FLOOR
    }
  };
  const summaryPath = path.join(outputDirectory, `summary-${runId}.json`);
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  const reportPath = path.join(projectRoot, 'docs/research', `${runId.slice(0, 10)}-mbcd-experimental-report.md`);
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, renderReport(summary));
  console.log(`MBCD_CAMPAIGN_COMPLETE run=${runId} runs=${records.length} seeds=${options.seeds} verdict=${verdict} budgetOverruns=${totalOracleBudgetOverruns} diagnosticMismatches=${totalDiagnosticMismatches}`);
}

main();
