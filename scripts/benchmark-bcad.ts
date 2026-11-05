import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import {
  decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery,
  predictBlackBoxReluNdBoundaryConditionedActiveDiscovery,
  BcadAcquisitionStrategy,
  BcadRankStrategy
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
  matchPlanes,
  randomTestPoints,
  SCENARIOS,
  Scenario,
  ScenarioName,
  solverBudget
} from './benchmark-asad';
import {
  ASAD_MAE_ABSOLUTE_FLOOR,
  ASAD_MAE_COMPARISON_TOLERANCE,
  maeValuesAreComparable,
  oracleCallReduction
} from './asad-benchmark-metrics';

type BcadMethod =
  | 'asad'
  | 'ransac'
  | 'star-probe'
  | 'adaptive'
  | 'bcad-boundary-delayed'
  | 'bcad-prediction-control'
  | 'bcad-gradient-jump'
  | 'bcad-full-dimensional'
  | 'bcad-fixed-initial-rank'
  | 'bcad-stable-adaptive-rank'
  | 'bcad-single-hypothesis-acquisition';

type BcadRunRecord = {
  schemaVersion: 1;
  seed: number;
  dimension: number;
  scenario: ScenarioName;
  method: BcadMethod;
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
  correctHyperplanes: number;
  falseAcceptedHyperplanes: number;
  missedHyperplanes: number;
  recoveredHyperplanes: number;
  activeRankEstimate: number | null;
  expectedActiveRank: number;
  activeRankMisestimated: boolean | null;
  reconstructionComplete: boolean;
  bcadDiagnostics?: {
    acquisitionStrategy: BcadAcquisitionStrategy;
    rankStrategy: BcadRankStrategy;
    competingHypothesesForAcquisition: boolean;
    effectiveQueryRank: number;
    rankStability: number;
    boundaryHypotheses: number;
    candidateUncertainty: number;
    boundaryConditionedQueries: number;
    gradientJumpQueries: number;
  };
  error?: string;
};

const DIMENSIONS = [16, 32];
const FOCUS_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank'
];
const METHODS: BcadMethod[] = [
  'asad',
  'ransac',
  'star-probe',
  'adaptive',
  'bcad-boundary-delayed',
  'bcad-prediction-control',
  'bcad-gradient-jump',
  'bcad-full-dimensional',
  'bcad-fixed-initial-rank',
  'bcad-stable-adaptive-rank',
  'bcad-single-hypothesis-acquisition'
];
const BASELINES: BcadMethod[] = ['asad', 'ransac', 'star-probe', 'adaptive'];
const PRIMARY_BCAD: BcadMethod = 'bcad-boundary-delayed';
const MAIN_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank'
];
const EVALUATION_POINTS = 192;

type BcadKeyCellAssessment = {
  dimension: number;
  scenario: ScenarioName;
  bestBaseline: BcadMethod | null;
  bcadMae: number | null;
  bestBaselineMae: number | null;
  bcadOracleCalls: number | null;
  bestComparableBaselineCalls: number | null;
  queryReduction: number | null;
  maeReduction: number | null;
  bcadFalseAcceptedRunRate: number | null;
  bestBaselineFalseAcceptedRunRate: number | null;
  falseAcceptedIncrease: number | null;
  performancePass: boolean;
  falseAcceptancePass: boolean;
  qualifies: boolean;
  reason: string;
};

type BcadVerdict = {
  verdict: 'promising' | 'partially-supported' | 'not-supported';
  primaryMethod: BcadMethod;
  mainScenarioWins: ScenarioName[];
  requiredMainScenarioWins: number;
  mainScenarioCoverage: boolean;
  fullFocusedCoverage: boolean;
  seedsPerCell: number;
  keyCells: BcadKeyCellAssessment[];
  budgetOverruns: number;
  falseAcceptanceTolerance: number;
  maeComparisonTolerance: number;
  absoluteMaeFloor: number;
};

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
  if (!Number.isSafeInteger(seeds) || seeds < 1) throw new Error('--seeds must be a positive safe integer');
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
    output: value('--output') ?? 'scratch/results/bcad'
  };
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function bcadArm(method: BcadMethod): {
  acquisitionStrategy: BcadAcquisitionStrategy;
  rankStrategy: BcadRankStrategy;
  competingHypothesesForAcquisition: boolean;
} | null {
  switch (method) {
    case 'bcad-boundary-delayed':
      return {
        acquisitionStrategy: 'boundary-uncertainty',
        rankStrategy: 'delayed-reduction',
        competingHypothesesForAcquisition: true
      };
    case 'bcad-prediction-control':
      return {
        acquisitionStrategy: 'prediction-disagreement',
        rankStrategy: 'delayed-reduction',
        competingHypothesesForAcquisition: true
      };
    case 'bcad-gradient-jump':
      return {
        acquisitionStrategy: 'gradient-jump',
        rankStrategy: 'delayed-reduction',
        competingHypothesesForAcquisition: true
      };
    case 'bcad-full-dimensional':
      return {
        acquisitionStrategy: 'boundary-uncertainty',
        rankStrategy: 'full-dimensional',
        competingHypothesesForAcquisition: true
      };
    case 'bcad-fixed-initial-rank':
      return {
        acquisitionStrategy: 'boundary-uncertainty',
        rankStrategy: 'fixed-initial',
        competingHypothesesForAcquisition: true
      };
    case 'bcad-stable-adaptive-rank':
      return {
        acquisitionStrategy: 'boundary-uncertainty',
        rankStrategy: 'stable-adaptive',
        competingHypothesesForAcquisition: true
      };
    case 'bcad-single-hypothesis-acquisition':
      return {
        acquisitionStrategy: 'boundary-uncertainty',
        rankStrategy: 'delayed-reduction',
        competingHypothesesForAcquisition: false
      };
    default:
      return null;
  }
}

function runOne(
  method: BcadMethod,
  scenario: Scenario,
  dimension: number,
  seed: number
): BcadRunRecord {
  const teacher = createTeacher(scenario);
  let actualOracleCalls = 0;
  const countedOracle = (input: number[]): number => {
    actualOracleCalls++;
    return teacher(input);
  };
  const { budget, lineCount } = solverBudget(dimension, scenario.terms.length);
  const scenarioOrdinal = SCENARIOS.indexOf(scenario.name);
  const modelSeed = seed * 104729 + dimension * 13007 + scenarioOrdinal * 8191;
  const evaluationPoints = randomTestPoints(seed, dimension);
  const scenarioModelHash = hash(scenario);
  const evaluationPointsHash = hash(evaluationPoints);
  const start = performance.now();
  let reconstructionComplete = false;
  let diagnosticOracleCalls: number | null = null;
  let activeRankEstimate: number | null = null;
  let bcadDiagnostics: BcadRunRecord['bcadDiagnostics'];
  let predictor: ((input: number[]) => number) | null = null;
  let recoveredPlanes: Array<{ normal: number[]; offset: number }> = [];
  let error: string | undefined;

  try {
    if (method === 'asad') {
      const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: modelSeed,
        maxGradientProbes: 96
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      activeRankEstimate = model.diagnostics.activeSubspaceRank;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdActiveSubspaceAdaptive(model, input);
    } else if (method === 'ransac' || method === 'star-probe' || method === 'adaptive') {
      const model: BlackBoxReluNdModel = decompileBlackBoxReluNdLineProbe(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        stageAMethod: method,
        lineCount,
        samplesPerLine: 96,
        ransacIterations: 64,
        minSupport: Math.max(dimension + 1, Math.floor(lineCount * 0.15)),
        randomSeed: modelSeed,
        oracleBudget: budget
      });
      diagnosticOracleCalls = model.adaptiveDiagnostics?.totalOracleCalls ?? null;
      reconstructionComplete = model.reconstructionComplete ?? true;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdLineProbeModel(model, input);
    } else {
      const arm = bcadArm(method);
      if (!arm) throw new Error(`Unknown BCAD method ${method}`);
      const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: modelSeed,
        maxGradientProbes: 96,
        acquisitionStrategy: arm.acquisitionStrategy,
        rankStrategy: arm.rankStrategy,
        competingHypothesesForAcquisition: arm.competingHypothesesForAcquisition
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      activeRankEstimate = model.diagnostics.estimatedActiveRank;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      bcadDiagnostics = {
        acquisitionStrategy: model.diagnostics.acquisitionStrategy,
        rankStrategy: model.diagnostics.rankStrategy,
        competingHypothesesForAcquisition: model.diagnostics.competingHypothesesForAcquisition,
        effectiveQueryRank: model.diagnostics.effectiveQueryRank,
        rankStability: model.diagnostics.rankStability,
        boundaryHypotheses: model.diagnostics.boundaryHypotheses,
        candidateUncertainty: model.diagnostics.candidateUncertainty,
        boundaryConditionedQueries: model.diagnostics.boundaryConditionedQueries,
        gradientJumpQueries: model.diagnostics.gradientJumpQueries
      };
      predictor = input => predictBlackBoxReluNdBoundaryConditionedActiveDiscovery(model, input);
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }

  const runtimeMs = performance.now() - start;
  const predictions = predictor ? evaluationPoints.map(predictor) : null;
  const mae = predictions === null
    ? null
    : evaluationPoints.reduce((sum, point, index) =>
      sum + Math.abs(teacher(point) - predictions[index]), 0) / evaluationPoints.length;
  const matching = matchPlanes(recoveredPlanes, scenario.terms);
  const status = error
    ? 'threw'
    : reconstructionComplete
      ? 'complete'
      : 'incomplete';
  return {
    schemaVersion: 1,
    seed,
    dimension,
    scenario: scenario.name,
    method,
    solverSeed: modelSeed,
    scenarioModelHash,
    evaluationPointsHash,
    status,
    oracleBudget: budget,
    actualOracleCalls,
    diagnosticOracleCalls,
    oracleCountReconciled: diagnosticOracleCalls !== null && diagnosticOracleCalls === actualOracleCalls,
    oracleBudgetExceeded: actualOracleCalls > budget,
    runtimeMs,
    mae,
    meanAngularErrorDeg: matching.meanAngle,
    correctHyperplanes: matching.correct,
    falseAcceptedHyperplanes: matching.falseAccepted,
    missedHyperplanes: matching.missed,
    recoveredHyperplanes: recoveredPlanes.length,
    activeRankEstimate,
    expectedActiveRank: scenario.expectedActiveRank,
    activeRankMisestimated: activeRankEstimate === null
      ? null
      : activeRankEstimate !== scenario.expectedActiveRank,
    reconstructionComplete,
    ...(bcadDiagnostics ? { bcadDiagnostics } : {}),
    ...(error ? { error } : {})
  };
}

function summarize(records: BcadRunRecord[]): Array<Record<string, unknown>> {
  const groups = new Map<string, BcadRunRecord[]>();
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    const group = groups.get(key) ?? [];
    group.push(record);
    groups.set(key, group);
  }
  return [...groups.entries()].sort(([first], [second]) => first.localeCompare(second)).map(([key, runs]) => {
    const [dimension, scenario, method] = key.split('|');
    const scored = runs.filter((run): run is BcadRunRecord & { mae: number } =>
      typeof run.mae === 'number');
    const rankRuns = runs.filter((run): run is BcadRunRecord & { activeRankMisestimated: boolean } =>
      typeof run.activeRankMisestimated === 'boolean');
    const diagnosticRuns = runs.flatMap(run => run.bcadDiagnostics ? [run.bcadDiagnostics] : []);
    const avg = (values: number[]): number | null => values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : null;
    return {
      dimension: Number(dimension),
      scenario,
      method,
      runs: runs.length,
      scoredRuns: scored.length,
      completeRate: avg(runs.map(run => run.reconstructionComplete ? 1 : 0)),
      thrownRate: avg(runs.map(run => run.status === 'threw' ? 1 : 0)),
      meanMae: avg(scored.map(run => run.mae)),
      meanAngularErrorDeg: avg(runs.map(run => run.meanAngularErrorDeg ?? 90)),
      meanCorrectHyperplanes: avg(runs.map(run => run.correctHyperplanes)),
      meanFalseAcceptedHyperplanes: avg(runs.map(run => run.falseAcceptedHyperplanes)),
      falseAcceptedRunRate: avg(runs.map(run => run.falseAcceptedHyperplanes > 0 ? 1 : 0)),
      meanMissedHyperplanes: avg(runs.map(run => run.missedHyperplanes)),
      meanOracleCalls: avg(runs.map(run => run.actualOracleCalls)),
      maximumOracleCalls: Math.max(...runs.map(run => run.actualOracleCalls)),
      meanRuntimeMs: avg(runs.map(run => run.runtimeMs)),
      activeRankMisestimateRate: rankRuns.length
        ? avg(rankRuns.map(run => run.activeRankMisestimated ? 1 : 0))
        : null,
      meanEstimatedActiveRank: avg(runs.flatMap(run =>
        run.activeRankEstimate === null ? [] : [run.activeRankEstimate])),
      meanEffectiveQueryRank: avg(diagnosticRuns.map(run => run.effectiveQueryRank)),
      meanRankStability: avg(diagnosticRuns.map(run => run.rankStability)),
      meanBoundaryHypotheses: avg(diagnosticRuns.map(run => run.boundaryHypotheses)),
      meanCandidateUncertainty: avg(diagnosticRuns.map(run => run.candidateUncertainty)),
      meanBoundaryConditionedQueries: avg(diagnosticRuns.map(run => run.boundaryConditionedQueries)),
      budgetOverruns: runs.filter(run => run.oracleBudgetExceeded).length,
      diagnosticMismatches: runs.filter(run =>
        run.diagnosticOracleCalls !== null && !run.oracleCountReconciled).length,
      diagnosticUnavailable: runs.filter(run => run.diagnosticOracleCalls === null).length
    };
  });
}

function keyCellAssessment(
  summary: Array<Record<string, unknown>>,
  dimension: number,
  scenario: ScenarioName
): BcadKeyCellAssessment {
  const get = (method: BcadMethod): Record<string, unknown> | undefined =>
    summary.find(row => row.dimension === dimension && row.scenario === scenario && row.method === method);
  const bcad = get(PRIMARY_BCAD);
  const baselines = BASELINES.map(get)
    .filter((row): row is Record<string, unknown> => row !== undefined && typeof row.meanMae === 'number');
  if (!bcad || typeof bcad.meanMae !== 'number' || !baselines.length) {
    return {
      dimension,
      scenario,
      bestBaseline: null,
      bcadMae: null,
      bestBaselineMae: null,
      bcadOracleCalls: null,
      bestComparableBaselineCalls: null,
      queryReduction: null,
      maeReduction: null,
      bcadFalseAcceptedRunRate: null,
      bestBaselineFalseAcceptedRunRate: null,
      falseAcceptedIncrease: null,
      performancePass: false,
      falseAcceptancePass: false,
      qualifies: false,
      reason: 'missing_scored_bcad_or_baseline'
    };
  }
  const bestBaseline = baselines.reduce((best, row) =>
    (row.meanMae as number) < (best.meanMae as number) ? row : best);
  const bestMae = bestBaseline.meanMae as number;
  const bestComparableCallRows = baselines.filter(row =>
    maeValuesAreComparable(row.meanMae as number, bestMae));
  const queryReduction = oracleCallReduction(
    bcad.meanOracleCalls as number,
    bestComparableCallRows.map(row => row.meanOracleCalls as number)
  );
  const maeReduction = bestMae > ASAD_MAE_ABSOLUTE_FLOOR
    ? 1 - (bcad.meanMae as number) / bestMae
    : null;
  const maePass = maeReduction !== null && maeReduction >= 0.25;
  const callPass = queryReduction !== null
    && queryReduction >= 0.3
    && !((bcad.meanMae as number) > bestMae * 1.05 + ASAD_MAE_ABSOLUTE_FLOOR);
  const falseAcceptedIncrease = (bcad.falseAcceptedRunRate as number)
    - (bestBaseline.falseAcceptedRunRate as number);
  const falseAcceptancePass = falseAcceptedIncrease <= 0.01;
  return {
    dimension,
    scenario,
    bestBaseline: bestBaseline.method as BcadMethod,
    bcadMae: bcad.meanMae as number,
    bestBaselineMae: bestMae,
    bcadOracleCalls: bcad.meanOracleCalls as number,
    bestComparableBaselineCalls: bestComparableCallRows.length
      ? Math.min(...bestComparableCallRows.map(row => row.meanOracleCalls as number))
      : null,
    queryReduction,
    maeReduction,
    bcadFalseAcceptedRunRate: bcad.falseAcceptedRunRate as number,
    bestBaselineFalseAcceptedRunRate: bestBaseline.falseAcceptedRunRate as number,
    falseAcceptedIncrease,
    performancePass: maePass || callPass,
    falseAcceptancePass,
    qualifies: (maePass || callPass) && falseAcceptancePass,
    reason: !falseAcceptancePass
      ? 'false_accepted_plane_rate_increase_over_1_percentage_point'
      : maePass
        ? 'at_least_25_percent_lower_mae'
        : callPass
          ? 'at_least_30_percent_fewer_calls_with_mae_within_5_percent'
          : 'neither_performance_threshold_met'
  };
}

function calculateVerdict(
  summary: Array<Record<string, unknown>>,
  dimensions: number[],
  scenarios: ScenarioName[],
  seeds: number,
  records: BcadRunRecord[]
): BcadVerdict {
  const mainScenarioWins = MAIN_SCENARIOS.filter(scenario =>
    dimensions.includes(16) && dimensions.includes(32)
    && [16, 32].every(dimension =>
      Boolean(keyCellAssessment(summary, dimension, scenario).qualifies)));
  const budgetOverruns = records.filter(record => record.oracleBudgetExceeded).length;
  const mainCoverage = MAIN_SCENARIOS.every(scenario => scenarios.includes(scenario));
  const requiredCoverage = dimensions.length === DIMENSIONS.length
    && FOCUS_SCENARIOS.every(scenario => scenarios.includes(scenario));
  const promising = mainScenarioWins.length >= 2
    && budgetOverruns === 0
    && seeds >= 100
    && requiredCoverage;
  return {
    verdict: promising
      ? 'promising'
      : mainScenarioWins.length > 0
        ? 'partially-supported'
        : 'not-supported',
    primaryMethod: PRIMARY_BCAD,
    mainScenarioWins,
    requiredMainScenarioWins: 2,
    mainScenarioCoverage: mainCoverage,
    fullFocusedCoverage: requiredCoverage,
    seedsPerCell: seeds,
    keyCells: dimensions.flatMap(dimension => MAIN_SCENARIOS.map(scenario =>
      keyCellAssessment(summary, dimension, scenario))),
    budgetOverruns,
    falseAcceptanceTolerance: 0.01,
    maeComparisonTolerance: ASAD_MAE_COMPARISON_TOLERANCE,
    absoluteMaeFloor: ASAD_MAE_ABSOLUTE_FLOOR
  };
}

function renderReport(
  runId: string,
  seeds: number,
  dimensions: number[],
  scenarios: ScenarioName[],
  methods: BcadMethod[],
  records: BcadRunRecord[],
  summary: Array<Record<string, unknown>>,
  verdict: ReturnType<typeof calculateVerdict>,
  rawFilename: string
): string {
  const rows = summary.map(row =>
    `| ${row.dimension} | ${row.scenario} | ${row.method} | ${row.scoredRuns}/${row.runs} | ${typeof row.meanMae === 'number' ? (row.meanMae as number).toPrecision(5) : 'n/a'} | ${Number(row.meanOracleCalls).toFixed(0)} | ${Number(row.meanAngularErrorDeg).toFixed(2)} | ${Number(row.meanCorrectHyperplanes).toFixed(2)} | ${Number(row.meanFalseAcceptedHyperplanes).toFixed(3)} | ${(100 * Number(row.falseAcceptedRunRate)).toFixed(1)}% | ${row.activeRankMisestimateRate === null ? 'n/a' : `${(100 * Number(row.activeRankMisestimateRate)).toFixed(1)}%`} | ${row.meanEstimatedActiveRank === null ? 'n/a' : Number(row.meanEstimatedActiveRank).toFixed(2)} | ${String(row.method).startsWith('bcad-') ? Number(row.meanEffectiveQueryRank).toFixed(2) : 'n/a'} | ${Number(row.meanRuntimeMs).toFixed(2)} | ${Number(row.thrownRate) * 100}% |`)
    .join('\n');
  const keyRows = verdict.keyCells.map(cell =>
    `| ${cell.dimension} | ${cell.scenario} | ${cell.bestBaseline ?? 'n/a'} | ${typeof cell.bcadMae === 'number' ? Number(cell.bcadMae).toPrecision(5) : 'n/a'} | ${typeof cell.bestBaselineMae === 'number' ? Number(cell.bestBaselineMae).toPrecision(5) : 'n/a'} | ${cell.queryReduction === null ? 'n/a' : `${(100 * Number(cell.queryReduction)).toFixed(1)}%`} | ${cell.maeReduction === null ? 'n/a' : `${(100 * Number(cell.maeReduction)).toFixed(1)}%`} | ${cell.falseAcceptedIncrease === null ? 'n/a' : `${(100 * Number(cell.falseAcceptedIncrease)).toFixed(2)} pp`} | ${cell.qualifies ? 'yes' : 'no'} | ${cell.reason} |`)
    .join('\n');
  const statusRows = methods.map(method => {
    const runs = records.filter(record => record.method === method);
    const complete = runs.filter(record => record.status === 'complete').length;
    const incomplete = runs.filter(record => record.status === 'incomplete').length;
    const threw = runs.filter(record => record.status === 'threw').length;
    return `| ${method} | ${complete} | ${incomplete} | ${threw} |`;
  }).join('\n');
  const totalComplete = records.filter(record => record.status === 'complete').length;
  const totalIncomplete = records.filter(record => record.status === 'incomplete').length;
  const totalThrew = records.filter(record => record.status === 'threw').length;
  const wins = verdict.mainScenarioWins.length
    ? verdict.mainScenarioWins.join(', ')
    : 'none';
  const outcome = verdict.verdict;
  return `# Boundary-Conditioned Active Discovery (BCAD): Focused Experiment

**Run:** ${runId}  
**Verdict:** ${outcome}  
**Dimensions:** ${dimensions.join(', ')}  
**Scenarios:** ${scenarios.join(', ')}  
**Methods/ablations:** ${methods.join(', ')}  
**Seeds per dimension/scenario/method cell:** ${seeds}  
**Raw observations:** [${rawFilename}](../../scratch/results/bcad/${rawFilename})

## Research question and implementation

BCAD maintains several candidate boundary planes inferred from oracle-only finite-difference gradient jumps and affine intercept differences. Near-parallel candidates with distinct offsets remain separate. Query acquisition can score the same candidate pool by (A) local affine prediction disagreement, (B) disagreement between competing boundary-conditioned gradient predictions plus probability of crossing a candidate kink, or (C) local gradient-jump detectability. Candidate boundary hypotheses do not imply that the active subspace or all facets have been identified.

One gradient observation costs exactly $2d+1$ actual oracle calls: one center value and two coordinate samples per input dimension. A single wrapper counts every call; full stencils are started only when the remaining budget can cover them. Model fitting performs no additional oracle calls. Solver input contains no teacher weights, normals, offsets, scenario labels, model hashes, or evaluation points.

## Fairness and evaluation

All methods share the deterministic teacher generated from the same dimension, scenario, and seed, the same oracle budget from the existing ASAD benchmark budget function, the same solver random seed, and the same independently generated 192-point test set. Teacher metadata and test points are used only by the benchmark after the solver returns. SHA-256 fingerprints of the teacher fixture and test points are recorded per run and independently reconciled by the verifier. Runtime measures only the solver call.

The fixed-rank arm freezes its gradient-covariance rank estimate after its initial sample window. The stable-adaptive arm changes rank only after the estimate and normalized spectrum remain stable across repeated updates. The delayed-reduction arm retains full-dimensional exploration until enough supported boundary candidates exist and their uncertainty is low. The full-dimensional arm ablates subspace restriction. The single-hypothesis acquisition arm tests the contribution of comparing alternatives. The BCAD prediction-control arm isolates acquisition scoring while retaining BCAD candidate generation; the unchanged ASAD arm is also included as the original algorithm baseline.

## Acceptance criteria and primary comparison

- The three primary problem classes are near-parallel facets, multiple crossings, and small-amplitude jumps at dimensions 16 and 32. A cell passes if primary BCAD has at least 25% lower MAE than the best scored existing mode, or uses at least 30% fewer calls than an existing mode within 5% MAE of the best existing mode while BCAD MAE is no more than 5% worse than that best MAE.
- False-accepted boundary rate must not exceed the lowest rate among the best-MAE baseline comparison group by more than one percentage point.
- At least two of the three primary problem classes must pass at both dimensions, with 100 seeds per cell and zero actual oracle-budget overruns.
- Best existing modes are selected from ASAD, RANSAC, Star-Probe, and Adaptive. An unscored method is not treated as a zero-MAE result. Empty MAE-comparable sets do not count as query savings.
- Relative MAE reduction is shown as n/a when the baseline is at or below the $1e-8$ absolute floor; percentage change from a near-zero error is not meaningful.

**Primary problem classes passing at both dimensions:** ${wins}.  
**Overall verdict:** ${outcome}.  
**Actual budget overruns:** ${verdict.budgetOverruns}.  
**Required seeds per cell reached:** ${seeds >= 100 ? 'yes' : 'no'}.

| D | Scenario | Best baseline | BCAD MAE | Baseline MAE | Call reduction | MAE reduction | False-accept increase | Pass | Reason |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
${keyRows}

## Results by method and ablation

| D | Scenario | Method | Scored/runs | MAE | Oracle calls | Angle (deg) | Correct planes | Mean false planes | Runs with false plane | Rank error | Estimated rank | Query rank | Runtime (ms) | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${rows}

Interpret component comparisons causally only within the listed one-factor arms: boundary-uncertainty versus prediction-control and gradient-jump isolate acquisition scoring; delayed-reduction versus full-dimensional isolates dimension reduction; delayed-reduction versus fixed-initial/stable-adaptive isolates rank policy; delayed-reduction versus single-hypothesis acquisition tests competing-boundary information. These comparisons do not tune thresholds on the independent test points.

## Completion and exception counts

| Method | Complete | Incomplete | Threw |
|---|---:|---:|---:|
${statusRows}
| **Total** | **${totalComplete}** | **${totalIncomplete}** | **${totalThrew}** |

## Conclusion

BCAD is **${outcome}** under the predefined criteria: ${verdict.keyCells.filter(cell => cell.qualifies).length} of ${verdict.keyCells.length} primary dimension/scenario cells qualify, and ${totalThrew} of ${records.length} solver attempts throw. Passing model-based regression tests does not establish a benchmark advantage. When the criteria fail, use the per-cell MAE, boundary counts, false-accept rates, rank errors, query counts, incomplete runs, and exceptions above to distinguish acquisition failure, rank-estimation failure, and unresolved boundary identifiability. No failing runs are replaced by synthetic scores.

## Evidence and limits

- Raw per-run records: [${rawFilename}](../../scratch/results/bcad/${rawFilename})
- The JSON summary next to the raw file contains per-cell aggregates, ablation arms, per-cell acceptance assessments, and the verdict inputs.
- This is a deterministic synthetic scalar-output experiment; it does not establish performance on noisy real APIs, deep neural networks, or language models.
- Training-fit residual is evaluated on acquired observations and is not an independent holdout; benchmark MAE uses a separate evaluation set.
`;
}

function main(): void {
  const { seeds, dimensions, scenarios, output } = parseArgs(process.argv.slice(2));
  const outputDirectory = path.resolve(output);
  fs.mkdirSync(outputDirectory, { recursive: true });
  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const rawFilename = `bcad-${runId}-raw.jsonl`;
  const rawPath = path.join(outputDirectory, rawFilename);
  const records: BcadRunRecord[] = [];
  const maximum = seeds * dimensions.length * scenarios.length * METHODS.length;
  let run = 0;
  for (const dimension of dimensions) {
    for (const scenarioName of scenarios) {
      for (let seed = 1; seed <= seeds; seed++) {
        const scenario = makeScenario(scenarioName, dimension, seed);
        for (const method of METHODS) {
          const record = runOne(method, scenario, dimension, seed);
          records.push(record);
          fs.appendFileSync(rawPath, `${JSON.stringify(record)}\n`);
          run++;
          if (run % 100 === 0 || run === maximum) {
            process.stdout.write(
              `BCAD_PROGRESS ${run}/${maximum} dimension=${dimension} scenario=${scenarioName} seed=${seed} method=${method}\n`
            );
          }
        }
      }
    }
  }
  const summary = summarize(records);
  const verdict = calculateVerdict(summary, dimensions, scenarios, seeds, records);
  const summaryFilename = `summary-${runId}.json`;
  fs.writeFileSync(path.join(outputDirectory, summaryFilename), `${JSON.stringify({
    schemaVersion: 1,
    runId,
    seedsPerCell: seeds,
    dimensions,
    scenarios,
    methods: METHODS,
    evaluationPoints: EVALUATION_POINTS,
    fixedOracleBudgetsPerCell: true,
    commonSeededTeachersAndEvaluationPoints: true,
    summary,
    verdict,
    totalRuns: records.length,
    totalOracleBudgetOverruns: records.filter(record => record.oracleBudgetExceeded).length,
    totalDiagnosticMismatches: records.filter(record =>
      record.diagnosticOracleCalls !== null && !record.oracleCountReconciled).length,
    totalDiagnosticUnavailable: records.filter(record => record.diagnosticOracleCalls === null).length,
    rawFile: rawFilename
  }, null, 2)}\n`);

  const fullFocusedCampaign = dimensions.length === DIMENSIONS.length
    && DIMENSIONS.every(dimension => dimensions.includes(dimension))
    && scenarios.length === FOCUS_SCENARIOS.length
    && FOCUS_SCENARIOS.every(scenario => scenarios.includes(scenario))
    && seeds >= 100;
  if (fullFocusedCampaign) {
    fs.writeFileSync(
      path.resolve('docs/research/2026-10-29-bcad-experimental-report.md'),
      renderReport(runId, seeds, dimensions, scenarios, METHODS, records, summary, verdict, rawFilename)
    );
  }
  process.stdout.write(
    `BCAD_BENCHMARK_COMPLETE seeds=${seeds} dimensions=${dimensions.length} scenarios=${scenarios.length} methods=${METHODS.length} runs=${records.length} verdict=${verdict.verdict} raw=${rawFilename} summary=${summaryFilename}\n`
  );
}

if (require.main === module) main();
