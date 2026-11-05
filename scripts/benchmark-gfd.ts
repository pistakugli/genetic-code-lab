import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import {
  decompileBlackBoxReluNdActiveSubspaceAdaptive,
  predictBlackBoxReluNdActiveSubspaceAdaptive
} from '../src/solvers/active-subspace-adaptive-discovery';
import {
  decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery,
  predictBlackBoxReluNdBoundaryConditionedActiveDiscovery
} from '../src/solvers/boundary-conditioned-active-discovery';
import {
  decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation,
  predictBlackBoxReluNdMultiHypothesisBoundaryContinuation
} from '../src/solvers/multi-hypothesis-boundary-continuation';
import {
  BlackBoxReluNdModel,
  decompileBlackBoxReluNdLineProbe,
  predictBlackBoxReluNdLineProbeModel
} from '../src/solvers/blackbox-relu-nd-line-probe';
import {
  createGlobalFunctionDistillationSampleInputs,
  GlobalFunctionDistillationConfig
} from '../src/solvers/global-function-distillation';
import { GfdCudaClient, GfdCudaModel } from './gfd-vp-cuda-client';
import {
  createTeacher,
  makeScenario,
  mean,
  randomTestPoints,
  SCENARIOS,
  Scenario,
  ScenarioName,
  solverBudget
} from './benchmark-asad';
import {
  ASAD_MAE_ABSOLUTE_FLOOR,
  maeValuesAreComparable
} from './asad-benchmark-metrics';

type Method =
  | 'asad'
  | 'ransac'
  | 'star-probe'
  | 'adaptive'
  | 'bcad'
  | 'mbcd-full'
  | 'gfd-profiled'
  | 'gfd-single-start'
  | 'gfd-joint-fit';
type GfdMethod = Extract<Method, `gfd-${string}`>;

type RunRecord = {
  schemaVersion: 1;
  seed: number;
  dimension: number;
  scenario: ScenarioName;
  method: Method;
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
  reconstructionComplete: boolean;
  gfdDiagnostics?: {
    sampleCount: number;
    trainingSamples: number;
    validationSamples: number;
    restartsAttempted: number;
    selectedRestart: number;
    trainingMae: number | null;
    validationMae: number | null;
    computeDevice: string;
    deviceName: string;
    cudaMemoryAllocatedBytes: number;
    trainingTimeMs: number;
    dtype: string;
  };
  error?: string;
};

type SummaryRow = {
  dimension: number;
  scenario: ScenarioName;
  method: Method;
  runs: number;
  scoredRuns: number;
  completeRuns: number;
  incompleteRuns: number;
  thrownRuns: number;
  completionRate: number;
  meanMae: number | null;
  meanOracleCalls: number;
  meanNormalErrorDeg: number | null;
  meanOffsetError: number | null;
  meanCorrectHyperplanes: number;
  meanFalseAcceptedHyperplanes: number;
  falseAcceptedRunRate: number | null;
  meanRuntimeMs: number;
  meanCudaMemoryAllocatedBytes: number | null;
  meanGpuTrainingMs: number | null;
};

type CellAssessment = {
  dimension: number;
  scenario: ScenarioName;
  bestBaseline: Method | null;
  gfdMae: number | null;
  baselineMae: number | null;
  gfdOracleCalls: number | null;
  bestComparableBaselineCalls: number | null;
  oracleCallReduction: number | null;
  maeReduction: number | null;
  gfdCompletionRate: number | null;
  baselineCompletionRate: number | null;
  falseAcceptedIncrease: number | null;
  qualifies: boolean;
  reason: string;
};

type CampaignSummary = {
  runId: string;
  seedsPerCell: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  methods: Method[];
  evaluationPoints: number;
  totalRuns: number;
  totalOracleBudgetOverruns: number;
  totalDiagnosticMismatches: number;
  statusTotals: { complete: number; incomplete: number; threw: number };
  rawFile: string;
  summary: SummaryRow[];
  verdict: {
    verdict: 'promising' | 'partially-supported' | 'not-supported';
    primaryMethod: GfdMethod;
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
const PRIMARY_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank'
];
const METHODS: Method[] = [
  'asad',
  'ransac',
  'star-probe',
  'adaptive',
  'bcad',
  'mbcd-full',
  'gfd-profiled',
  'gfd-single-start',
  'gfd-joint-fit'
];
const EXISTING_METHODS: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full'
];
const EVALUATION_POINTS = 192;
const FALSE_ACCEPTANCE_TOLERANCE = 0.01;
const MAE_COMPARISON_TOLERANCE = 0.05;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(args: string[]): {
  seeds: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  methods: Method[];
  output: string;
  pilot: boolean;
} {
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index >= 0 ? args[index + 1] : undefined;
  };
  const pilot = args.includes('--pilot');
  const seeds = Number(value('--seeds') ?? (pilot ? 3 : 100));
  if (!Number.isSafeInteger(seeds) || seeds < 1) throw new Error('--seeds must be a positive safe integer');
  const dimensions = (value('--dimensions') ?? DIMENSIONS.join(',')).split(',').map(Number);
  if (!dimensions.length || dimensions.some(dimension => !DIMENSIONS.includes(dimension))) {
    throw new Error(`--dimensions must be selected from ${DIMENSIONS.join(',')}`);
  }
  const scenarios = (value('--scenarios') ?? FOCUS_SCENARIOS.join(',')).split(',') as ScenarioName[];
  if (!scenarios.length || scenarios.some(scenario => !FOCUS_SCENARIOS.includes(scenario))) {
    throw new Error(`--scenarios must be selected from ${FOCUS_SCENARIOS.join(',')}`);
  }
  const methods = (value('--methods') ?? METHODS.join(',')).split(',') as Method[];
  if (!methods.length || methods.some(method => !METHODS.includes(method))) {
    throw new Error(`--methods must be selected from ${METHODS.join(',')}`);
  }
  return {
    seeds,
    dimensions: [...new Set(dimensions)],
    scenarios: [...new Set(scenarios)],
    methods: [...new Set(methods)],
    output: value('--output') ?? 'scratch/results/gfd',
    pilot
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
  truth: Scenario['terms']
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
  pairs.sort((first, second) => first.angle + 100 * first.offset - (second.angle + 100 * second.offset));
  const usedPredicted = new Set<number>();
  const usedTruth = new Set<number>();
  const angles: number[] = [];
  const offsets: number[] = [];
  for (const pair of pairs) {
    if (pair.angle > 10 || pair.offset > 0.1
      || usedPredicted.has(pair.predictedIndex) || usedTruth.has(pair.truthIndex)) continue;
    usedPredicted.add(pair.predictedIndex);
    usedTruth.add(pair.truthIndex);
    angles.push(pair.angle);
    offsets.push(pair.offset);
  }
  return {
    correct: usedTruth.size,
    falseAccepted: predicted.length - usedPredicted.size,
    missed: truth.length - usedTruth.size,
    meanAngle: truth.length
      ? (angles.reduce((sum, item) => sum + item, 0) + 90 * (truth.length - angles.length)) / truth.length
      : null,
    meanOffset: offsets.length ? mean(offsets) : null
  };
}

async function runOne(
  method: Method,
  scenario: Scenario,
  dimension: number,
  seed: number,
  cudaWorker: GfdCudaClient | null
): Promise<RunRecord> {
  const teacher = createTeacher(scenario);
  let actualOracleCalls = 0;
  const countedOracle = (input: number[]): number => {
    actualOracleCalls++;
    return teacher(input);
  };
  const { budget, lineCount } = solverBudget(dimension, scenario.terms.length);
  const solverSeed = seed * 104729 + dimension * 13007 + SCENARIOS.indexOf(scenario.name) * 8191;
  const evaluationPoints = randomTestPoints(seed, dimension);
  const evaluationPointsHash = hash(evaluationPoints);
  const scenarioModelHash = hash(scenario);
  const start = performance.now();
  let reconstructionComplete = false;
  let diagnosticOracleCalls: number | null = null;
  let predictor: ((input: number[]) => number) | null = null;
  let recoveredPlanes: Array<{ normal: number[]; offset: number }> = [];
  let gfdDiagnostics: RunRecord['gfdDiagnostics'];
  let error: string | undefined;
  try {
    if (method.startsWith('gfd-')) {
      if (!cudaWorker) throw new Error('GFD-VP requires the CUDA worker');
      const config: GlobalFunctionDistillationConfig = {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: solverSeed,
        sampleCount: Math.min(4096, Math.floor(budget * 0.6)),
        restarts: method === 'gfd-single-start' ? 1 : 4,
        optimizationSteps: 220,
        batchSize: 256,
        profileOutputLayer: method !== 'gfd-joint-fit'
      };
      const inputs = createGlobalFunctionDistillationSampleInputs(config);
      const outputs = inputs.map(countedOracle);
      const model: GfdCudaModel = await cudaWorker.fit({
        inputs,
        outputs,
        dimension,
        termCount: scenario.terms.length,
        randomSeed: solverSeed,
        restarts: config.restarts!,
        optimizationSteps: config.optimizationSteps!,
        batchSize: config.batchSize!,
        profileOutputLayer: config.profileOutputLayer!
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => {
        if (input.length !== model.dimension) {
          throw new Error('GFD-VP prediction dimension mismatch');
        }
        let output = model.baseIntercept;
        for (let axis = 0; axis < model.dimension; axis++) {
          output += model.baseWeights[axis] * input[axis];
        }
        for (const term of model.terms) {
          output += term.readout * Math.max(0,
            term.normal.reduce((sum, value, axis) => sum + value * input[axis], term.offset));
        }
        return output;
      };
      gfdDiagnostics = {
        sampleCount: model.diagnostics.sampleCount,
        trainingSamples: model.diagnostics.trainingSamples,
        validationSamples: model.diagnostics.validationSamples,
        restartsAttempted: model.diagnostics.restartsAttempted,
        selectedRestart: model.diagnostics.selectedRestart,
        trainingMae: model.diagnostics.trainingMae,
        validationMae: model.diagnostics.validationMae,
        computeDevice: model.diagnostics.computeDevice,
        deviceName: cudaWorker.device.deviceName,
        cudaMemoryAllocatedBytes: model.diagnostics.cudaMemoryAllocatedBytes,
        trainingTimeMs: model.diagnostics.trainingTimeMs,
        dtype: model.diagnostics.dtype
      };
    } else if (method === 'mbcd-full') {
      const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: solverSeed,
        maxGradientProbes: 80,
        queryPolicy: 'contradiction-seeking',
        rankPolicy: 'retain-alternatives',
        validationPolicy: 'independent'
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdMultiHypothesisBoundaryContinuation(model, input);
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
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdActiveSubspaceAdaptive(model, input);
    } else {
      const stageAMethod = method as 'ransac' | 'star-probe' | 'adaptive';
      const model: BlackBoxReluNdModel = decompileBlackBoxReluNdLineProbe(countedOracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        stageAMethod,
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
  const status = error ? 'threw' : reconstructionComplete ? 'complete' : 'incomplete';
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
    reconstructionComplete,
    ...(gfdDiagnostics ? { gfdDiagnostics } : {}),
    ...(error ? { error } : {})
  };
}

function summarize(records: RunRecord[]): SummaryRow[] {
  const groups = new Map<string, RunRecord[]>();
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    groups.set(key, [...(groups.get(key) ?? []), record]);
  }
  return [...groups.values()].map(group => {
    const first = group[0];
    const scored = group.filter(record => record.mae !== null);
    return {
      dimension: first.dimension,
      scenario: first.scenario,
      method: first.method,
      runs: group.length,
      scoredRuns: scored.length,
      completeRuns: group.filter(record => record.status === 'complete').length,
      incompleteRuns: group.filter(record => record.status === 'incomplete').length,
      thrownRuns: group.filter(record => record.status === 'threw').length,
      completionRate: group.filter(record => record.reconstructionComplete).length / group.length,
      meanMae: scored.length ? mean(scored.map(record => record.mae!)) : null,
      meanOracleCalls: mean(group.map(record => record.actualOracleCalls)),
      meanNormalErrorDeg: scored.length
        ? mean(scored.map(record => record.meanAngularErrorDeg ?? 90))
        : null,
      meanOffsetError: scored.length
        ? mean(scored.map(record => record.meanOffsetError ?? 1))
        : null,
      meanCorrectHyperplanes: mean(scored.map(record => record.correctHyperplanes)),
      meanFalseAcceptedHyperplanes: mean(scored.map(record => record.falseAcceptedHyperplanes)),
      falseAcceptedRunRate: scored.length
        ? scored.filter(record => record.falseAcceptedHyperplanes > 0).length / scored.length
        : null,
      meanRuntimeMs: mean(group.map(record => record.runtimeMs)),
      meanCudaMemoryAllocatedBytes: group.some(record => record.gfdDiagnostics)
        ? mean(group.flatMap(record => record.gfdDiagnostics
          ? [record.gfdDiagnostics.cudaMemoryAllocatedBytes] : []))
        : null,
      meanGpuTrainingMs: group.some(record => record.gfdDiagnostics)
        ? mean(group.flatMap(record => record.gfdDiagnostics
          ? [record.gfdDiagnostics.trainingTimeMs] : []))
        : null
    };
  }).sort((first, second) =>
    first.dimension - second.dimension
    || FOCUS_SCENARIOS.indexOf(first.scenario) - FOCUS_SCENARIOS.indexOf(second.scenario)
    || METHODS.indexOf(first.method) - METHODS.indexOf(second.method));
}

function assessCell(rows: SummaryRow[], dimension: number, scenario: ScenarioName): CellAssessment {
  const cell = rows.filter(row => row.dimension === dimension && row.scenario === scenario);
  const candidate = cell.find(row => row.method === 'gfd-profiled');
  const baselines = cell.filter(row =>
    EXISTING_METHODS.includes(row.method) && row.meanMae !== null && row.scoredRuns > 0);
  const best = [...baselines].sort((first, second) => first.meanMae! - second.meanMae!)[0] ?? null;
  if (!candidate || candidate.meanMae === null || !best || best.meanMae === null) {
    return {
      dimension, scenario, bestBaseline: best?.method ?? null,
      gfdMae: candidate?.meanMae ?? null, baselineMae: best?.meanMae ?? null,
      gfdOracleCalls: candidate?.meanOracleCalls ?? null, bestComparableBaselineCalls: null,
      oracleCallReduction: null, maeReduction: null,
      gfdCompletionRate: candidate?.completionRate ?? null,
      baselineCompletionRate: best?.completionRate ?? null,
      falseAcceptedIncrease: null, qualifies: false, reason: 'missing_scored_primary_method_or_baseline'
    };
  }
  const comparable = baselines.filter(row => maeValuesAreComparable(row.meanMae!, best.meanMae!));
  const comparableCalls = comparable.length
    ? Math.min(...comparable.map(row => row.meanOracleCalls))
    : null;
  const callReduction = comparableCalls !== null && comparableCalls > 0
    ? 1 - candidate.meanOracleCalls / comparableCalls
    : null;
  const maeReduction = best.meanMae > ASAD_MAE_ABSOLUTE_FLOOR
    ? 1 - candidate.meanMae / best.meanMae
    : null;
  const maePass = maeReduction !== null && maeReduction >= 0.25;
  const efficiencyPass = callReduction !== null
    && callReduction >= 0.3
    && candidate.meanMae <= best.meanMae * (1 + MAE_COMPARISON_TOLERANCE)
      + ASAD_MAE_ABSOLUTE_FLOOR;
  const baselineFalseRate = comparable.map(row => row.falseAcceptedRunRate)
    .filter((value): value is number => value !== null);
  const falseIncrease = candidate.falseAcceptedRunRate !== null && baselineFalseRate.length
    ? candidate.falseAcceptedRunRate - Math.min(...baselineFalseRate)
    : null;
  const falsePass = falseIncrease !== null && falseIncrease <= FALSE_ACCEPTANCE_TOLERANCE;
  const qualifies = (maePass || efficiencyPass) && falsePass;
  return {
    dimension, scenario, bestBaseline: best.method,
    gfdMae: candidate.meanMae, baselineMae: best.meanMae,
    gfdOracleCalls: candidate.meanOracleCalls, bestComparableBaselineCalls: comparableCalls,
    oracleCallReduction: callReduction, maeReduction,
    gfdCompletionRate: candidate.completionRate,
    baselineCompletionRate: best.completionRate,
    falseAcceptedIncrease: falseIncrease, qualifies,
    reason: qualifies
      ? maePass ? 'mae_improved_by_at_least_25_percent'
        : 'at_least_30_percent_fewer_calls_with_mae_within_5_percent'
      : !(maePass || efficiencyPass)
        ? 'neither_predeclared_performance_threshold_met'
        : 'false_accepted_run_rate_increased_by_more_than_1_percentage_point'
  };
}

function format(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? 'n/a' : Number(value.toPrecision(5)).toString();
}

function percent(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? 'n/a' : `${(100 * value).toFixed(2)}%`;
}

export function renderReport(summary: CampaignSummary, pilot = false): string {
  const primaryRows = summary.verdict.keyCells.map(cell =>
    `| ${cell.dimension} | ${cell.scenario} | ${cell.bestBaseline ?? 'n/a'} | ${format(cell.gfdMae)} | ${format(cell.baselineMae)} | ${percent(cell.oracleCallReduction)} | ${percent(cell.maeReduction)} | ${percent(cell.gfdCompletionRate)} | ${percent(cell.falseAcceptedIncrease)} | ${cell.qualifies ? 'yes' : 'no'} | ${cell.reason} |`
  ).join('\n');
  const metricRows = summary.summary.map(row =>
    `| ${row.dimension} | ${row.scenario} | ${row.method} | ${row.scoredRuns}/${row.runs} | ${format(row.meanMae)} | ${format(row.meanOracleCalls)} | ${percent(row.completionRate)} | ${format(row.meanNormalErrorDeg)} | ${format(row.meanOffsetError)} | ${percent(row.falseAcceptedRunRate)} | ${format(row.meanRuntimeMs)} | ${row.meanCudaMemoryAllocatedBytes === null ? 'n/a' : format(row.meanCudaMemoryAllocatedBytes / (1024 * 1024))} | ${format(row.meanGpuTrainingMs)} | ${row.thrownRuns} |`
  ).join('\n');
  const total = summary.statusTotals;
  return `# Global Function Distillation with Variable Projection (GFD-VP)${pilot ? ' - Pilot' : ''}

**Run:** ${summary.runId}  
**Verdict:** ${summary.verdict.verdict}  
**Seeds per cell:** ${summary.seedsPerCell}  
**Dimensions:** ${summary.dimensions.join(', ')}  
**Scenarios:** ${summary.scenarios.join(', ')}  
**Methods:** ${summary.methods.join(', ')}  
**Raw runs:** [${summary.rawFile}](../../scratch/results/gfd/${pilot ? 'pilot/' : ''}${summary.rawFile})

${pilot ? `## Pilot scope

This is a ten-seed screening experiment, not the predeclared 100-seed confirmation. Its purpose is to reject an unpromising approach early. A “not-supported” pilot verdict is evidence against proceeding unchanged, not a formal 100-seed estimate.
` : ''}

## Hypothesis and algorithm

GFD-VP fits the complete scalar ReLU sum directly to uniformly sampled oracle values. It does not estimate local oracle gradients, active rank, line kinks, or candidate hyperplanes before fitting. At each optimization checkpoint, it profiles out the affine/readout layer by solving a regularized least-squares system for the current hidden ReLU features; only hidden normals and offsets are updated by gradients of the value-fit objective. Hidden normals are gauge-normalized after every update. Independent held-out oracle samples select among deterministic restarts. Solver inputs contain only the oracle, dimension, ranges, term count, seed, and budget.

The method is non-convex and depends on the known term count. A successful value fit does not imply geometrically identified units because ReLU decompositions can be non-identifiable or nearly cancelling; all units and matched geometric errors are still reported.

## Paired design and unchanged acceptance criteria

Every method receives the same deterministic teacher, model seed, oracle-call ceiling, and independently generated 192-point test set for each seed/dimension/scenario. Existing methods include ASAD, RANSAC, Star-Probe, Adaptive, BCAD, and MBCD. The three primary classes are near-parallel facets, multiple crossings, and small-amplitude jumps at 16D and 32D; contaminated probes and orthogonal full-rank teachers are controls. This run uses ${summary.seedsPerCell} seeds per cell; the predeclared full campaign uses 100. All method predictions are scored on every seed; exceptions remain unscored and are not dropped from run counts. GFD-VP fits are dispatched through one persistent CUDA worker; samples are staged in pinned host memory and transferred to CUDA tensors. Each run records the CUDA device, tensor device, peak allocated VRAM, and GPU fit time separately from end-to-end solver runtime.

The previously fixed cell criteria are unchanged: at least 25% lower MAE than the best scored existing method, or at least 30% fewer oracle calls than a baseline within 5% of the best baseline MAE while GFD MAE is within 5% of that best MAE; false-accepted-run rate may rise by at most one percentage point. At least two of three primary classes must qualify at both dimensions, with no budget overrun. The verdict is not inferred from tests.

## Primary comparison

| D | Scenario | Best existing method | GFD-VP MAE | Baseline MAE | Call reduction | MAE reduction | GFD complete rate | False-accepted increase | Qualifies | Reason |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
${primaryRows}

**Primary scenario classes qualifying at both dimensions:** ${summary.verdict.primaryScenarioWins.length}/3 (${summary.verdict.primaryScenarioWins.join(', ') || 'none'}).  
**Actual oracle-budget overruns:** ${summary.verdict.budgetOverruns}.  
**Run statuses:** ${total.complete} complete, ${total.incomplete} incomplete, ${total.threw} threw.

## All-seed method metrics

| D | Scenario | Method | Scored/runs | MAE | Oracle calls | Complete rate | Normal error (deg) | Offset error | False-accepted run rate | Runtime (ms) | CUDA VRAM (MiB) | GPU fit (ms) | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${metricRows}

## Ablations and limitations

GFD-single-start removes restart selection; GFD-joint-fit removes variable projection and trains all readout parameters jointly. Compare their matched MAE, geometry, completion, oracle use, and runtime against GFD-VP; ablations are explanatory and cannot replace the predeclared primary verdict. Synthetic scalar-output results do not establish effectiveness on deep networks or real noisy APIs. The verifier independently checks paired teacher/test fingerprints, run completeness, actual oracle counts, diagnostics, budget ceilings, and verdict calculations from raw records.

${pilot ? `## Pilot interpretation and next experiment

The pilot does not meet the predeclared success criteria in any primary cell. The global fit is exact enough to complete the small-amplitude cases, but its MAE and oracle use do not beat ASAD there. It does not complete the near-parallel or multi-crossing reconstructions; multi-crossing normal estimates can be close while the function error and completion criterion still fail. The contaminated and full-rank controls also do not show reliable completion. This rejects the current GFD-VP configuration as a candidate for an unchanged 100-seed campaign; it is not evidence that global value fitting can never work.

The next distinct hypothesis is **global spectral-moment inversion**: use the distributional Hessian structure of a ReLU sum, whose Fourier transform encodes plane orientations and offsets, instead of optimizing latent hidden units against sampled values. This is unverified and should first be tested on a small, fixed synthetic pilot with the same oracle-only access and exact query accounting. It must demonstrate geometric identification on near-parallel and full-rank controls before any larger campaign.
` : ''}
`;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const outputDirectory = path.resolve(options.output);
  fs.mkdirSync(outputDirectory, { recursive: true });
  const projectRoot = path.resolve(__dirname, '..');
  const cudaWorker = options.methods.some(method => method.startsWith('gfd-'))
    ? await GfdCudaClient.start(projectRoot)
    : null;
  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const rawFile = `gfd-${runId}-raw.jsonl`;
  const rawPath = path.join(outputDirectory, rawFile);
  const records: RunRecord[] = [];
  const totalRuns = options.seeds * options.dimensions.length
    * options.scenarios.length * options.methods.length;
  let completedRuns = 0;
  try {
    for (const dimension of options.dimensions) {
      for (const scenarioName of options.scenarios) {
        for (let seed = 1; seed <= options.seeds; seed++) {
          const scenario = makeScenario(scenarioName, dimension, seed);
          for (const method of options.methods) {
            const record = await runOne(method, scenario, dimension, seed, cudaWorker);
            records.push(record);
            fs.appendFileSync(rawPath, `${JSON.stringify(record)}\n`);
            completedRuns++;
            if (completedRuns % 100 === 0 || completedRuns === totalRuns) {
              process.stdout.write(`GFD_PROGRESS ${completedRuns}/${totalRuns} dimension=${dimension} scenario=${scenarioName} seed=${seed} method=${method}\n`);
            }
          }
        }
      }
    }
  } finally {
    if (cudaWorker) await cudaWorker.close();
  }
  const summaryRows = summarize(records);
  const keyCells = options.dimensions.flatMap(dimension =>
    options.scenarios.filter(scenario => PRIMARY_SCENARIOS.includes(scenario))
      .map(scenario => assessCell(summaryRows, dimension, scenario)));
  const primaryScenarioWins = PRIMARY_SCENARIOS.filter(scenario =>
    options.dimensions.includes(16) && options.dimensions.includes(32)
    && [16, 32].every(dimension => keyCells.some(cell =>
      cell.dimension === dimension && cell.scenario === scenario && cell.qualifies)));
  const budgetOverruns = records.filter(record => record.oracleBudgetExceeded).length;
  const diagnosticMismatches = records.filter(record => !record.oracleCountReconciled).length;
  const verdict: CampaignSummary['verdict']['verdict'] = budgetOverruns > 0
    ? 'not-supported'
    : primaryScenarioWins.length >= 2 ? 'promising'
      : primaryScenarioWins.length === 1 ? 'partially-supported' : 'not-supported';
  const summary: CampaignSummary = {
    runId,
    seedsPerCell: options.seeds,
    dimensions: options.dimensions,
    scenarios: options.scenarios,
    methods: options.methods,
    evaluationPoints: EVALUATION_POINTS,
    totalRuns: records.length,
    totalOracleBudgetOverruns: budgetOverruns,
    totalDiagnosticMismatches: diagnosticMismatches,
    statusTotals: {
      complete: records.filter(record => record.status === 'complete').length,
      incomplete: records.filter(record => record.status === 'incomplete').length,
      threw: records.filter(record => record.status === 'threw').length
    },
    rawFile,
    summary: summaryRows,
    verdict: {
      verdict,
      primaryMethod: 'gfd-profiled',
      primaryScenarioWins,
      requiredPrimaryScenarioWins: 2,
      seedsPerCell: options.seeds,
      keyCells,
      budgetOverruns,
      falseAcceptanceTolerance: FALSE_ACCEPTANCE_TOLERANCE,
      maeComparisonTolerance: MAE_COMPARISON_TOLERANCE,
      absoluteMaeFloor: ASAD_MAE_ABSOLUTE_FLOOR
    }
  };
  const summaryFile = `summary-${runId}.json`;
  fs.writeFileSync(path.join(outputDirectory, summaryFile), `${JSON.stringify(summary, null, 2)}\n`);
  if (options.pilot && options.seeds === 10
    && DIMENSIONS.every(dimension => options.dimensions.includes(dimension))
    && FOCUS_SCENARIOS.every(scenario => options.scenarios.includes(scenario))) {
    const reportPath = path.resolve('docs/research/2026-10-29-gfd-vp-pilot-report.md');
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, renderReport(summary, true));
  } else if (!options.pilot && options.seeds >= 100
    && DIMENSIONS.every(dimension => options.dimensions.includes(dimension))
    && FOCUS_SCENARIOS.every(scenario => options.scenarios.includes(scenario))) {
    const reportPath = path.resolve('docs/research/2026-10-29-gfd-vp-experimental-report.md');
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, renderReport(summary));
  }
  const marker = options.pilot ? 'GFD_PILOT_COMPLETE' : 'GFD_CAMPAIGN_COMPLETE';
  console.log(`${marker} run=${runId} runs=${records.length} seeds=${options.seeds} verdict=${verdict} budgetOverruns=${budgetOverruns} diagnosticMismatches=${diagnosticMismatches} computeDevice=${cudaWorker?.device.computeDevice ?? 'none'} deviceName=${cudaWorker?.device.deviceName ?? 'none'} raw=${rawFile} summary=${summaryFile}`);
}

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
