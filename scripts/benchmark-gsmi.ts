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
import { GfdCudaClient } from './gfd-vp-cuda-client';
import {
  decompileBlackBoxReluNdGlobalSpectralMomentInversion,
  GlobalSpectralMomentModel,
  predictBlackBoxReluNdGlobalSpectralMomentInversion
} from '../src/solvers/global-spectral-moment-inversion';
import { GsiCudaClient } from './gsi-cuda-client';
import {
  createTeacher,
  makeScenario,
  matchPlanes,
  mean,
  randomTestPoints,
  SCENARIOS,
  Scenario,
  ScenarioName,
  solverBudget
} from './benchmark-asad';

type Method =
  | 'asad' | 'ransac' | 'star-probe' | 'adaptive' | 'bcad' | 'mbcd-full'
  | 'gfd-profiled' | 'gsmi-full' | 'gsmi-second-only';
type GsmiMethod = Extract<Method, `gsmi-${string}`>;
type Plane = { normal: number[]; offset: number };

type RawRun = {
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
  gpuDiagnostics?: {
    computeDevice: string;
    deviceName: string;
    cudaMemoryAllocatedBytes: number;
    gpuTimeMs: number;
  };
  gsmiDiagnostics?: {
    sampleCount: number;
    sampleSigma: number;
    momentOrder: string;
    estimatedRank: number;
    rankThreshold: number;
    absoluteMomentEigenvalues: number[];
    unresolvedComponents: number;
    matrixPencilResidual: number | null;
    computeDevice: string;
    deviceName: string;
    cudaMemoryAllocatedBytes: number;
    momentAndInversionTimeMs: number;
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
  meanEstimatedRank: number | null;
  meanPencilResidual: number | null;
};

type CellResult = {
  dimension: number;
  scenario: ScenarioName;
  gsmiMae: number | null;
  bestBaseline: Method | null;
  bestBaselineMae: number | null;
  gsmiCompleteRate: number | null;
  gsmiOracleCalls: number | null;
  falseAcceptedRunRate: number | null;
};

type CampaignSummary = {
  runId: string;
  seedsPerCell: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  methods: Method[];
  totalRuns: number;
  totalOracleBudgetOverruns: number;
  totalDiagnosticMismatches: number;
  statusTotals: { complete: number; incomplete: number; threw: number };
  rawFile: string;
  summary: SummaryRow[];
  gsmiCells: CellResult[];
  verdict: 'promising' | 'not-supported';
};

const DIMENSIONS = [16, 32];
const REQUIRED_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank'
];
const METHODS: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full',
  'gfd-profiled', 'gsmi-full', 'gsmi-second-only'
];
const BASELINES: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full', 'gfd-profiled'
];
const PRIMARY_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank', 'multi-crossing-low-rank', 'small-amplitude-low-rank'
];

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function parseArgs(args: string[]): {
  seeds: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  methods: Method[];
  output: string;
} {
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index >= 0 ? args[index + 1] : undefined;
  };
  const seeds = Number(value('--seeds') ?? 10);
  if (!Number.isSafeInteger(seeds) || seeds < 1) throw new Error('--seeds must be positive');
  const dimensions = (value('--dimensions') ?? DIMENSIONS.join(',')).split(',').map(Number);
  const scenarios = (value('--scenarios') ?? REQUIRED_SCENARIOS.join(',')).split(',') as ScenarioName[];
  const methods = (value('--methods') ?? METHODS.join(',')).split(',') as Method[];
  if (!dimensions.length || dimensions.some(item => !DIMENSIONS.includes(item))) {
    throw new Error(`--dimensions must be selected from ${DIMENSIONS.join(',')}`);
  }
  if (!scenarios.length || scenarios.some(item => !REQUIRED_SCENARIOS.includes(item))) {
    throw new Error(`--scenarios must be selected from ${REQUIRED_SCENARIOS.join(',')}`);
  }
  if (!methods.length || methods.some(item => !METHODS.includes(item))) {
    throw new Error(`--methods must be selected from ${METHODS.join(',')}`);
  }
  return {
    seeds,
    dimensions: [...new Set(dimensions)],
    scenarios: [...new Set(scenarios)],
    methods: [...new Set(methods)],
    output: value('--output') ?? 'scratch/results/gsmi/pilot'
  };
}

function geometryMetrics(planes: Plane[], scenario: Scenario): {
  correct: number;
  falseAccepted: number;
  missed: number;
  meanAngle: number | null;
  meanOffset: number | null;
} {
  const result = matchPlanes(planes, scenario.terms);
  let offsetSum = 0;
  const pairs: Array<{ predicted: number; truth: number; score: number }> = [];
  for (let predicted = 0; predicted < planes.length; predicted++) {
    for (let truth = 0; truth < scenario.terms.length; truth++) {
      const first = planes[predicted];
      const second = scenario.terms[truth];
      const dot = first.normal.reduce((sum, value, axis) => sum + value * second.normal[axis], 0);
      const firstNorm = Math.sqrt(first.normal.reduce((sum, value) => sum + value * value, 0));
      const secondNorm = Math.sqrt(second.normal.reduce((sum, value) => sum + value * value, 0));
      const angle = Math.acos(Math.min(1, Math.abs(dot / Math.max(1e-12, firstNorm * secondNorm))));
      const sign = dot < 0 ? -1 : 1;
      const offsetError = Math.abs(sign * first.offset - second.offset);
      pairs.push({ predicted, truth, score: angle + 100 * offsetError });
    }
  }
  pairs.sort((a, b) => a.score - b.score);
  const usedPredicted = new Set<number>();
  const usedTruth = new Set<number>();
  for (const pair of pairs) {
    if (usedPredicted.has(pair.predicted) || usedTruth.has(pair.truth)) continue;
    const first = planes[pair.predicted];
    const second = scenario.terms[pair.truth];
    const dot = first.normal.reduce((sum, value, axis) => sum + value * second.normal[axis], 0);
    const firstNorm = Math.sqrt(first.normal.reduce((sum, value) => sum + value * value, 0));
    const secondNorm = Math.sqrt(second.normal.reduce((sum, value) => sum + value * value, 0));
    const angle = Math.acos(Math.min(1, Math.abs(dot / Math.max(1e-12, firstNorm * secondNorm))));
    const offsetError = Math.abs((dot < 0 ? -first.offset : first.offset) - second.offset);
    if (angle > 10 * Math.PI / 180 || offsetError > 0.1) continue;
    usedPredicted.add(pair.predicted);
    usedTruth.add(pair.truth);
    offsetSum += offsetError;
  }
  return {
    ...result,
    meanOffset: usedTruth.size
      ? (offsetSum + 1 * (scenario.terms.length - usedTruth.size)) / scenario.terms.length
      : scenario.terms.length ? 1 : 0
  };
}

async function runOne(
  method: Method,
  scenario: Scenario,
  dimension: number,
  seed: number,
  gfdWorker: GfdCudaClient | null,
  gsmiWorker: GsiCudaClient | null
): Promise<RawRun> {
  const teacher = createTeacher(scenario);
  let actualOracleCalls = 0;
  const countedOracle = (input: number[]): number => {
    actualOracleCalls++;
    return teacher(input);
  };
  const { budget, lineCount } = solverBudget(dimension, scenario.terms.length);
  const scenarioIndex = SCENARIOS.indexOf(scenario.name);
  const solverSeed = seed * 104729 + dimension * 13007 + scenarioIndex * 8191;
  const points = randomTestPoints(seed, dimension);
  const start = performance.now();
  let predictor: ((input: number[]) => number) | null = null;
  let planes: Plane[] = [];
  let diagnosticOracleCalls: number | null = null;
  let gpuDiagnostics: RawRun['gpuDiagnostics'];
  let gsmiDiagnostics: RawRun['gsmiDiagnostics'];
  let error: string | undefined;
  let explicitComplete: boolean | undefined;
  try {
    if (method.startsWith('gsmi-')) {
      if (!gsmiWorker) throw new Error('GSMI requires the CUDA worker');
      const model: GlobalSpectralMomentModel = await decompileBlackBoxReluNdGlobalSpectralMomentInversion(
        countedOracle,
        {
          dimension,
          oracleBudget: budget,
          randomSeed: solverSeed,
          sampleCount: Math.min(4096, Math.floor(budget * 0.6)),
          sampleSigma: 0.2,
          maximumAbsoluteInput: 1,
          momentOrder: method === 'gsmi-second-only' ? 'second-only' : 'second-and-third'
        },
        gsmiWorker
      );
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      explicitComplete = model.reconstructionComplete;
      planes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdGlobalSpectralMomentInversion(model, input);
      gpuDiagnostics = {
        computeDevice: model.diagnostics.computeDevice,
        deviceName: model.diagnostics.deviceName,
        cudaMemoryAllocatedBytes: model.diagnostics.cudaMemoryAllocatedBytes,
        gpuTimeMs: model.diagnostics.momentAndInversionTimeMs
      };
      gsmiDiagnostics = {
        sampleCount: model.diagnostics.sampleCount,
        sampleSigma: model.diagnostics.sampleSigma,
        momentOrder: model.diagnostics.momentOrder,
        estimatedRank: model.diagnostics.estimatedRank,
        rankThreshold: model.diagnostics.rankThreshold,
        absoluteMomentEigenvalues: model.diagnostics.absoluteMomentEigenvalues,
        unresolvedComponents: model.diagnostics.unresolvedComponents,
        matrixPencilResidual: model.diagnostics.matrixPencilResidual,
        computeDevice: model.diagnostics.computeDevice,
        deviceName: model.diagnostics.deviceName,
        cudaMemoryAllocatedBytes: model.diagnostics.cudaMemoryAllocatedBytes,
        momentAndInversionTimeMs: model.diagnostics.momentAndInversionTimeMs
      };
    } else if (method === 'gfd-profiled') {
      if (!gfdWorker) throw new Error('GFD comparator requires its CUDA worker');
      const config: GlobalFunctionDistillationConfig = {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: solverSeed,
        sampleCount: Math.min(4096, Math.floor(budget * 0.6)),
        restarts: 4,
        optimizationSteps: 220,
        batchSize: 256
      };
      const inputs = createGlobalFunctionDistillationSampleInputs(config);
      const outputs = inputs.map(countedOracle);
      const model = await gfdWorker.fit({
        inputs, outputs, dimension, termCount: scenario.terms.length,
        randomSeed: solverSeed, restarts: 4, optimizationSteps: 220,
        batchSize: 256, profileOutputLayer: true
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      explicitComplete = model.reconstructionComplete;
      planes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      gpuDiagnostics = {
        computeDevice: model.diagnostics.computeDevice,
        deviceName: gfdWorker.device.deviceName,
        cudaMemoryAllocatedBytes: model.diagnostics.cudaMemoryAllocatedBytes,
        gpuTimeMs: model.diagnostics.trainingTimeMs
      };
      predictor = input => {
        let value = model.baseIntercept;
        for (let axis = 0; axis < dimension; axis++) value += model.baseWeights[axis] * input[axis];
        for (const term of model.terms) {
          const preactivation = term.normal.reduce((sum, item, axis) => sum + item * input[axis], term.offset);
          value += term.readout * Math.max(0, preactivation);
        }
        return value;
      };
      gsmiDiagnostics = undefined;
    } else if (method === 'mbcd-full') {
      const model = decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(countedOracle, {
        dimension, ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length, oracleBudget: budget, randomSeed: solverSeed,
        maxGradientProbes: 80, queryPolicy: 'contradiction-seeking',
        rankPolicy: 'retain-alternatives', validationPolicy: 'independent'
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      explicitComplete = model.reconstructionComplete;
      planes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdMultiHypothesisBoundaryContinuation(model, input);
    } else if (method === 'bcad') {
      const model = decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(countedOracle, {
        dimension, ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length, oracleBudget: budget, randomSeed: solverSeed,
        maxGradientProbes: 96
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      explicitComplete = model.reconstructionComplete;
      planes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdBoundaryConditionedActiveDiscovery(model, input);
    } else if (method === 'asad') {
      const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(countedOracle, {
        dimension, ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length, oracleBudget: budget, randomSeed: solverSeed,
        maxGradientProbes: 96
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      explicitComplete = model.reconstructionComplete;
      planes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdActiveSubspaceAdaptive(model, input);
    } else {
      const stageAMethod = method as 'ransac' | 'star-probe' | 'adaptive';
      const model: BlackBoxReluNdModel = decompileBlackBoxReluNdLineProbe(countedOracle, {
        dimension, ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length, stageAMethod, lineCount, samplesPerLine: 96,
        ransacIterations: 64, minSupport: Math.max(dimension + 1, Math.floor(lineCount * 0.15)),
        randomSeed: solverSeed, oracleBudget: budget
      });
      diagnosticOracleCalls = model.adaptiveDiagnostics?.totalOracleCalls ?? null;
      explicitComplete = model.reconstructionComplete ?? true;
      planes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdLineProbeModel(model, input);
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }
  const runtimeMs = performance.now() - start;
  const predictions = predictor ? points.map(predictor) : null;
  const mae = predictions === null
    ? null
    : mean(points.map((point, index) => Math.abs(teacher(point) - predictions[index])));
  const geometry = geometryMetrics(planes, scenario);
  const reconstructionComplete = explicitComplete === true
    && geometry.correct === scenario.terms.length
    && geometry.falseAccepted === 0
    && geometry.missed === 0;
  const status = error ? 'threw' : reconstructionComplete ? 'complete' : 'incomplete';
  return {
    schemaVersion: 1, seed, dimension, scenario: scenario.name, method, solverSeed,
    scenarioModelHash: hash(scenario), evaluationPointsHash: hash(points), status,
    oracleBudget: budget, actualOracleCalls, diagnosticOracleCalls,
    oracleCountReconciled: diagnosticOracleCalls === null
      ? status === 'threw'
      : diagnosticOracleCalls === actualOracleCalls,
    oracleBudgetExceeded: actualOracleCalls > budget,
    runtimeMs, mae, meanAngularErrorDeg: geometry.meanAngle,
    meanOffsetError: geometry.meanOffset, correctHyperplanes: geometry.correct,
    falseAcceptedHyperplanes: geometry.falseAccepted, missedHyperplanes: geometry.missed,
    recoveredHyperplanes: planes.length, reconstructionComplete,
    ...(gpuDiagnostics ? { gpuDiagnostics } : {}),
    ...(gsmiDiagnostics ? { gsmiDiagnostics } : {}),
    ...(error ? { error } : {})
  };
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function summarize(records: RawRun[]): SummaryRow[] {
  const keys = new Map<string, RawRun[]>();
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    keys.set(key, [...(keys.get(key) ?? []), record]);
  }
  return [...keys.entries()].map(([key, rows]) => {
    const [dimension, scenario, method] = key.split('|');
    const scored = rows.filter(row => row.mae !== null);
    const completeRuns = rows.filter(row => row.reconstructionComplete).length;
    const gpuRows = rows.flatMap(row => row.gpuDiagnostics ? [row.gpuDiagnostics] : []);
    const gsmiRows = rows.flatMap(row => row.gsmiDiagnostics ? [row.gsmiDiagnostics] : []);
    return {
      dimension: Number(dimension), scenario: scenario as ScenarioName, method: method as Method,
      runs: rows.length, scoredRuns: scored.length, completeRuns,
      incompleteRuns: rows.filter(row => row.status === 'incomplete').length,
      thrownRuns: rows.filter(row => row.status === 'threw').length,
      completionRate: completeRuns / Math.max(1, rows.length),
      meanMae: scored.length ? average(scored.map(row => row.mae!)) : null,
      meanOracleCalls: average(rows.map(row => row.actualOracleCalls)),
      meanNormalErrorDeg: scored.length
        ? average(scored.map(row => row.meanAngularErrorDeg ?? 90)) : null,
      meanOffsetError: scored.length ? average(scored.map(row => row.meanOffsetError ?? 1)) : null,
      meanCorrectHyperplanes: average(rows.map(row => row.correctHyperplanes)),
      meanFalseAcceptedHyperplanes: average(rows.map(row => row.falseAcceptedHyperplanes)),
      falseAcceptedRunRate: scored.length
        ? scored.filter(row => row.falseAcceptedHyperplanes > 0).length / scored.length : null,
      meanRuntimeMs: average(rows.map(row => row.runtimeMs)),
      meanCudaMemoryAllocatedBytes: gpuRows.length
        ? average(gpuRows.map(row => row.cudaMemoryAllocatedBytes)) : null,
      meanGpuTrainingMs: gpuRows.length ? average(gpuRows.map(row => row.gpuTimeMs)) : null,
      meanEstimatedRank: gsmiRows.length ? average(gsmiRows.map(row => row.estimatedRank)) : null,
      meanPencilResidual: gsmiRows.length
        ? average(gsmiRows.flatMap(row => row.matrixPencilResidual === null ? [] : [row.matrixPencilResidual]))
        : null
    };
  }).sort((a, b) => a.dimension - b.dimension
    || a.scenario.localeCompare(b.scenario) || a.method.localeCompare(b.method));
}

function cellResults(rows: SummaryRow[]): CellResult[] {
  return DIMENSIONS.flatMap(dimension => PRIMARY_SCENARIOS.map(scenario => {
    const gsmi = rows.find(row => row.dimension === dimension && row.scenario === scenario && row.method === 'gsmi-full');
    const baselines = rows.filter(row => row.dimension === dimension && row.scenario === scenario
      && BASELINES.includes(row.method) && row.meanMae !== null);
    baselines.sort((first, second) => first.meanMae! - second.meanMae!);
    return {
      dimension, scenario, gsmiMae: gsmi?.meanMae ?? null,
      bestBaseline: baselines[0]?.method ?? null,
      bestBaselineMae: baselines[0]?.meanMae ?? null,
      gsmiCompleteRate: gsmi?.completionRate ?? null,
      gsmiOracleCalls: gsmi?.meanOracleCalls ?? null,
      falseAcceptedRunRate: gsmi?.falseAcceptedRunRate ?? null
    };
  }));
}

export function renderReport(summary: CampaignSummary): string {
  const fullRows = summary.summary.filter(row => row.method === 'gsmi-full');
  const primaryFullRows = fullRows.filter(row => PRIMARY_SCENARIOS.includes(row.scenario));
  const fullCompletions = fullRows.reduce((total, row) => total + row.completeRuns, 0);
  const fullRuns = fullRows.reduce((total, row) => total + row.runs, 0);
  const primaryCompletions = primaryFullRows.reduce((total, row) => total + row.completeRuns, 0);
  const primaryRuns = primaryFullRows.reduce((total, row) => total + row.runs, 0);
  const residuals = primaryFullRows
    .map(row => row.meanPencilResidual)
    .filter((value): value is number => value !== null);
  const residualRange = residuals.length
    ? `${Math.min(...residuals).toFixed(3)}–${Math.max(...residuals).toFixed(3)}`
    : 'unavailable';
  const fullRankRows = fullRows.filter(row => row.scenario === 'orthogonal-full-rank');
  const fullRankSummary = fullRankRows.map(row =>
    `${row.dimension}D rank ${row.meanEstimatedRank?.toFixed(2) ?? 'n/a'}, `
      + `${row.meanFalseAcceptedHyperplanes.toFixed(2)} false planes/run, `
      + `${(row.completionRate * 100).toFixed(0)}% complete`
  ).join('; ');
  const secondOnlyRows = summary.summary.filter(row => row.method === 'gsmi-second-only');
  const secondOnlyCompletions = secondOnlyRows.reduce((total, row) => total + row.completeRuns, 0);
  const secondOnlyRuns = secondOnlyRows.reduce((total, row) => total + row.runs, 0);
  const secondOnlyBetterCells = fullRows.filter(full => {
    const ablation = secondOnlyRows.find(row =>
      row.dimension === full.dimension && row.scenario === full.scenario);
    return ablation?.meanMae !== null && ablation?.meanMae !== undefined
      && full.meanMae !== null && ablation.meanMae < full.meanMae;
  }).length;
  const cells = summary.gsmiCells.map(cell =>
    `| ${cell.dimension} | ${cell.scenario} | ${cell.bestBaseline ?? 'n/a'} | ${cell.gsmiMae?.toExponential(4) ?? 'n/a'} | ${cell.bestBaselineMae?.toExponential(4) ?? 'n/a'} | ${((cell.gsmiCompleteRate ?? 0) * 100).toFixed(1)}% | ${cell.gsmiOracleCalls?.toFixed(0) ?? 'n/a'} | ${((cell.falseAcceptedRunRate ?? 0) * 100).toFixed(1)}% |`
  ).join('\n');
  const metrics = summary.summary.map(row =>
    `| ${row.dimension} | ${row.scenario} | ${row.method} | ${row.scoredRuns}/${row.runs} | ${row.meanMae?.toExponential(4) ?? 'n/a'} | ${(row.completionRate * 100).toFixed(1)}% | ${row.meanOracleCalls.toFixed(0)} | ${row.meanNormalErrorDeg?.toFixed(2) ?? 'n/a'} | ${row.meanOffsetError?.toExponential(3) ?? 'n/a'} | ${row.meanFalseAcceptedHyperplanes.toFixed(2)} | ${row.meanRuntimeMs.toFixed(1)} | ${row.meanCudaMemoryAllocatedBytes === null ? 'n/a' : (row.meanCudaMemoryAllocatedBytes / 1048576).toFixed(2)} | ${row.meanGpuTrainingMs?.toFixed(1) ?? 'n/a'} | ${row.meanEstimatedRank?.toFixed(2) ?? 'n/a'} | ${row.meanPencilResidual?.toExponential(3) ?? 'n/a'} | ${row.thrownRuns} |`
  ).join('\n');
  return `# Global Spectral-Moment Inversion (GSMI) pilot

**Run:** ${summary.runId}  
**Verdict:** ${summary.verdict}  
**Seeds per cell:** ${summary.seedsPerCell}  
**Dimensions:** ${summary.dimensions.join(', ')}  
**Scenarios:** ${summary.scenarios.join(', ')}  
**Methods:** ${summary.methods.join(', ')}  
**Raw runs:** [${summary.rawFile}](../../scratch/results/gsmi/pilot/${summary.rawFile})

## Mathematical derivation

Consider the globally defined scalar function

\`f(x)=c+b^T x+Σ_j a_j (n_j^T x+t_j)_+\`, with \`||n_j||₂=1\`.
In the distributional sense,
\`∂²f/∂x∂x^T=Σ_j a_j n_j n_j^T δ(n_j^T x+t_j)\`.
Use the Fourier convention \`F[g](ω)=∫g(x)e^{-iω^Tx}dx\`. A plane term has Fourier support on the line parallel to \`n_j\`; along \`ω=λ n_j\`, its phase is \`e^{i λ t_j}\`. Thus exact unwindowed distributional Fourier data separates nonparallel plane orientations by support and encodes offsets in phase. Finite windows broaden each line, so directions separated by less than their reciprocal spatial width are poorly conditioned.

GSMI uses Gaussian-windowed spatial moments, equivalent to derivatives at zero frequency. Let \`ρσ(x)\` be the \`N(0,σ²I)\` density and define \`M⁽⁰⁾_{pq}=⟨∂pq f,ρσ⟩\`, \`M⁽¹⁾_{pqr}=⟨∂pq f,x_rρσ⟩\`. Integration by parts gives estimators from values alone:

\`\`\`
M⁽⁰⁾pq = E[f(X)(Xp Xq/σ⁴ − δpq/σ²)]
M⁽¹⁾pqr = E[f(X)(Xp Xq Xr/σ⁴ −
                   (δpq Xr + δpr Xq + δqr Xp)/σ²)]
\`\`\`

For a plane, Gaussian conditioning gives \`∫δ(nᵀx+t)ρσ(x)dx=φσ(t)\` and \`∫xδ(nᵀx+t)ρσ(x)dx=−t n φσ(t)\`. Therefore

\`\`\`
M⁽⁰⁾ = N diag(wj) Nᵀ,       wj = aj φσ(tj)
M⁽¹⁾[:,:,v] = N diag(wj (qjᵀv)) Nᵀ,   qj = −tj nj
\`\`\`

On the signal subspace, a matrix pencil \`(M⁽¹⁾[:,:,v], M⁽⁰⁾)\` has eigenvalues \`qjᵀv\`; repeating contractions in coordinate directions yields \`qj\`. For \`qj ≠ 0\`, \`nj=qj/||qj||\` and the gauge-invariant offset is \`tj=−||qj||\`. The plane Hessian weight then determines \`aj=wj/φσ(tj)\`. A final linear least-squares readout fits the affine part and amplitudes from the same global observations; it does not choose geometry.

## Identifiability and falsifiable predictions

- **Mathematical claim:** with distinct components, nonzero \`wj\`, linearly independent normals, nonzero offsets, and a non-colliding generic pencil spectrum, exact \`M⁽⁰⁾,M⁽¹⁾\` identify the unoriented planes and Hessian amplitudes up to component permutation. The ReLU affine gauge remains unidentifiable, so the final affine readout is not uniquely attributable to individual units.
- **Non-identifiable cases:** \`t=0\` makes \`q=0\`; third moments contain no normal direction for that component. A rank-deficient or cancelling signed \`M⁽⁰⁾\`, repeated pencil eigenvalues, more components than identifiable signal rank, and exactly coincident planes likewise defeat this inversion. The second-moment-only ablation cannot separate non-orthogonal components from their aggregate matrix.
- **Near-parallel prediction:** normals with small separation make \`N\` ill-conditioned and pencil eigenvectors unstable; finite-sample error should increase before exact orientation information disappears. GSMI is not expected to resolve arbitrarily close planes at a fixed sample budget.
- **Contamination prediction:** a smooth local bump has a continuous Hessian spectrum, not a finite sum of plane atoms; a finite-rank pencil may create false components or leave a large residual.
- **Sampling approximation:** the worker samples a Gaussian with σ=0.2, rejecting points outside the cube [-1,1]^d. At the configured five-sigma bound, the omitted Gaussian tail is small but nonzero; the derivation is exact for an untruncated Gaussian, so this finite-domain implementation has a stated tail approximation.

## Paired pilot results

All methods use the same deterministic teacher, solver seed, oracle ceiling, and independent 192-point test set per seed/dimension/scenario. GSMI receives only oracle values, dimension, query budget, and fixed sampling policy; scenario construction and hidden terms are used only by the benchmark evaluator. GFD-VP remains an unchanged comparator. The pilot has ${summary.seedsPerCell} seeds per cell and is screening evidence, not 100-seed confirmation.

| D | Scenario | Best existing baseline | GSMI MAE | Baseline MAE | GSMI complete rate | Oracle calls | False-accepted run rate |
|---:|---|---|---:|---:|---:|---:|---:|
${cells}

## All-seed method metrics

| D | Scenario | Method | Scored/runs | MAE | Complete | Oracle calls | Normal error ° | Offset error | False planes/run | Runtime ms | VRAM MiB | GPU moment ms | Estimated rank | Pencil residual | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${metrics}

**Run status totals:** ${summary.statusTotals.complete} complete, ${summary.statusTotals.incomplete} incomplete, ${summary.statusTotals.threw} threw.  
**Budget overruns:** ${summary.totalOracleBudgetOverruns}; **oracle diagnostic mismatches:** ${summary.totalDiagnosticMismatches}.

## Ablations

\`gsmi-second-only\` removes the third Gaussian-windowed moment and its matrix-pencil inversion. It uses only eigenvectors of the second moment, which are aggregate directions rather than guaranteed individual planes. Comparing it with \`gsmi-full\` tests whether offset-bearing third moments and pencil inversion add identifiable geometry. Both variants use the same queried samples and CUDA worker.

## Pilot failure analysis

The full GSMI method completed ${fullCompletions}/${fullRuns} reconstructions overall and ${primaryCompletions}/${primaryRuns} in the three primary low-rank scenarios. Across those primary cells, the mean relative moment-reconstruction residual ranged from ${residualRange}. The full-rank controls reported ${fullRankSummary || 'no rows'}. The second-moment-only ablation completed ${secondOnlyCompletions}/${secondOnlyRuns}; it had lower mean MAE than the full method in ${secondOnlyBetterCells}/${fullRows.length} cells, but did not complete the geometry in those cases.

These observations locate the empirical failure in turning estimated low-order moments into plane geometry: the full method's reconstructed moments disagree strongly with the sampled moments, and omitting the pencil improves some function-value errors without recovering complete geometry. This is consistent with finite-sample moment error amplified by the ill-conditioned inversion, but this pilot did not isolate sampling variance, bounded-domain truncation, and structural non-identifiability with separate ablations. The full-rank controls also do not show reliable rank recovery. Do not treat the low oracle count as evidence of successful geometric reconstruction.

## Conclusion and next experiment

${summary.verdict === 'promising'
    ? 'The pilot meets its predeclared geometric screen: at least half the seeds in one same primary scenario complete at both dimensions, and each full-rank control cell has at most one false-accepted run and at most one mean false plane. This permits consideration of a larger confirmatory campaign; it does not establish general success.'
    : 'The pilot does not pass the required pre-campaign screen: no qualifying hard-case geometry/control pair is demonstrated. Do not start a larger seed campaign unchanged. The next experiment should change the mathematical information source—for example, test finite-frequency Fourier-ridge support and phase estimates directly against the same global-query controls—rather than tune rank or acceptance thresholds. First determine whether that representation separates sampling error from the observed pencil-conditioning failure.'}

The pilot verdict is computed from all raw runs and the declared screen; tests establish implementation/accounting behavior, not reconstruction accuracy. No raw results from earlier campaigns are modified.
`;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const projectRoot = path.resolve(__dirname, '..');
  const outputDirectory = path.resolve(options.output);
  fs.mkdirSync(outputDirectory, { recursive: true });
  const needsGfd = options.methods.includes('gfd-profiled');
  const needsGsmi = options.methods.some(method => method.startsWith('gsmi-'));
  const gfdWorker = needsGfd ? await GfdCudaClient.start(projectRoot) : null;
  const gsmiWorker = needsGsmi ? await GsiCudaClient.start(projectRoot) : null;
  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const rawFile = `gsmi-${runId}-raw.jsonl`;
  const rawPath = path.join(outputDirectory, rawFile);
  const records: RawRun[] = [];
  const total = options.seeds * options.dimensions.length * options.scenarios.length * options.methods.length;
  try {
    for (const dimension of options.dimensions) {
      for (const scenarioName of options.scenarios) {
        for (let seed = 1; seed <= options.seeds; seed++) {
          const scenario = makeScenario(scenarioName, dimension, seed);
          for (const method of options.methods) {
            const record = await runOne(method, scenario, dimension, seed, gfdWorker, gsmiWorker);
            records.push(record);
            fs.appendFileSync(rawPath, `${JSON.stringify(record)}\n`);
            if (records.length % 50 === 0 || records.length === total) {
              process.stdout.write(`GSMI_PROGRESS ${records.length}/${total} d=${dimension} scenario=${scenarioName} seed=${seed} method=${method}\n`);
            }
          }
        }
      }
    }
  } finally {
    if (gfdWorker) await gfdWorker.close();
    if (gsmiWorker) await gsmiWorker.close();
  }
  const summaryRows = summarize(records);
  const cells = cellResults(summaryRows);
  const overrun = records.filter(record => record.oracleBudgetExceeded).length;
  const mismatches = records.filter(record => !record.oracleCountReconciled).length;
  const primaryComplete = PRIMARY_SCENARIOS.some(scenario =>
    DIMENSIONS.every(dimension => cells.some(cell => cell.dimension === dimension
      && cell.scenario === scenario && (cell.gsmiCompleteRate ?? 0) >= 0.5)));
  const controls = summaryRows.filter(row => row.method === 'gsmi-full'
    && row.scenario === 'orthogonal-full-rank');
  const controlGuard = controls.length === 2
    && controls.every(row => (row.falseAcceptedRunRate ?? 1) <= 0.1
      && row.meanFalseAcceptedHyperplanes <= 1);
  const verdict: CampaignSummary['verdict'] = primaryComplete && controlGuard
    && overrun === 0 && mismatches === 0 ? 'promising' : 'not-supported';
  const summary: CampaignSummary = {
    runId, seedsPerCell: options.seeds, dimensions: options.dimensions,
    scenarios: options.scenarios, methods: options.methods, totalRuns: records.length,
    totalOracleBudgetOverruns: overrun, totalDiagnosticMismatches: mismatches,
    statusTotals: {
      complete: records.filter(record => record.status === 'complete').length,
      incomplete: records.filter(record => record.status === 'incomplete').length,
      threw: records.filter(record => record.status === 'threw').length
    },
    rawFile, summary: summaryRows, gsmiCells: cells, verdict
  };
  const summaryFile = `summary-${runId}.json`;
  fs.writeFileSync(path.join(outputDirectory, summaryFile), `${JSON.stringify(summary, null, 2)}\n`);
  const reportPath = path.resolve('docs/research/2026-10-30-global-spectral-inversion-pilot.md');
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, renderReport(summary));
  console.log(`GSMI_PILOT_COMPLETE run=${runId} runs=${records.length} seeds=${options.seeds} verdict=${verdict} budgetOverruns=${overrun} diagnosticMismatches=${mismatches} gsiDevice=${gsmiWorker?.device.computeDevice ?? 'none'} deviceName=${gsmiWorker?.device.deviceName ?? 'none'} raw=${rawFile} summary=${summaryFile}`);
}

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
