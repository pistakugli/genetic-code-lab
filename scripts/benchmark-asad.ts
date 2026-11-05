import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
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
  ASAD_MAE_ABSOLUTE_FLOOR,
  ASAD_MAE_COMPARISON_TOLERANCE,
  exceedsAllowedMaeRegression,
  maeValuesAreComparable,
  oracleCallReduction
} from './asad-benchmark-metrics';

export type ScenarioName =
  | 'isolated-low-rank'
  | 'multi-crossing-low-rank'
  | 'near-parallel-low-rank'
  | 'small-amplitude-low-rank'
  | 'contaminated-local-probes'
  | 'orthogonal-full-rank'
  | 'dense-crossing-full-rank';

export type MethodName = 'asad' | 'ransac' | 'star-probe' | 'adaptive';

export type TeacherTerm = { normal: number[]; offset: number; readout: number };
export type Scenario = {
  name: ScenarioName;
  terms: TeacherTerm[];
  baseIntercept: number;
  baseWeights: number[];
  expectedActiveRank: number;
  contaminationCenter?: number[];
  contaminationAmplitude?: number;
  contaminationWidth?: number;
};

type RunRecord = {
  schemaVersion: 1;
  seed: number;
  dimension: number;
  scenario: ScenarioName;
  method: MethodName;
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
  asadDiagnostics?: {
    rankUncertainty: number;
    activeSubspaceEnergyRatio: number;
    hypothesisDisagreement: number;
    candidateUncertainty: number;
    disagreementQueries: number;
    activeSubspaceQueries: number;
    orthogonalExplorationQueries: number;
    globalExplorationQueries: number;
    strategySwitches: number;
    fitResidualRms: number | null;
    fitResidualRelative: number | null;
    fitValidationObservations: number;
    requiredFitValidationObservations: number;
  };
  error?: string;
};

const DIMENSIONS = [6, 8, 16, 32];
export const SCENARIOS: ScenarioName[] = [
  'isolated-low-rank',
  'multi-crossing-low-rank',
  'near-parallel-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank',
  'dense-crossing-full-rank'
];
const METHODS: MethodName[] = ['asad', 'ransac', 'star-probe', 'adaptive'];
const ANGLE_TOLERANCE_DEGREES = 10;
const OFFSET_TOLERANCE = 0.1;
const EVALUATION_POINTS = 192;

function parseArgs(args: string[]): { seeds: number; dimensions: number[]; scenarios: ScenarioName[]; output: string } {
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index >= 0 ? args[index + 1] : undefined;
  };
  const seeds = Number(value('--seeds') ?? 100);
  if (!Number.isSafeInteger(seeds) || seeds < 1) throw new Error('--seeds must be a positive safe integer');
  const dimensions = (value('--dimensions') ?? DIMENSIONS.join(','))
    .split(',').map(Number);
  if (!dimensions.length || dimensions.some(dimension => !DIMENSIONS.includes(dimension))) {
    throw new Error(`--dimensions must be selected from ${DIMENSIONS.join(',')}`);
  }
  const scenarios = (value('--scenarios') ?? SCENARIOS.join(','))
    .split(',') as ScenarioName[];
  if (!scenarios.length || scenarios.some(scenario => !SCENARIOS.includes(scenario))) {
    throw new Error(`--scenarios must be selected from ${SCENARIOS.join(',')}`);
  }
  return {
    seeds,
    dimensions: [...new Set(dimensions)],
    scenarios: [...new Set(scenarios)],
    output: value('--output') ?? 'scratch/results/asad'
  };
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function gaussian(random: () => number): number {
  const first = Math.max(1e-12, random());
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random());
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, axis) => sum + value * b[axis], 0);
}

function normalize(vector: number[]): number[] {
  const length = Math.sqrt(dot(vector, vector));
  return length > 1e-12 ? vector.map(value => value / length) : vector.map((_, axis) => axis === 0 ? 1 : 0);
}

function randomUnit(random: () => number, dimension: number): number[] {
  return normalize(Array.from({ length: dimension }, () => gaussian(random)));
}

function orthogonalize(vector: number[], basis: number[][]): number[] {
  const result = [...vector];
  for (const direction of basis) {
    const projection = dot(result, direction);
    for (let axis = 0; axis < result.length; axis++) result[axis] -= projection * direction[axis];
  }
  return result;
}

function randomOrthonormalBasis(random: () => number, dimension: number): number[][] {
  const basis: number[][] = [];
  for (let index = 0; index < dimension; index++) {
    let vector = orthogonalize(randomUnit(random, dimension), basis);
    if (Math.sqrt(dot(vector, vector)) < 1e-8) {
      vector = orthogonalize(Array.from({ length: dimension }, (_, axis) => axis === index ? 1 : 0), basis);
    }
    const length = Math.sqrt(dot(vector, vector));
    basis.push(vector.map(value => value / length));
  }
  return basis;
}

function randomReadout(random: () => number, index: number): number {
  return (random() < 0.5 ? -1 : 1) * (0.55 + 0.9 * random() + 0.08 * (index % 3));
}

export function makeScenario(name: ScenarioName, dimension: number, seed: number): Scenario {
  const random = makeRandom(seed * 1009 + dimension * 9176 + SCENARIOS.indexOf(name) * 65537);
  const basis = randomOrthonormalBasis(random, dimension);
  const baseWeights = Array.from({ length: dimension }, () => (random() - 0.5) * 0.18);
  const baseIntercept = (random() - 0.5) * 0.3;
  const term = (normal: number[], offset: number, readout: number): TeacherTerm => ({
    normal: normalize(normal),
    offset,
    readout
  });
  if (name === 'isolated-low-rank') {
    return { name, terms: [term(basis[0], (random() - 0.5) * 0.3, randomReadout(random, 0))], baseIntercept, baseWeights, expectedActiveRank: 1 };
  }
  if (name === 'multi-crossing-low-rank') {
    return {
      name,
      terms: [
        term(basis[0], -0.08, randomReadout(random, 0)),
        term(basis[1], 0.05, randomReadout(random, 1)),
        term(basis[2], -0.02, randomReadout(random, 2))
      ],
      baseIntercept,
      baseWeights,
      expectedActiveRank: 3
    };
  }
  if (name === 'near-parallel-low-rank') {
    const perpendicular = basis[1];
    return {
      name,
      terms: [
        term(basis[0], -0.11, randomReadout(random, 0)),
        term(basis[0].map((value, axis) => value + 0.035 * perpendicular[axis]), 0.11, randomReadout(random, 1))
      ],
      baseIntercept,
      baseWeights,
      expectedActiveRank: 2
    };
  }
  if (name === 'small-amplitude-low-rank') {
    return {
      name,
      terms: [term(basis[0], (random() - 0.5) * 0.3, 0.025)],
      baseIntercept,
      baseWeights,
      expectedActiveRank: 1
    };
  }
  if (name === 'contaminated-local-probes') {
    return {
      name,
      terms: [term(basis[0], (random() - 0.5) * 0.25, 0.8)],
      baseIntercept,
      baseWeights,
      expectedActiveRank: dimension,
      contaminationCenter: Array.from({ length: dimension }, () => (random() - 0.5) * 0.9),
      contaminationAmplitude: 0.12,
      contaminationWidth: 0.22
    };
  }
  const terms = basis.map((normal, index) =>
    term(normal, (random() - 0.5) * 0.28, randomReadout(random, index)));
  return { name, terms, baseIntercept, baseWeights, expectedActiveRank: dimension };
}

export function createTeacher(scenario: Scenario): (input: number[]) => number {
  return input => {
    let value = scenario.baseIntercept + dot(scenario.baseWeights, input);
    for (const term of scenario.terms) value += term.readout * Math.max(0, dot(term.normal, input) + term.offset);
    if (scenario.contaminationCenter && scenario.contaminationAmplitude && scenario.contaminationWidth) {
      const squaredDistance = input.reduce((sum, coordinate, axis) =>
        sum + (coordinate - scenario.contaminationCenter![axis]) ** 2, 0);
      value += scenario.contaminationAmplitude
        * Math.exp(-squaredDistance / (2 * scenario.contaminationWidth ** 2));
    }
    return value;
  };
}

export function randomTestPoints(seed: number, dimension: number): number[][] {
  const random = makeRandom(seed * 32452843 + dimension * 49979687 + 7727);
  return Array.from({ length: EVALUATION_POINTS }, () =>
    Array.from({ length: dimension }, () => 2 * random() - 1));
}

function angleDegrees(first: number[], second: number[]): number {
  const similarity = Math.min(1, Math.max(-1, Math.abs(dot(normalize(first), normalize(second)))));
  return Math.acos(similarity) * (180 / Math.PI);
}

function planeOffsetError(predicted: { normal: number[]; offset: number }, truth: TeacherTerm): number {
  const orientation = dot(normalize(predicted.normal), normalize(truth.normal)) < 0 ? -1 : 1;
  return Math.abs(orientation * predicted.offset - truth.offset);
}

export function matchPlanes(
  predicted: Array<{ normal: number[]; offset: number }>,
  truth: TeacherTerm[]
): { correct: number; falseAccepted: number; missed: number; meanAngle: number | null } {
  const pairs: Array<{ predictedIndex: number; truthIndex: number; angle: number; offsetError: number }> = [];
  for (let predictedIndex = 0; predictedIndex < predicted.length; predictedIndex++) {
    for (let truthIndex = 0; truthIndex < truth.length; truthIndex++) {
      pairs.push({
        predictedIndex,
        truthIndex,
        angle: angleDegrees(predicted[predictedIndex].normal, truth[truthIndex].normal),
        offsetError: planeOffsetError(predicted[predictedIndex], truth[truthIndex])
      });
    }
  }
  pairs.sort((a, b) => a.angle + 100 * a.offsetError - (b.angle + 100 * b.offsetError));
  const usedPredicted = new Set<number>();
  const usedTruth = new Set<number>();
  const matchedAngles: number[] = [];
  for (const pair of pairs) {
    if (pair.angle > ANGLE_TOLERANCE_DEGREES || pair.offsetError > OFFSET_TOLERANCE) continue;
    if (usedPredicted.has(pair.predictedIndex) || usedTruth.has(pair.truthIndex)) continue;
    usedPredicted.add(pair.predictedIndex);
    usedTruth.add(pair.truthIndex);
    matchedAngles.push(pair.angle);
  }
  return {
    correct: usedTruth.size,
    falseAccepted: predicted.length - usedPredicted.size,
    missed: truth.length - usedTruth.size,
    meanAngle: truth.length === 0
      ? null
      : (matchedAngles.reduce((sum, value) => sum + value, 0) + 90 * (truth.length - matchedAngles.length)) / truth.length
  };
}

export function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function percentile(values: number[], quantile: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(quantile * sorted.length))];
}

export function solverBudget(dimension: number, termCount: number): { budget: number; lineCount: number; fitSamples: number } {
  const lineCount = Math.max(64, 4 * dimension + 24);
  const fitSamples = Math.max(256, 80 * (dimension + termCount));
  const starProbeReserve = (dimension + 8) * (4 * dimension + 2);
  return {
    budget: lineCount * 96 + fitSamples + starProbeReserve + 256,
    lineCount,
    fitSamples
  };
}

function runOne(
  method: MethodName,
  scenario: Scenario,
  dimension: number,
  seed: number
): RunRecord {
  const teacher = createTeacher(scenario);
  const counted = (() => {
    let calls = 0;
    return {
      oracle: (input: number[]): number => {
        calls++;
        return teacher(input);
      },
      calls: (): number => calls
    };
  })();
  const { budget, lineCount } = solverBudget(dimension, scenario.terms.length);
  const modelSeed = seed * 104729 + dimension * 13007 + SCENARIOS.indexOf(scenario.name) * 8191;
  const evaluationPoints = randomTestPoints(seed, dimension);
  const start = performance.now();
  let reconstructionComplete = false;
  let diagnosticOracleCalls: number | null = null;
  let activeRankEstimate: number | null = null;
  let asadDiagnostics: RunRecord['asadDiagnostics'];
  let predictor: ((input: number[]) => number) | null = null;
  let recoveredPlanes: Array<{ normal: number[]; offset: number }> = [];
  let error: string | undefined;
  try {
    if (method === 'asad') {
      const model = decompileBlackBoxReluNdActiveSubspaceAdaptive(counted.oracle, {
        dimension,
        ranges: Array.from({ length: dimension }, () => [-1, 1] as [number, number]),
        termCount: scenario.terms.length,
        oracleBudget: budget,
        randomSeed: modelSeed,
        maxGradientProbes: 96
      });
      diagnosticOracleCalls = model.diagnostics.totalOracleCalls;
      activeRankEstimate = model.diagnostics.activeSubspaceRank;
      asadDiagnostics = {
        rankUncertainty: model.diagnostics.rankUncertainty,
        activeSubspaceEnergyRatio: model.diagnostics.activeSubspaceEnergyRatio,
        hypothesisDisagreement: model.diagnostics.hypothesisDisagreement,
        candidateUncertainty: model.diagnostics.candidateUncertainty,
        disagreementQueries: model.diagnostics.disagreementQueries,
        activeSubspaceQueries: model.diagnostics.activeSubspaceQueries,
        orthogonalExplorationQueries: model.diagnostics.orthogonalExplorationQueries,
        globalExplorationQueries: model.diagnostics.globalExplorationQueries,
        strategySwitches: model.diagnostics.strategySwitches,
        fitResidualRms: model.diagnostics.fitResidualRms,
        fitResidualRelative: model.diagnostics.fitResidualRelative,
        fitValidationObservations: model.diagnostics.fitValidationObservations,
        requiredFitValidationObservations: model.diagnostics.requiredFitValidationObservations
      };
      reconstructionComplete = model.reconstructionComplete;
      recoveredPlanes = model.terms.map(({ normal, offset }) => ({ normal, offset }));
      predictor = input => predictBlackBoxReluNdActiveSubspaceAdaptive(model, input);
    } else {
      const model: BlackBoxReluNdModel = decompileBlackBoxReluNdLineProbe(counted.oracle, {
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
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }
  const runtimeMs = performance.now() - start;
  const actualOracleCalls = counted.calls();
  const predictions = predictor ? evaluationPoints.map(predictor) : null;
  const mae = predictions === null
    ? null
    : mean(evaluationPoints.map((point, index) => Math.abs(teacher(point) - predictions[index])));
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
    activeRankMisestimated: activeRankEstimate === null ? null : activeRankEstimate !== scenario.expectedActiveRank,
    reconstructionComplete,
    ...(asadDiagnostics ? { asadDiagnostics } : {}),
    ...(error ? { error } : {})
  };
}

function summarize(records: RunRecord[]): Array<Record<string, unknown>> {
  const keys = new Map<string, RunRecord[]>();
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    const values = keys.get(key) ?? [];
    values.push(record);
    keys.set(key, values);
  }
  return [...keys.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([key, runs]) => {
    const [dimension, scenario, method] = key.split('|');
    const scored = runs.filter(run => run.mae !== null);
    const maes = scored.map(run => run.mae!);
    const ranks = runs.filter(run => run.activeRankMisestimated !== null);
    const fitResiduals = runs.flatMap(run => {
      const value = run.asadDiagnostics?.fitResidualRelative;
      return typeof value === 'number' ? [value] : [];
    });
    return {
      dimension: Number(dimension),
      scenario,
      method,
      runs: runs.length,
      scoredRuns: scored.length,
      completeRate: mean(runs.map(run => run.reconstructionComplete ? 1 : 0)),
      thrownRate: mean(runs.map(run => run.status === 'threw' ? 1 : 0)),
      meanMae: maes.length ? mean(maes) : null,
      p90Mae: percentile(maes, 0.9),
      meanAngularErrorDeg: mean(runs.map(run => run.meanAngularErrorDeg ?? 90)),
      meanCorrectHyperplanes: mean(runs.map(run => run.correctHyperplanes)),
      meanFalseAcceptedHyperplanes: mean(runs.map(run => run.falseAcceptedHyperplanes)),
      meanMissedHyperplanes: mean(runs.map(run => run.missedHyperplanes)),
      meanOracleCalls: mean(runs.map(run => run.actualOracleCalls)),
      maximumOracleCalls: Math.max(...runs.map(run => run.actualOracleCalls)),
      meanRuntimeMs: mean(runs.map(run => run.runtimeMs)),
      activeRankMisestimateRate: ranks.length
        ? mean(ranks.map(run => run.activeRankMisestimated ? 1 : 0))
        : null,
      meanRankUncertainty: mean(runs.map(run => run.asadDiagnostics?.rankUncertainty ?? 0)),
      meanCandidateUncertainty: mean(runs.map(run => run.asadDiagnostics?.candidateUncertainty ?? 0)),
      meanHypothesisDisagreement: mean(runs.map(run => run.asadDiagnostics?.hypothesisDisagreement ?? 0)),
      meanStrategySwitches: mean(runs.map(run => run.asadDiagnostics?.strategySwitches ?? 0)),
      meanFitResidualRelative: fitResiduals.length ? mean(fitResiduals) : null,
      budgetOverruns: runs.filter(run => run.oracleBudgetExceeded).length,
      diagnosticMismatches: runs.filter(run => !run.oracleCountReconciled).length
    };
  });
}

function reportVerdict(summary: Array<Record<string, unknown>>, seeds: number): {
  verdict: 'promising' | 'partially-supported' | 'not-supported';
  keyCellsMeetingCriteria: number;
  nonKeyRegressionCells: number;
  nonKeyScoredCells: number;
  nonKeyUnscoredCells: number;
  budgetOverruns: number;
  relativeMaeTolerance: number;
  absoluteMaeFloor: number;
} {
  const get = (dimension: number, scenario: string, method: string): Record<string, unknown> | undefined =>
    summary.find(row => row.dimension === dimension && row.scenario === scenario && row.method === method);
  const keyScenarios = [
    'isolated-low-rank',
    'multi-crossing-low-rank',
    'near-parallel-low-rank',
    'small-amplitude-low-rank'
  ];
  let keyCellsMeetingCriteria = 0;
  for (const dimension of [16, 32]) {
    for (const scenario of keyScenarios) {
      const asad = get(dimension, scenario, 'asad');
      const baselines = ['ransac', 'star-probe', 'adaptive']
        .map(method => get(dimension, scenario, method))
        .filter((row): row is Record<string, unknown> => row !== undefined && typeof row.meanMae === 'number');
      if (!asad || typeof asad.meanMae !== 'number' || typeof asad.meanOracleCalls !== 'number' || !baselines.length) continue;
      const bestBaselineMae = Math.min(...baselines.map(row => row.meanMae as number));
      const comparableBaselines = baselines.filter(row =>
        maeValuesAreComparable(row.meanMae as number, asad.meanMae as number));
      const queryReduction = oracleCallReduction(
        asad.meanOracleCalls as number,
        comparableBaselines.map(row => row.meanOracleCalls as number)
      );
      const maeReduction = bestBaselineMae > 1e-12
        ? 1 - (asad.meanMae as number) / bestBaselineMae
        : 0;
      if ((queryReduction !== null && queryReduction >= 0.2) || maeReduction >= 0.25) {
        keyCellsMeetingCriteria++;
      }
    }
  }
  let nonKeyRegressionCells = 0;
  let nonKeyScoredCells = 0;
  let nonKeyUnscoredCells = 0;
  const keyScenarioSet = new Set(keyScenarios);
  for (const row of summary) {
    if (row.method !== 'asad') continue;
    const isKeyCell = (row.dimension as number) >= 16 && keyScenarioSet.has(row.scenario as string);
    if (isKeyCell) continue;
    const baselines = ['ransac', 'star-probe', 'adaptive']
      .map(method => get(row.dimension as number, row.scenario as string, method))
      .filter((candidate): candidate is Record<string, unknown> =>
        candidate !== undefined && typeof candidate.meanMae === 'number');
    if (typeof row.meanMae !== 'number' || !baselines.length) {
      nonKeyUnscoredCells++;
    } else {
      nonKeyScoredCells++;
      const baselineMae = Math.min(...baselines.map(candidate => candidate.meanMae as number));
      if (exceedsAllowedMaeRegression(row.meanMae as number, baselineMae)) nonKeyRegressionCells++;
    }
  }
  const budgetOverruns = summary.reduce((sum, row) => sum + (row.budgetOverruns as number), 0);
  const completeSeedCount = seeds >= 100;
  const promising = keyCellsMeetingCriteria > 0 && nonKeyRegressionCells === 0
    && nonKeyUnscoredCells === 0 && budgetOverruns === 0 && completeSeedCount;
  return {
    verdict: promising ? 'promising' : keyCellsMeetingCriteria > 0 ? 'partially-supported' : 'not-supported',
    keyCellsMeetingCriteria,
    nonKeyRegressionCells,
    nonKeyScoredCells,
    nonKeyUnscoredCells,
    budgetOverruns,
    relativeMaeTolerance: ASAD_MAE_COMPARISON_TOLERANCE,
    absoluteMaeFloor: ASAD_MAE_ABSOLUTE_FLOOR
  };
}

function renderReport(
  runId: string,
  seeds: number,
  dimensions: number[],
  scenarios: ScenarioName[],
  summary: Array<Record<string, unknown>>,
  verdict: ReturnType<typeof reportVerdict>,
  rawFilename: string
): string {
  const rows = summary.map(row =>
    `| ${row.dimension} | ${row.scenario} | ${row.method} | ${row.scoredRuns}/${row.runs} | ${typeof row.meanMae === 'number' ? row.meanMae.toPrecision(5) : 'n/a'} | ${Number(row.meanAngularErrorDeg).toFixed(2)} | ${Number(row.meanCorrectHyperplanes).toFixed(2)} | ${Number(row.meanFalseAcceptedHyperplanes).toFixed(2)} | ${Number(row.meanOracleCalls).toFixed(0)} | ${Number(row.meanRuntimeMs).toFixed(2)} | ${row.activeRankMisestimateRate === null ? 'n/a' : `${(Number(row.activeRankMisestimateRate) * 100).toFixed(1)}%`} | ${row.method === 'asad' ? Number(row.meanRankUncertainty).toFixed(3) : 'n/a'} | ${row.method === 'asad' ? Number(row.meanCandidateUncertainty).toFixed(3) : 'n/a'} | ${typeof row.meanFitResidualRelative === 'number' ? Number(row.meanFitResidualRelative).toFixed(4) : 'n/a'} | ${(Number(row.completeRate) * 100).toFixed(1)}% | ${(Number(row.thrownRate) * 100).toFixed(1)}% |`)
    .join('\n');
  return `# Active-Subspace Adaptive Discovery (ASAD): Experimental Report

**Run:** ${runId}  
**Verdict:** ${verdict.verdict}  
**Dimensions:** ${dimensions.join(', ')}  
**Scenarios:** ${scenarios.join(', ')}  
**Seeds per dimension/scenario/method cell:** ${seeds}  
**Raw observations:** [${rawFilename}](../../scratch/results/asad/${rawFilename})

## Research question and method

ASAD keeps local affine hypotheses built only from oracle values and finite-difference gradients. For each next center query, it scores a deterministic random candidate pool by the disagreement of those hypotheses' predicted values, with a novelty term. Gradient covariance estimates a changing active subspace; when rank is not identifiable from the available observations, ASAD switches to global exploration rather than treating a low observed rank as evidence of a low true rank. Candidate hyperplanes come from agreement between gradient jumps and two-sided affine intercepts. Within-side gradient variation lowers candidate confidence; ASAD refuses to label an uncertain or incomplete result as a complete reconstruction.

The solver receives no teacher weights, normals, offsets, scenario label, or test-point data. The benchmark uses deterministic teacher metadata only after solving to score structure and prediction. Evaluation points are generated from a separate seed and never passed to a solver.

Each method receives the same oracle-call cap for its dimension/scenario cell. The benchmark independently increments a wrapper around every solver oracle call; a returned diagnostic count is checked against that measurement. Evaluation-only teacher calls are outside the solver query budget. Runtime is measured around the solver call, not around test scoring.

## Results

| D | Scenario | Method | Scored/runs | MAE | Mean angle (deg) | Correct planes | False planes | Oracle calls | Runtime (ms) | Rank error | Rank uncertainty | Candidate uncertainty | Relative fit residual | Complete | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${rows}

## Acceptance criteria

- High-dimensional key cells are dimensions 16 and 32 for isolated, multi-crossing, near-parallel, and small-amplitude low-rank scenarios. Each has ${seeds} seeds. A cell qualifies if ASAD uses at least 20% fewer calls than an existing mode whose MAE is within 5% relative difference (symmetric denominator; absolute floor ${ASAD_MAE_ABSOLUTE_FLOOR}), or has at least 25% lower MAE at the common call cap.
- Key cells meeting either numerical criterion: ${verdict.keyCellsMeetingCriteria}.
- Non-key required cells where ASAD mean MAE is over 5% worse than the best scored baseline (baseline-relative, with the same absolute floor): ${verdict.nonKeyRegressionCells} of ${verdict.nonKeyScoredCells} scored cells; ${verdict.nonKeyUnscoredCells} cells lack a comparable scored baseline or ASAD output.
- Actual oracle-budget overruns across all records: ${verdict.budgetOverruns}.
- All required seed cells have at least 100 seeds: ${seeds >= 100 ? 'yes' : 'no'}.

## Conclusion

**${verdict.verdict}**. This label is determined by the recorded key-cell criteria, no-worsening check, budget-overrun count, and seed count; it is not inferred from selected successful runs. See the JSON summary and per-run JSONL for missing/incomplete/throwing cases, per-cell counts, and raw measurements.

## Evidence and limits

- Raw per-run records: [${rawFilename}](../../scratch/results/asad/${rawFilename})
- Summary JSON: [summary-${runId}.json](../../scratch/results/asad/summary-${runId}.json)
- Oracle-call reconciliation is recorded per run as actual wrapper invocations versus solver diagnostics; records without diagnostics are explicitly marked unavailable.
- ReLU teacher and contamination generator are deterministic from the benchmark code and recorded seed. The test corpus is independent of solver acquisition.
- The experiment is synthetic and scalar-output only. It does not establish performance on deep networks, noisy/quantized real APIs, or high-dimensional language-model behavior.
- Full-rank cases are expensive and expose finite-sample and support limitations; missing predictions from solver exceptions are not replaced with invented MAE values and remain null in the raw records.
`;
}

function main(): void {
  const { seeds, dimensions, scenarios, output } = parseArgs(process.argv.slice(2));
  const outputDirectory = path.resolve(output);
  fs.mkdirSync(outputDirectory, { recursive: true });
  const runId = new Date().toISOString().replace(/[:.]/g, '-');
  const rawFilename = `asad-${runId}-raw.jsonl`;
  const rawPath = path.join(outputDirectory, rawFilename);
  const records: RunRecord[] = [];
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
            process.stdout.write(`ASAD_PROGRESS ${run}/${maximum} dimension=${dimension} scenario=${scenarioName} seed=${seed} method=${method}\n`);
          }
        }
      }
    }
  }
  const summary = summarize(records);
  const verdict = reportVerdict(summary, seeds);
  const summaryFilename = `summary-${runId}.json`;
  const summaryPath = path.join(outputDirectory, summaryFilename);
  fs.writeFileSync(summaryPath, `${JSON.stringify({
    schemaVersion: 1,
    runId,
    seedsPerCell: seeds,
    dimensions,
    scenarios,
    methods: METHODS,
    evaluationPoints: EVALUATION_POINTS,
    summary,
    verdict,
    totalRuns: records.length,
    totalOracleBudgetOverruns: records.filter(record => record.oracleBudgetExceeded).length,
    totalDiagnosticMismatches: records.filter(record =>
      record.diagnosticOracleCalls !== null && !record.oracleCountReconciled).length,
    totalDiagnosticUnavailable: records.filter(record => record.diagnosticOracleCalls === null).length,
    rawFile: rawFilename
  }, null, 2)}\n`);
  const completeStudy = dimensions.length === DIMENSIONS.length
    && scenarios.length === SCENARIOS.length
    && seeds >= 100;
  if (completeStudy) {
    const reportPath = path.resolve('docs/research/2026-10-29-asad-experimental-report.md');
    fs.writeFileSync(reportPath, renderReport(runId, seeds, dimensions, scenarios, summary, verdict, rawFilename));
  }
  process.stdout.write(`ASAD_BENCHMARK_COMPLETE seeds=${seeds} dimensions=${dimensions.length} scenarios=${scenarios.length} methods=${METHODS.length} runs=${records.length} verdict=${verdict.verdict} raw=${rawFilename} summary=${summaryFilename}\n`);
}

if (require.main === module) main();
