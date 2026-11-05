import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  makeScenario,
  randomTestPoints,
  SCENARIOS as ASAD_SCENARIOS,
  ScenarioName,
  solverBudget
} from './benchmark-asad';
import {
  ASAD_MAE_ABSOLUTE_FLOOR,
  ASAD_MAE_COMPARISON_TOLERANCE,
  maeValuesAreComparable
} from './asad-benchmark-metrics';

type Method =
  | 'asad' | 'ransac' | 'star-probe' | 'adaptive' | 'bcad' | 'mbcd-full'
  | 'gfd-profiled' | 'gfd-single-start' | 'gfd-joint-fit';
type RawRun = {
  schemaVersion: number;
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
  mae: number | null;
  falseAcceptedHyperplanes: number;
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
};
type KeyCell = {
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
type Summary = {
  runId: string;
  seedsPerCell: number;
  dimensions: number[];
  scenarios: ScenarioName[];
  methods: Method[];
  totalRuns: number;
  totalOracleBudgetOverruns: number;
  totalDiagnosticMismatches: number;
  rawFile: string;
  summary: Array<{
    dimension: number;
    scenario: ScenarioName;
    method: Method;
    runs: number;
    scoredRuns: number;
    completeRuns: number;
    meanMae: number | null;
    meanOracleCalls: number;
    falseAcceptedRunRate: number | null;
    meanCudaMemoryAllocatedBytes: number | null;
    meanGpuTrainingMs: number | null;
  }>;
  verdict: {
    verdict: 'promising' | 'partially-supported' | 'not-supported';
    primaryMethod: 'gfd-profiled';
    primaryScenarioWins: ScenarioName[];
    requiredPrimaryScenarioWins: number;
    seedsPerCell: number;
    keyCells: KeyCell[];
    budgetOverruns: number;
    falseAcceptanceTolerance: number;
    maeComparisonTolerance: number;
    absoluteMaeFloor: number;
  };
};

const DIMENSIONS = [16, 32];
const SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank'
];
const METHODS: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full',
  'gfd-profiled', 'gfd-single-start', 'gfd-joint-fit'
];
const BASELINES: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full'
];
const PRIMARY_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank'
];
const RESEARCH_FALSE_ACCEPTANCE_TOLERANCE = 0.01;
const RESEARCH_MAE_TOLERANCE = 0.05;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function readLines(filePath: string): RawRun[] {
  const contents = fs.readFileSync(filePath, 'utf8');
  assert(contents.endsWith('\n'), `Raw campaign lacks final newline: ${filePath}`);
  return contents.trim().split('\n').filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as RawRun;
    } catch (error) {
      throw new Error(`Invalid JSON in ${filePath}:${index + 1}: ${String(error)}`);
    }
  });
}

function findLatestSummary(outputDirectory: string, expectedSeeds: number): { path: string; summary: Summary } {
  const candidates = fs.readdirSync(outputDirectory)
    .filter(file => /^summary-.*\.json$/.test(file))
    .sort()
    .reverse();
  for (const file of candidates) {
    const summaryPath = path.join(outputDirectory, file);
    const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as Summary;
    if (summary.seedsPerCell === expectedSeeds
      && DIMENSIONS.every(value => summary.dimensions.includes(value))
      && SCENARIOS.every(value => summary.scenarios.includes(value))
      && METHODS.every(value => summary.methods.includes(value))) {
      return { path: summaryPath, summary };
    }
  }
  throw new Error(`No GFD campaign summary with ${expectedSeeds} seeds per cell was found`);
}

function groupByCell(records: RawRun[]): Map<string, RawRun[]> {
  const groups = new Map<string, RawRun[]>();
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    groups.set(key, [...(groups.get(key) ?? []), record]);
  }
  return groups;
}

function getGroup(
  groups: Map<string, RawRun[]>,
  dimension: number,
  scenario: ScenarioName,
  method: Method
): RawRun[] {
  return groups.get(`${dimension}|${scenario}|${method}`) ?? [];
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function reconstructCell(
  groups: Map<string, RawRun[]>,
  dimension: number,
  scenario: ScenarioName
): KeyCell {
  const gfd = getGroup(groups, dimension, scenario, 'gfd-profiled');
  const gfdMaeValues = gfd.map(record => record.mae).filter((value): value is number => value !== null);
  const gfdMae = gfdMaeValues.length ? average(gfdMaeValues) : null;
  const gfdFalseRate = gfdMaeValues.length
    ? gfd.filter(record => record.mae !== null && record.falseAcceptedHyperplanes > 0).length / gfdMaeValues.length
    : null;
  const gfdCalls = gfd.length ? average(gfd.map(record => record.actualOracleCalls)) : null;
  const gfdComplete = gfd.length
    ? gfd.filter(record => record.reconstructionComplete).length / gfd.length
    : null;
  const baselines = BASELINES.flatMap(method => {
    const rows = getGroup(groups, dimension, scenario, method);
    const values = rows.map(record => record.mae).filter((value): value is number => value !== null);
    if (!values.length) return [];
    return [{
      method,
      meanMae: average(values),
      meanCalls: average(rows.map(record => record.actualOracleCalls)),
      falseRate: rows.filter(record => record.mae !== null && record.falseAcceptedHyperplanes > 0).length / values.length,
      completionRate: rows.filter(record => record.reconstructionComplete).length / rows.length
    }];
  });
  baselines.sort((first, second) => first.meanMae - second.meanMae);
  const best = baselines[0];
  if (gfdMae === null || gfdCalls === null || gfdComplete === null || !best) {
    return {
      dimension, scenario, bestBaseline: best?.method ?? null, gfdMae,
      baselineMae: best?.meanMae ?? null, gfdOracleCalls: gfdCalls,
      bestComparableBaselineCalls: null, oracleCallReduction: null, maeReduction: null,
      gfdCompletionRate: gfdComplete, baselineCompletionRate: best?.completionRate ?? null,
      falseAcceptedIncrease: null, qualifies: false,
      reason: 'missing_scored_primary_method_or_baseline'
    };
  }
  const comparable = baselines.filter(row => maeValuesAreComparable(row.meanMae, best.meanMae));
  const comparableCalls = comparable.length ? Math.min(...comparable.map(row => row.meanCalls)) : null;
  const callReduction = comparableCalls !== null && comparableCalls > 0
    ? 1 - gfdCalls / comparableCalls : null;
  const maeReduction = best.meanMae > ASAD_MAE_ABSOLUTE_FLOOR
    ? 1 - gfdMae / best.meanMae : null;
  const maePass = maeReduction !== null && maeReduction >= 0.25;
  const efficiencyPass = callReduction !== null && callReduction >= 0.3
    && gfdMae <= best.meanMae * (1 + RESEARCH_MAE_TOLERANCE)
      + ASAD_MAE_ABSOLUTE_FLOOR;
  const baselineFalseRate = comparable.length
    ? Math.min(...comparable.map(row => row.falseRate)) : null;
  const falseIncrease = gfdFalseRate !== null && baselineFalseRate !== null
    ? gfdFalseRate - baselineFalseRate : null;
  const falsePass = falseIncrease !== null
    && falseIncrease <= RESEARCH_FALSE_ACCEPTANCE_TOLERANCE;
  const qualifies = (maePass || efficiencyPass) && falsePass;
  return {
    dimension, scenario, bestBaseline: best.method, gfdMae, baselineMae: best.meanMae,
    gfdOracleCalls: gfdCalls, bestComparableBaselineCalls: comparableCalls,
    oracleCallReduction: callReduction, maeReduction, gfdCompletionRate: gfdComplete,
    baselineCompletionRate: best.completionRate, falseAcceptedIncrease: falseIncrease,
    qualifies,
    reason: qualifies
      ? maePass ? 'mae_improved_by_at_least_25_percent'
        : 'at_least_30_percent_fewer_calls_with_mae_within_5_percent'
      : !(maePass || efficiencyPass)
        ? 'neither_predeclared_performance_threshold_met'
        : 'false_accepted_run_rate_increased_by_more_than_1_percentage_point'
  };
}

function closeEnough(first: number | null, second: number | null): boolean {
  if (first === null || second === null) return first === second;
  return Math.abs(first - second) <= 1e-10 * Math.max(1, Math.abs(first), Math.abs(second));
}

function main(): void {
  const pilotMode = process.argv.includes('--pilot');
  const expectedSeeds = pilotMode ? 10 : 100;
  const projectRoot = path.resolve(__dirname, '..');
  const outputDirectory = path.join(projectRoot, 'scratch/results/gfd', pilotMode ? 'pilot' : '');
  const { summary } = findLatestSummary(outputDirectory, expectedSeeds);
  const rawPath = path.join(outputDirectory, summary.rawFile);
  assert(fs.existsSync(rawPath), `Raw data file is missing: ${summary.rawFile}`);
  const records = readLines(rawPath);
  const expectedTotal = expectedSeeds * DIMENSIONS.length * SCENARIOS.length * METHODS.length;
  assert(records.length === expectedTotal, `Expected ${expectedTotal} runs, found ${records.length}`);
  assert(summary.totalRuns === expectedTotal, 'Summary run count differs from the campaign design');
  assert(summary.seedsPerCell === expectedSeeds, `Campaign does not have ${expectedSeeds} seeds per cell`);
  assert(summary.totalOracleBudgetOverruns === 0, 'Summary reports an oracle-budget overrun');
  assert(summary.totalDiagnosticMismatches === 0, 'Summary reports oracle diagnostic mismatches');

  const seen = new Set<string>();
  const pairedHashes = new Map<string, { model: string; points: string }>();
  let actualBudgetOverruns = 0;
  let actualDiagnosticMismatches = 0;
  for (const record of records) {
    assert(record.schemaVersion === 1, 'Unexpected raw record schema');
    assert(DIMENSIONS.includes(record.dimension), `Unexpected dimension ${record.dimension}`);
    assert(SCENARIOS.includes(record.scenario), `Unexpected scenario ${record.scenario}`);
    assert(METHODS.includes(record.method), `Unexpected method ${record.method}`);
    assert(Number.isSafeInteger(record.seed) && record.seed >= 1 && record.seed <= expectedSeeds,
      `Invalid seed ${record.seed}`);
    const key = `${record.dimension}|${record.scenario}|${record.seed}|${record.method}`;
    assert(!seen.has(key), `Duplicate run: ${key}`);
    seen.add(key);
    const scenario = makeScenario(record.scenario, record.dimension, record.seed);
    const expectedModelHash = hash(scenario);
    const expectedPointsHash = hash(randomTestPoints(record.seed, record.dimension));
    assert(record.scenarioModelHash === expectedModelHash, `Teacher fixture mismatch: ${key}`);
    assert(record.evaluationPointsHash === expectedPointsHash, `Evaluation set mismatch: ${key}`);
    const expectedSolverSeed = record.seed * 104729 + record.dimension * 13007
      + ASAD_SCENARIOS.indexOf(record.scenario) * 8191;
    assert(record.solverSeed === expectedSolverSeed, `Solver seed mismatch: ${key}`);
    const expectedBudget = solverBudget(record.dimension, scenario.terms.length).budget;
    assert(record.oracleBudget === expectedBudget, `Oracle budget mismatch: ${key}`);
    actualBudgetOverruns += Number(record.actualOracleCalls > record.oracleBudget);
    assert(record.oracleBudgetExceeded === (record.actualOracleCalls > record.oracleBudget),
      `Budget diagnostic mismatch: ${key}`);
    if (record.diagnosticOracleCalls !== null) {
      assert(record.diagnosticOracleCalls === record.actualOracleCalls, `Oracle call count mismatch: ${key}`);
      assert(record.oracleCountReconciled, `Oracle reconciliation flag is false: ${key}`);
    } else {
      assert(record.status === 'threw' && record.oracleCountReconciled,
        `Missing oracle diagnostic on a non-thrown run: ${key}`);
    }
    if (record.method.startsWith('gfd-')) {
      assert(record.gfdDiagnostics !== undefined, `GFD diagnostics missing: ${key}`);
      assert(record.actualOracleCalls === record.gfdDiagnostics.sampleCount,
        `GFD actual sample count mismatch: ${key}`);
      assert(record.gfdDiagnostics.sampleCount <= record.oracleBudget, `GFD sample budget exceeded: ${key}`);
      assert(record.gfdDiagnostics.computeDevice === 'cuda:0',
        `GFD hidden fitting did not report CUDA tensors: ${key}`);
      assert(record.gfdDiagnostics.deviceName.length > 0, `GFD CUDA device name missing: ${key}`);
      assert(record.gfdDiagnostics.cudaMemoryAllocatedBytes > 0,
        `GFD CUDA memory allocation is empty: ${key}`);
      assert(record.gfdDiagnostics.trainingTimeMs > 0,
        `GFD CUDA training time is missing: ${key}`);
      assert(record.gfdDiagnostics.dtype === 'torch.float64',
        `Unexpected GFD CUDA numerical dtype: ${key}`);
    } else {
      assert(record.gfdDiagnostics === undefined, `GFD diagnostics leaked into another method: ${key}`);
    }
    const pairKey = `${record.dimension}|${record.scenario}|${record.seed}`;
    const previous = pairedHashes.get(pairKey);
    if (previous) {
      assert(previous.model === record.scenarioModelHash
        && previous.points === record.evaluationPointsHash, `Paired fixture mismatch: ${pairKey}`);
    } else {
      pairedHashes.set(pairKey, {
        model: record.scenarioModelHash,
        points: record.evaluationPointsHash
      });
    }
  }
  assert(actualBudgetOverruns === 0, `Found ${actualBudgetOverruns} actual budget overruns`);
  assert(actualDiagnosticMismatches === 0, 'Actual oracle counts do not reconcile');
  for (const dimension of DIMENSIONS) {
    for (const scenario of SCENARIOS) {
      for (let seed = 1; seed <= expectedSeeds; seed++) {
        for (const method of METHODS) {
          assert(seen.has(`${dimension}|${scenario}|${seed}|${method}`),
            `Required run is missing: ${dimension}/${scenario}/${seed}/${method}`);
        }
      }
    }
  }
  for (const row of summary.summary) {
    const rows = records.filter(record =>
      record.dimension === row.dimension && record.scenario === row.scenario && record.method === row.method);
    const scored = rows.filter(record => record.mae !== null);
    const completeRuns = rows.filter(record => record.status === 'complete').length;
    const meanMae = scored.length
      ? average(scored.map(record => record.mae!)) : null;
    const calls = average(rows.map(record => record.actualOracleCalls));
    const falseRate = scored.length
      ? scored.filter(record => record.falseAcceptedHyperplanes > 0).length / scored.length : null;
    const gpuRows = rows.flatMap(record => record.gfdDiagnostics ? [record.gfdDiagnostics] : []);
    const meanCudaMemory = gpuRows.length
      ? average(gpuRows.map(diagnostic => diagnostic.cudaMemoryAllocatedBytes)) : null;
    const meanGpuTraining = gpuRows.length
      ? average(gpuRows.map(diagnostic => diagnostic.trainingTimeMs)) : null;
    assert(row.runs === rows.length && row.scoredRuns === scored.length,
      `Summary run/scored count mismatch: ${row.dimension}/${row.scenario}/${row.method}`);
    assert(row.completeRuns === completeRuns, `Completion count mismatch: ${row.method}`);
    assert(closeEnough(row.meanMae, meanMae), `Summary MAE mismatch: ${row.method}`);
    assert(closeEnough(row.meanOracleCalls, calls), `Summary call count mismatch: ${row.method}`);
    assert(closeEnough(row.falseAcceptedRunRate, falseRate), `False-accept rate mismatch: ${row.method}`);
    assert(closeEnough(row.meanCudaMemoryAllocatedBytes, meanCudaMemory),
      `Summary CUDA memory mismatch: ${row.method}`);
    assert(closeEnough(row.meanGpuTrainingMs, meanGpuTraining),
      `Summary GPU fit-time mismatch: ${row.method}`);
  }

  const groups = groupByCell(records);
  const keyCells = DIMENSIONS.flatMap(dimension =>
    PRIMARY_SCENARIOS.map(scenario => reconstructCell(groups, dimension, scenario)));
  assert(summary.verdict.keyCells.length === keyCells.length, 'Primary cell count differs');
  keyCells.forEach((cell, index) => {
    const stored = summary.verdict.keyCells[index];
    assert(stored.dimension === cell.dimension && stored.scenario === cell.scenario,
      `Primary cell order mismatch at ${index}`);
    assert(stored.qualifies === cell.qualifies && stored.reason === cell.reason,
      `Primary acceptance recomputation mismatch: ${cell.dimension}/${cell.scenario}`);
    assert(closeEnough(stored.gfdMae, cell.gfdMae)
      && closeEnough(stored.baselineMae, cell.baselineMae)
      && closeEnough(stored.gfdOracleCalls, cell.gfdOracleCalls)
      && closeEnough(stored.oracleCallReduction, cell.oracleCallReduction)
      && closeEnough(stored.maeReduction, cell.maeReduction)
      && closeEnough(stored.falseAcceptedIncrease, cell.falseAcceptedIncrease),
    `Primary metrics differ: ${cell.dimension}/${cell.scenario}`);
  });
  const wins = PRIMARY_SCENARIOS.filter(scenario =>
    DIMENSIONS.every(dimension => keyCells.some(cell =>
      cell.dimension === dimension && cell.scenario === scenario && cell.qualifies)));
  const verdict = actualBudgetOverruns > 0 ? 'not-supported'
    : wins.length >= 2 ? 'promising' : wins.length === 1 ? 'partially-supported' : 'not-supported';
  assert(summary.verdict.primaryScenarioWins.join('|') === wins.join('|'),
    'Primary scenario wins differ from recomputation');
  assert(summary.verdict.verdict === verdict, 'Verdict differs from recomputation');
  assert(summary.verdict.requiredPrimaryScenarioWins === 2, 'The predeclared win requirement changed');
  assert(summary.verdict.falseAcceptanceTolerance === RESEARCH_FALSE_ACCEPTANCE_TOLERANCE,
    'False-acceptance threshold changed');
  assert(summary.verdict.maeComparisonTolerance === RESEARCH_MAE_TOLERANCE,
    'MAE tolerance changed');
  assert(summary.verdict.absoluteMaeFloor === ASAD_MAE_ABSOLUTE_FLOOR,
    'MAE absolute floor changed');

  const reportPath = path.join(projectRoot, 'docs/research', pilotMode
    ? '2026-10-29-gfd-vp-pilot-report.md'
    : '2026-10-29-gfd-vp-experimental-report.md');
  assert(fs.existsSync(reportPath), 'GFD-VP report is missing');
  const report = fs.readFileSync(reportPath, 'utf8');
  assert(report.includes(summary.runId), 'GFD-VP report does not match the verified campaign run');
  assert(report.includes(summary.rawFile), 'GFD-VP report does not link the verified raw file');
  const marker = pilotMode ? 'GFD_PILOT_VERIFICATION_PASSED' : 'GFD_RESULT_VERIFICATION_PASSED';
  console.log(`${marker} runs=${records.length} seeds=${expectedSeeds} cells=${summary.summary.length} overrun=${actualBudgetOverruns} diagnosticMismatches=${actualDiagnosticMismatches} verdict=${verdict}`);
}

if (require.main === module) main();
