import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  makeScenario,
  randomTestPoints,
  SCENARIOS,
  ScenarioName,
  solverBudget
} from './benchmark-asad';

type Method =
  | 'asad' | 'ransac' | 'star-probe' | 'adaptive' | 'bcad' | 'mbcd-full'
  | 'gfd-profiled' | 'gsmi-full' | 'gsmi-second-only';
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
type Summary = {
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
  'near-parallel-low-rank', 'multi-crossing-low-rank', 'small-amplitude-low-rank',
  'contaminated-local-probes', 'orthogonal-full-rank'
];
const METHODS: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full',
  'gfd-profiled', 'gsmi-full', 'gsmi-second-only'
];
const PRIMARY_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank', 'multi-crossing-low-rank', 'small-amplitude-low-rank'
];
const BASELINES: Method[] = [
  'asad', 'ransac', 'star-probe', 'adaptive', 'bcad', 'mbcd-full', 'gfd-profiled'
];
const EXPECTED_SEEDS = 10;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function closeEnough(first: number | null, second: number | null): boolean {
  if (first === null || second === null) return first === second;
  return Math.abs(first - second) <= 1e-10 * Math.max(1, Math.abs(first), Math.abs(second));
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function readRaw(filePath: string): RawRun[] {
  const contents = fs.readFileSync(filePath, 'utf8');
  assert(contents.endsWith('\n'), 'Raw GSI run file lacks a final newline');
  return contents.trim().split('\n').filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as RawRun;
    } catch (error) {
      throw new Error(`Invalid raw JSON at line ${index + 1}: ${String(error)}`);
    }
  });
}

function recomputeSummary(records: RawRun[]): SummaryRow[] {
  const groups = new Map<string, RawRun[]>();
  for (const row of records) {
    const key = `${row.dimension}|${row.scenario}|${row.method}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.entries()].map(([key, rows]) => {
    const [dimension, scenario, method] = key.split('|');
    const scored = rows.filter(row => row.mae !== null);
    const completeRuns = rows.filter(row => row.reconstructionComplete).length;
    const gpu = rows.flatMap(row => row.gpuDiagnostics ? [row.gpuDiagnostics] : []);
    const gsmi = rows.flatMap(row => row.gsmiDiagnostics ? [row.gsmiDiagnostics] : []);
    return {
      dimension: Number(dimension),
      scenario: scenario as ScenarioName,
      method: method as Method,
      runs: rows.length,
      scoredRuns: scored.length,
      completeRuns,
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
      meanCudaMemoryAllocatedBytes: gpu.length
        ? average(gpu.map(row => row.cudaMemoryAllocatedBytes)) : null,
      meanGpuTrainingMs: gpu.length
        ? average(gpu.map(row => row.gpuTimeMs)) : null,
      meanEstimatedRank: gsmi.length ? average(gsmi.map(row => row.estimatedRank)) : null,
      meanPencilResidual: gsmi.length
        ? average(gsmi.flatMap(row => row.matrixPencilResidual === null ? [] : [row.matrixPencilResidual]))
        : null
    };
  }).sort((a, b) => a.dimension - b.dimension
    || a.scenario.localeCompare(b.scenario) || a.method.localeCompare(b.method));
}

function recomputeCells(rows: SummaryRow[]): CellResult[] {
  return DIMENSIONS.flatMap(dimension => PRIMARY_SCENARIOS.map(scenario => {
    const gsi = rows.find(row => row.dimension === dimension
      && row.scenario === scenario && row.method === 'gsmi-full');
    const baselines = rows.filter(row => row.dimension === dimension
      && row.scenario === scenario && BASELINES.includes(row.method) && row.meanMae !== null)
      .sort((a, b) => a.meanMae! - b.meanMae!);
    return {
      dimension, scenario, gsmiMae: gsi?.meanMae ?? null,
      bestBaseline: baselines[0]?.method ?? null,
      bestBaselineMae: baselines[0]?.meanMae ?? null,
      gsmiCompleteRate: gsi?.completionRate ?? null,
      gsmiOracleCalls: gsi?.meanOracleCalls ?? null,
      falseAcceptedRunRate: gsi?.falseAcceptedRunRate ?? null
    };
  }));
}

function verifySummaryRows(actual: SummaryRow[], expected: SummaryRow[]): void {
  assert(actual.length === expected.length, 'Summary row count differs from raw data');
  for (let index = 0; index < expected.length; index++) {
    const first = actual[index];
    const second = expected[index];
    for (const key of [
      'dimension', 'scenario', 'method', 'runs', 'scoredRuns', 'completeRuns',
      'incompleteRuns', 'thrownRuns'
    ] as const) {
      assert(first[key] === second[key], `Summary ${key} mismatch in ${second.method}/${second.scenario}`);
    }
    for (const key of [
      'completionRate', 'meanMae', 'meanOracleCalls', 'meanNormalErrorDeg', 'meanOffsetError',
      'meanCorrectHyperplanes', 'meanFalseAcceptedHyperplanes', 'falseAcceptedRunRate',
      'meanRuntimeMs', 'meanCudaMemoryAllocatedBytes', 'meanGpuTrainingMs',
      'meanEstimatedRank', 'meanPencilResidual'
    ] as const) {
      assert(closeEnough(first[key], second[key]), `Summary ${key} mismatch in ${second.method}/${second.scenario}`);
    }
  }
}

function verify(): void {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'scratch/results/gsmi/pilot');
  const summaries = fs.readdirSync(output).filter(file => /^summary-.*\.json$/.test(file)).sort();
  assert(summaries.length > 0, 'No GSMI pilot summary exists');
  const summaryPath = path.join(output, summaries[summaries.length - 1]);
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as Summary;
  assert(summary.seedsPerCell === EXPECTED_SEEDS, 'Pilot does not contain ten seeds per cell');
  assert(DIMENSIONS.every(item => summary.dimensions.includes(item)), 'Pilot dimensions are incomplete');
  assert(REQUIRED_SCENARIOS.every(item => summary.scenarios.includes(item)), 'Pilot scenarios are incomplete');
  assert(METHODS.every(item => summary.methods.includes(item)), 'Pilot methods or ablations are missing');
  const records = readRaw(path.join(output, summary.rawFile));
  const expectedTotal = EXPECTED_SEEDS * DIMENSIONS.length * REQUIRED_SCENARIOS.length * METHODS.length;
  assert(records.length === expectedTotal && summary.totalRuns === expectedTotal,
    `Expected ${expectedTotal} paired runs, found ${records.length}`);

  const seen = new Set<string>();
  const fixtures = new Map<string, { model: string; points: string }>();
  let overruns = 0;
  let diagnosticMismatches = 0;
  for (const row of records) {
    assert(row.schemaVersion === 1, 'Unexpected GSI raw schema version');
    assert(DIMENSIONS.includes(row.dimension) && REQUIRED_SCENARIOS.includes(row.scenario)
      && METHODS.includes(row.method), 'Unexpected run design cell');
    assert(Number.isSafeInteger(row.seed) && row.seed >= 1 && row.seed <= EXPECTED_SEEDS,
      `Invalid seed: ${row.seed}`);
    const key = `${row.dimension}|${row.scenario}|${row.seed}|${row.method}`;
    assert(!seen.has(key), `Duplicate run: ${key}`);
    seen.add(key);
    const scenario = makeScenario(row.scenario, row.dimension, row.seed);
    const expectedModelHash = hash(scenario);
    const expectedPointsHash = hash(randomTestPoints(row.seed, row.dimension));
    assert(row.scenarioModelHash === expectedModelHash, `Teacher hash mismatch: ${key}`);
    assert(row.evaluationPointsHash === expectedPointsHash, `Test-point hash mismatch: ${key}`);
    const expectedSeed = row.seed * 104729 + row.dimension * 13007
      + SCENARIOS.indexOf(row.scenario) * 8191;
    assert(row.solverSeed === expectedSeed, `Solver seed mismatch: ${key}`);
    const expectedBudget = solverBudget(row.dimension, scenario.terms.length).budget;
    assert(row.oracleBudget === expectedBudget, `Oracle budget mismatch: ${key}`);
    assert(Number.isSafeInteger(row.actualOracleCalls) && row.actualOracleCalls >= 0,
      `Invalid oracle count: ${key}`);
    overruns += Number(row.actualOracleCalls > row.oracleBudget);
    assert(row.oracleBudgetExceeded === (row.actualOracleCalls > row.oracleBudget),
      `Budget diagnostic mismatch: ${key}`);
    if (row.diagnosticOracleCalls === null) {
      assert(row.status === 'threw' && row.oracleCountReconciled,
        `Missing diagnostics on a non-thrown run: ${key}`);
    } else {
      assert(row.diagnosticOracleCalls === row.actualOracleCalls && row.oracleCountReconciled,
        `Oracle counters do not reconcile: ${key}`);
    }
    if (row.method.startsWith('gsmi-')) {
      assert(row.gsmiDiagnostics !== undefined, `GSMI diagnostics missing: ${key}`);
      const diagnostic = row.gsmiDiagnostics!;
      assert(row.actualOracleCalls === diagnostic.sampleCount,
        `Actual calls differ from Gaussian sample count: ${key}`);
      assert(diagnostic.sampleCount <= row.oracleBudget, `GSMI exceeded budget: ${key}`);
      assert(diagnostic.sampleSigma === 0.2, `GSMI sampling scale changed: ${key}`);
      assert(diagnostic.computeDevice === 'cuda:0' && diagnostic.deviceName.length > 0,
        `GSMI did not report the configured CUDA device: ${key}`);
      assert(diagnostic.cudaMemoryAllocatedBytes > 0 && diagnostic.momentAndInversionTimeMs > 0,
        `GSMI GPU resource evidence is empty: ${key}`);
      assert(diagnostic.momentOrder === (row.method === 'gsmi-full'
        ? 'second-and-third' : 'second-only'), `Wrong moment ablation: ${key}`);
      assert(row.gpuDiagnostics !== undefined
        && row.gpuDiagnostics.computeDevice === diagnostic.computeDevice
        && row.gpuDiagnostics.deviceName === diagnostic.deviceName
        && row.gpuDiagnostics.cudaMemoryAllocatedBytes === diagnostic.cudaMemoryAllocatedBytes
        && row.gpuDiagnostics.gpuTimeMs === diagnostic.momentAndInversionTimeMs,
      `GSMI CUDA resource counters differ: ${key}`);
    } else {
      assert(row.gsmiDiagnostics === undefined, `GSMI diagnostics leaked to another method: ${key}`);
      if (row.method === 'gfd-profiled') {
        assert(row.gpuDiagnostics?.computeDevice === 'cuda:0'
          && row.gpuDiagnostics.cudaMemoryAllocatedBytes > 0
          && row.gpuDiagnostics.gpuTimeMs > 0, `GFD GPU comparator diagnostics missing: ${key}`);
      } else {
        assert(row.gpuDiagnostics === undefined, `Unexpected GPU diagnostics on CPU method: ${key}`);
      }
    }
    const fixtureKey = `${row.dimension}|${row.scenario}|${row.seed}`;
    const previous = fixtures.get(fixtureKey);
    if (previous) {
      assert(previous.model === row.scenarioModelHash
        && previous.points === row.evaluationPointsHash, `Unpaired fixture: ${fixtureKey}`);
    } else {
      fixtures.set(fixtureKey, { model: row.scenarioModelHash, points: row.evaluationPointsHash });
    }
  }
  diagnosticMismatches = records.filter(row => !row.oracleCountReconciled).length;
  assert(overruns === 0, `Found ${overruns} actual oracle budget overruns`);
  assert(diagnosticMismatches === 0, 'Oracle diagnostic reconciliation failed');
  for (const dimension of DIMENSIONS) {
    for (const scenario of REQUIRED_SCENARIOS) {
      for (let seed = 1; seed <= EXPECTED_SEEDS; seed++) {
        for (const method of METHODS) {
          assert(seen.has(`${dimension}|${scenario}|${seed}|${method}`),
            `Required paired run missing: ${dimension}/${scenario}/${seed}/${method}`);
        }
      }
    }
  }

  const expectedRows = recomputeSummary(records);
  verifySummaryRows(summary.summary, expectedRows);
  const expectedStatuses = {
    complete: records.filter(row => row.status === 'complete').length,
    incomplete: records.filter(row => row.status === 'incomplete').length,
    threw: records.filter(row => row.status === 'threw').length
  };
  assert(summary.statusTotals.complete === expectedStatuses.complete
    && summary.statusTotals.incomplete === expectedStatuses.incomplete
    && summary.statusTotals.threw === expectedStatuses.threw,
  'Run status totals differ from raw records');
  const expectedCells = recomputeCells(expectedRows);
  assert(summary.gsmiCells.length === expectedCells.length, 'Primary cell count differs');
  for (let index = 0; index < expectedCells.length; index++) {
    const first = summary.gsmiCells[index];
    const second = expectedCells[index];
    assert(first.dimension === second.dimension && first.scenario === second.scenario,
      `Primary cell order mismatch at ${index}`);
    for (const key of [
      'gsmiMae', 'bestBaselineMae', 'gsmiCompleteRate', 'gsmiOracleCalls', 'falseAcceptedRunRate'
    ] as const) {
      assert(closeEnough(first[key], second[key]), `Primary cell ${key} mismatch: ${second.scenario}`);
    }
    assert(first.bestBaseline === second.bestBaseline, `Best baseline mismatch: ${second.scenario}`);
  }
  const qualifies = PRIMARY_SCENARIOS.some(scenario =>
    DIMENSIONS.every(dimension => expectedCells.some(cell => cell.dimension === dimension
      && cell.scenario === scenario && (cell.gsmiCompleteRate ?? 0) >= 0.5)));
  const controlRows = expectedRows.filter(row => row.method === 'gsmi-full'
    && row.scenario === 'orthogonal-full-rank');
  assert(controlRows.length === 2, 'Full-rank controls are missing');
  const controlPass = controlRows.every(row => (row.falseAcceptedRunRate ?? 1) <= 0.1
    && row.meanFalseAcceptedHyperplanes <= 1);
  const verdict = qualifies && controlPass && overruns === 0 ? 'promising' : 'not-supported';
  assert(summary.verdict === verdict, 'Pilot verdict differs from independent recomputation');
  const reportPath = path.join(root, 'docs/research/2026-10-30-global-spectral-inversion-pilot.md');
  assert(fs.existsSync(reportPath), 'GSMI pilot report is missing');
  const report = fs.readFileSync(reportPath, 'utf8');
  assert(report.includes(summary.runId) && report.includes(summary.rawFile),
    'GSMI pilot report does not match latest raw data');
  console.log(`GSMI_PILOT_VERIFICATION_PASSED runs=${records.length} seeds=${EXPECTED_SEEDS} cells=${expectedRows.length} budgetOverruns=${overruns} diagnosticMismatches=${diagnosticMismatches} verdict=${verdict}`);
}

if (require.main === module) verify();
