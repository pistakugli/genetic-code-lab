import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  makeScenario,
  randomTestPoints,
  ScenarioName,
  SCENARIOS as ASAD_SCENARIOS,
  solverBudget
} from './benchmark-asad';
import {
  ASAD_MAE_ABSOLUTE_FLOOR,
  maeValuesAreComparable,
  oracleCallReduction
} from './asad-benchmark-metrics';

type RawRun = {
  seed: number;
  dimension: number;
  scenario: ScenarioName;
  method: string;
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
  activeRankMisestimated: boolean | null;
};

type SummaryRow = {
  dimension: number;
  scenario: string;
  method: string;
  runs: number;
  scoredRuns: number;
  meanMae: number | null;
  meanOracleCalls: number;
  falseAcceptedRunRate: number;
  activeRankMisestimateRate: number | null;
};

const DIMENSIONS = [16, 32];
const FOCUS_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank'
];
const METHODS = [
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
const BASELINES = ['asad', 'ransac', 'star-probe', 'adaptive'];
const MAIN_SCENARIOS: ScenarioName[] = [
  'near-parallel-low-rank',
  'multi-crossing-low-rank',
  'small-amplitude-low-rank'
];
const PRIMARY_METHOD = 'bcad-boundary-delayed';
const EXPECTED_EVALUATION_POINTS = 192;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function readJsonLines<T>(filePath: string): T[] {
  return fs.readFileSync(filePath, 'utf8').trim().split('\n').filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line) as T;
      } catch (error) {
        throw new Error(`Invalid JSON in ${filePath}:${index + 1}: ${String(error)}`);
      }
    });
}

function main(): void {
  const projectRoot = path.resolve(__dirname, '..');
  const reportPath = path.join(projectRoot, 'docs/research/2026-10-29-bcad-experimental-report.md');
  const report = fs.readFileSync(reportPath, 'utf8');
  const runId = report.match(/^\*\*Run:\*\* (.+)$/m)?.[1].trim();
  assert(runId, 'BCAD report does not name its run id');
  const outputDirectory = path.join(projectRoot, 'scratch/results/bcad');
  const summaryPath = path.join(outputDirectory, `summary-${runId}.json`);
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as {
    runId: string;
    seedsPerCell: number;
    dimensions: number[];
    scenarios: ScenarioName[];
    methods: string[];
    evaluationPoints: number;
    totalRuns: number;
    totalOracleBudgetOverruns: number;
    totalDiagnosticMismatches: number;
    rawFile: string;
    summary: SummaryRow[];
    verdict: {
      verdict: string;
      mainScenarioWins: ScenarioName[];
      keyCells: Array<{
        dimension: number;
        scenario: ScenarioName;
        qualifies: boolean;
        queryReduction: number | null;
        maeReduction: number | null;
        falseAcceptedIncrease: number | null;
        performancePass: boolean;
        falseAcceptancePass: boolean;
      }>;
      budgetOverruns: number;
    };
  };
  const records = readJsonLines<RawRun>(path.join(outputDirectory, summary.rawFile));
  assert(summary.runId === runId, 'BCAD summary and report run ids differ');
  assert(summary.seedsPerCell === 100, 'BCAD acceptance campaign must use at least 100 seeds per cell');
  assert(summary.dimensions.length === DIMENSIONS.length
    && DIMENSIONS.every(dimension => summary.dimensions.includes(dimension)), 'BCAD campaign omits dimension 16 or 32');
  assert(summary.scenarios.length === FOCUS_SCENARIOS.length
    && FOCUS_SCENARIOS.every(scenario => summary.scenarios.includes(scenario)), 'BCAD campaign omits a required focus scenario');
  assert(summary.methods.length === METHODS.length
    && METHODS.every(method => summary.methods.includes(method)), 'BCAD campaign omits a baseline or ablation arm');
  assert(summary.evaluationPoints === EXPECTED_EVALUATION_POINTS, 'BCAD evaluation point count changed');
  const expectedTotal = DIMENSIONS.length * FOCUS_SCENARIOS.length * summary.seedsPerCell * METHODS.length;
  assert(records.length === expectedTotal && summary.totalRuns === expectedTotal,
    `Expected ${expectedTotal} raw runs, found ${records.length}`);

  const runKeys = new Set<string>();
  const groups = new Map<string, RawRun[]>();
  let actualBudgetOverruns = 0;
  let actualDiagnosticMismatches = 0;
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.seed}|${record.method}`;
    assert(!runKeys.has(key), `Duplicate run record ${key}`);
    runKeys.add(key);
    assert(DIMENSIONS.includes(record.dimension), `Unexpected dimension in ${key}`);
    assert(FOCUS_SCENARIOS.includes(record.scenario), `Unexpected scenario in ${key}`);
    assert(METHODS.includes(record.method), `Unexpected method in ${key}`);
    assert(record.seed >= 1 && record.seed <= summary.seedsPerCell, `Unexpected seed in ${key}`);
    assert(record.actualOracleCalls <= record.oracleBudget, `Actual oracle budget overrun in ${key}`);
    const teacherHash = hash(makeScenario(record.scenario, record.dimension, record.seed));
    const testPointsHash = hash(randomTestPoints(record.seed, record.dimension));
    assert(record.scenarioModelHash === teacherHash, `Teacher fixture hash mismatch in ${key}`);
    assert(record.evaluationPointsHash === testPointsHash, `Evaluation set hash mismatch in ${key}`);
    const expectedSolverSeed = record.seed * 104729
      + record.dimension * 13007
      + ASAD_SCENARIOS.indexOf(record.scenario) * 8191;
    assert(record.solverSeed === expectedSolverSeed, `Solver seed mismatch in ${key}`);
    const expectedBudget = solverBudget(record.dimension, makeScenario(
      record.scenario,
      record.dimension,
      record.seed
    ).terms.length).budget;
    assert(record.oracleBudget === expectedBudget, `Oracle budget mismatch in ${key}`);
    if (record.diagnosticOracleCalls === null) {
      assert(record.status === 'threw', `Missing diagnostic count on a non-throwing run ${key}`);
    } else {
      assert(record.diagnosticOracleCalls === record.actualOracleCalls, `Diagnostic call count mismatch in ${key}`);
      assert(record.oracleCountReconciled, `Oracle reconciliation flag is false in ${key}`);
    }
    if (record.oracleBudgetExceeded) actualBudgetOverruns++;
    if (record.diagnosticOracleCalls !== null && !record.oracleCountReconciled) actualDiagnosticMismatches++;
    if (record.status === 'threw') assert(record.mae === null, `Throwing run has a synthetic MAE in ${key}`);
    else assert(typeof record.mae === 'number' && Number.isFinite(record.mae), `Scored run has no MAE in ${key}`);
    const cellKey = `${record.dimension}|${record.scenario}|${record.method}`;
    const cell = groups.get(cellKey) ?? [];
    cell.push(record);
    groups.set(cellKey, cell);
  }
  assert(groups.size === DIMENSIONS.length * FOCUS_SCENARIOS.length * METHODS.length, 'Incorrect number of summary cells');
  assert(actualBudgetOverruns === 0 && summary.totalOracleBudgetOverruns === actualBudgetOverruns,
    'At least one actual oracle budget was exceeded');
  assert(actualDiagnosticMismatches === summary.totalDiagnosticMismatches,
    'Diagnostic mismatch count differs from raw records');
  assert(summary.totalDiagnosticMismatches === 0, 'At least one available diagnostic count mismatched actual calls');

  const cellGroupKeys = new Set<string>();
  for (const dimension of DIMENSIONS) {
    for (const scenario of FOCUS_SCENARIOS) {
      const scenarioData = makeScenario(scenario, dimension, 1);
      for (let seed = 1; seed <= summary.seedsPerCell; seed++) {
        const key = `${dimension}|${scenario}|${seed}`;
        const rowSet = METHODS.map(method => {
          const matches = records.filter(record =>
            record.dimension === dimension && record.scenario === scenario
            && record.seed === seed && record.method === method);
          assert(matches.length === 1, `Missing/duplicate paired method run ${key}|${method}`);
          return matches[0];
        });
        assert(new Set(rowSet.map(record => record.scenarioModelHash)).size === 1,
          `Methods did not share the same teacher ${key}`);
        assert(new Set(rowSet.map(record => record.evaluationPointsHash)).size === 1,
          `Methods did not share the same evaluation points ${key}`);
        assert(new Set(rowSet.map(record => record.solverSeed)).size === 1,
          `Methods did not share the same solver seed ${key}`);
        assert(new Set(rowSet.map(record => record.oracleBudget)).size === 1,
          `Methods did not share the same oracle budget ${key}`);
      }
      assert(scenarioData.name === scenario, `Scenario fixture generation mismatch for ${dimension}|${scenario}`);
      for (const method of METHODS) cellGroupKeys.add(`${dimension}|${scenario}|${method}`);
    }
  }
  assert(cellGroupKeys.size === groups.size, 'Paired run grid differs from run summary grid');

  const recalculatedSummary = summary.summary.map(row => {
    const raw = groups.get(`${row.dimension}|${row.scenario}|${row.method}`);
    assert(raw && raw.length === summary.seedsPerCell, 'Summary cell lacks 100 raw seeds');
    const scored = raw.filter((record): record is RawRun & { mae: number } =>
      typeof record.mae === 'number');
    const rankRuns = raw.filter((record): record is RawRun & { activeRankMisestimated: boolean } =>
      typeof record.activeRankMisestimated === 'boolean');
    const mae = scored.length ? average(scored.map(record => record.mae)) : null;
    const calls = average(raw.map(record => record.actualOracleCalls));
    const falseRate = average(raw.map(record => record.falseAcceptedHyperplanes > 0 ? 1 : 0));
    assert(row.runs === raw.length && row.scoredRuns === scored.length, 'Summary run counts differ from raw');
    assert(row.meanMae === null ? mae === null : mae !== null
      && Math.abs(row.meanMae - mae) <= 1e-10 * Math.max(1, Math.abs(row.meanMae), Math.abs(mae)),
    `Summary MAE differs from raw for ${row.dimension}|${row.scenario}|${row.method}`);
    assert(Math.abs(row.meanOracleCalls - calls) <= 1e-10 * Math.max(1, Math.abs(calls)),
      `Summary oracle calls differ from raw for ${row.dimension}|${row.scenario}|${row.method}`);
    assert(Math.abs(row.falseAcceptedRunRate - falseRate) <= 1e-10,
      `Summary false-accept rate differs from raw for ${row.dimension}|${row.scenario}|${row.method}`);
    const rankRate = rankRuns.length
      ? average(rankRuns.map(record => record.activeRankMisestimated ? 1 : 0))
      : null;
    assert(row.activeRankMisestimateRate === null ? rankRate === null : rankRate !== null
      && Math.abs(row.activeRankMisestimateRate - rankRate) <= 1e-10,
    `Summary rank error rate differs from raw for ${row.dimension}|${row.scenario}|${row.method}`);
    return {
      dimension: row.dimension,
      scenario: row.scenario,
      method: row.method,
      meanMae: mae,
      meanOracleCalls: calls,
      falseAcceptedRunRate: falseRate
    };
  });

  const primaryKeyCells = DIMENSIONS.flatMap(dimension => MAIN_SCENARIOS.map(scenario => {
    const get = (method: string) => recalculatedSummary.find(row =>
      row.dimension === dimension && row.scenario === scenario && row.method === method);
    const bcad = get(PRIMARY_METHOD);
    const baselines = BASELINES.map(get)
      .filter((row): row is NonNullable<typeof row> => row !== undefined && typeof row.meanMae === 'number');
    if (!bcad || typeof bcad.meanMae !== 'number' || !baselines.length) {
      return {
        dimension,
        scenario,
        qualifies: false,
        queryReduction: null,
        maeReduction: null,
        falseAcceptedIncrease: null,
        performancePass: false,
        falseAcceptancePass: false
      };
    }
    const bestBaseline = baselines.reduce((best, row) =>
      (row.meanMae as number) < (best.meanMae as number) ? row : best);
    const bestMae = bestBaseline.meanMae as number;
    const comparable = baselines.filter(row => maeValuesAreComparable(row.meanMae as number, bestMae));
    const callReduction = oracleCallReduction(
      bcad.meanOracleCalls,
      comparable.map(row => row.meanOracleCalls)
    );
    const maeReduction = bestMae > ASAD_MAE_ABSOLUTE_FLOOR
      ? 1 - (bcad.meanMae as number) / bestMae
      : null;
    const maePass = maeReduction !== null && maeReduction >= 0.25;
    const callPass = callReduction !== null
      && callReduction >= 0.3
      && (bcad.meanMae as number) <= bestMae * 1.05 + ASAD_MAE_ABSOLUTE_FLOOR;
    const falseIncrease = bcad.falseAcceptedRunRate - bestBaseline.falseAcceptedRunRate;
    const falseAcceptancePass = falseIncrease <= 0.01;
    const performancePass = maePass || callPass;
    return {
      dimension,
      scenario,
      qualifies: performancePass && falseAcceptancePass,
      queryReduction: callReduction,
      maeReduction,
      falseAcceptedIncrease: falseIncrease,
      performancePass,
      falseAcceptancePass
    };
  }));
  const scenarioWins = MAIN_SCENARIOS.filter(scenario =>
    [16, 32].every(dimension =>
      primaryKeyCells.find(cell => cell.dimension === dimension && cell.scenario === scenario)?.qualifies));
  const recordedWins = summary.verdict.mainScenarioWins;
  assert(JSON.stringify([...scenarioWins].sort()) === JSON.stringify([...recordedWins].sort()),
    'Verdict main scenario wins differ from independent raw calculation');
  assert(summary.verdict.keyCells.length === primaryKeyCells.length, 'Verdict key-cell count mismatch');
  for (const cell of primaryKeyCells) {
    const recorded = summary.verdict.keyCells.find(candidate =>
      candidate.dimension === cell.dimension && candidate.scenario === cell.scenario);
    const matchesNullableNumber = (first: number | null, second: number | null): boolean =>
      first === null || second === null
        ? first === second
        : Math.abs(first - second) <= 1e-10 * Math.max(1, Math.abs(first), Math.abs(second));
    assert(recorded
      && recorded.qualifies === cell.qualifies
      && recorded.performancePass === cell.performancePass
      && recorded.falseAcceptancePass === cell.falseAcceptancePass
      && matchesNullableNumber(recorded.queryReduction, cell.queryReduction)
      && matchesNullableNumber(recorded.maeReduction, cell.maeReduction)
      && matchesNullableNumber(recorded.falseAcceptedIncrease, cell.falseAcceptedIncrease),
      `Verdict key-cell decision differs for ${cell.dimension}|${cell.scenario}`);
  }
  assert(report.includes(`**Run:** ${runId}`), 'Report run id mismatch');
  assert(report.includes(`**Verdict:** ${summary.verdict.verdict}`), 'Report verdict mismatch');
  assert(report.includes('Relative MAE reduction is shown as n/a'),
    'Report omits the near-zero MAE comparison rule');
  const statusCounts = new Map<string, Record<RawRun['status'], number>>();
  for (const method of METHODS) statusCounts.set(method, { complete: 0, incomplete: 0, threw: 0 });
  for (const record of records) {
    const counts = statusCounts.get(record.method);
    assert(counts, `Missing status-count method ${record.method}`);
    counts[record.status]++;
  }
  for (const method of METHODS) {
    const counts = statusCounts.get(method);
    assert(counts && report.includes(
      `| ${method} | ${counts.complete} | ${counts.incomplete} | ${counts.threw} |`
    ), `Report status counts differ from raw records for ${method}`);
  }
  const totalStatuses = [...statusCounts.values()].reduce((total, counts) => ({
    complete: total.complete + counts.complete,
    incomplete: total.incomplete + counts.incomplete,
    threw: total.threw + counts.threw
  }), { complete: 0, incomplete: 0, threw: 0 });
  assert(report.includes(
    `| **Total** | **${totalStatuses.complete}** | **${totalStatuses.incomplete}** | **${totalStatuses.threw}** |`
  ), 'Report total completion/exception counts differ from raw records');
  process.stdout.write(
    `BCAD_RESULT_VERIFICATION_PASSED runs=${records.length} cells=${groups.size} seeds=${summary.seedsPerCell} dimensions=${summary.dimensions.length} scenarios=${summary.scenarios.length} overrun=${actualBudgetOverruns} statuses=verified raw=${summary.rawFile}\n`
  );
}

main();
