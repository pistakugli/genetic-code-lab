import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  makeScenario,
  randomTestPoints,
  ScenarioName,
  SCENARIOS,
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
  | 'mbcd-single-rank'
  | 'mbcd-no-contradiction'
  | 'mbcd-no-independent-validation';

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
  activeRankEstimate: number | null;
  expectedActiveRank: number;
  activeRankMisestimated: boolean | null;
  reconstructionComplete: boolean;
  mbcdDiagnostics?: {
    totalOracleCalls: number;
    gradientProbes: number;
    contradictionQueries: number;
    predictionDisagreementQueries: number;
    acquisitionQueries: number;
    reconstructionComplete: boolean;
    unresolvedHypothesisCount: number;
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
  bestBaseline: Method | null;
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

const REQUIRED_DIMENSIONS = [16, 32];
const REQUIRED_SCENARIOS: ScenarioName[] = [
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
const REQUIRED_METHODS: Method[] = [
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
const BASELINES: Method[] = ['asad', 'ransac', 'star-probe', 'adaptive', 'bcad'];
const MAE_TOLERANCE = 0.05;
const FALSE_ACCEPTANCE_TOLERANCE = 0.01;
const EVALUATION_POINTS = 192;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function hash(value: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function nearlyEqual(first: number | null, second: number | null, tolerance = 1e-9): boolean {
  if (first === null || second === null) return first === second;
  return Math.abs(first - second) <= tolerance * Math.max(1, Math.abs(first), Math.abs(second));
}

function readJsonLines<T>(filePath: string): T[] {
  const contents = fs.readFileSync(filePath, 'utf8');
  assert(contents.endsWith('\n'), `Raw results lack a final newline: ${filePath}`);
  return contents.trim().split('\n').filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as T;
    } catch (error) {
      throw new Error(`Invalid JSON in ${filePath}:${index + 1}: ${String(error)}`);
    }
  });
}

function compareSummary(actual: SummaryRow[], expected: SummaryRow[]): void {
  assert(actual.length === expected.length, 'Summary row count differs from raw-data grouping');
  const keys = new Set<string>();
  for (const row of actual) {
    const key = `${row.dimension}|${row.scenario}|${row.method}`;
    assert(!keys.has(key), `Duplicate summary row ${key}`);
    keys.add(key);
    const computed = expected.find(candidate =>
      candidate.dimension === row.dimension
      && candidate.scenario === row.scenario
      && candidate.method === row.method);
    assert(computed, `Summary row ${key} is absent from raw data`);
    for (const field of [
      'runs', 'scoredRuns', 'completeRuns', 'incompleteRuns', 'thrownRuns'
    ] as const) {
      assert(row[field] === computed[field], `${field} mismatch for ${key}`);
    }
    for (const field of [
      'reconstructionCompleteRate',
      'meanMae',
      'meanOracleCalls',
      'meanNormalErrorDeg',
      'meanOffsetError',
      'meanCorrectHyperplanes',
      'meanFalseAcceptedHyperplanes',
      'falseAcceptedRunRate',
      'meanUnresolvedHypotheses',
      'activeRankMisestimateRate',
      'meanRuntimeMs'
    ] as const) {
      assert(nearlyEqual(row[field], computed[field]), `${field} mismatch for ${key}`);
    }
  }
}

function summarizeRaw(records: RawRun[]): SummaryRow[] {
  const groups = new Map<string, RawRun[]>();
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
    const mbcdRuns = group.filter(record => record.mbcdDiagnostics);
    const rankScored = group.filter(record => record.activeRankMisestimated !== null);
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
      meanUnresolvedHypotheses: mbcdRuns.length
        ? mean(mbcdRuns.map(record => record.mbcdDiagnostics!.unresolvedHypothesisCount))
        : null,
      activeRankMisestimateRate: rankScored.length
        ? rankScored.filter(record => record.activeRankMisestimated).length / rankScored.length
        : null,
      meanRuntimeMs: mean(group.map(record => record.runtimeMs))
    };
  });
}

function recomputeCell(rows: SummaryRow[], dimension: number, scenario: ScenarioName): CellAssessment {
  const group = rows.filter(row => row.dimension === dimension && row.scenario === scenario);
  const primary = group.find(row => row.method === 'mbcd-full');
  const baselines = group.filter(row =>
    BASELINES.includes(row.method) && row.meanMae !== null && row.scoredRuns > 0);
  const orderedBaselines = [...baselines].sort((first, second) => first.meanMae! - second.meanMae!);
  const best = orderedBaselines[0] ?? null;
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
  const calls = comparable.map(row => row.meanOracleCalls);
  const bestComparableCalls = calls.length ? Math.min(...calls) : null;
  const callReduction = bestComparableCalls !== null && bestComparableCalls > 0
    ? 1 - primary.meanOracleCalls / bestComparableCalls
    : null;
  const maeReduction = best.meanMae > ASAD_MAE_ABSOLUTE_FLOOR
    ? 1 - primary.meanMae / best.meanMae
    : null;
  const maePass = maeReduction !== null && maeReduction >= 0.25;
  const efficiencyPass = callReduction !== null
    && callReduction >= 0.3
    && primary.meanMae <= best.meanMae * (1 + MAE_TOLERANCE) + ASAD_MAE_ABSOLUTE_FLOOR;
  const referenceRates = comparable.map(row => row.falseAcceptedRunRate)
    .filter((value): value is number => value !== null);
  const baselineFalseRate = referenceRates.length ? Math.min(...referenceRates) : null;
  const falseIncrease = primary.falseAcceptedRunRate !== null && baselineFalseRate !== null
    ? primary.falseAcceptedRunRate - baselineFalseRate
    : null;
  const falsePass = falseIncrease !== null && falseIncrease <= FALSE_ACCEPTANCE_TOLERANCE;
  const performancePass = maePass || efficiencyPass;
  const qualifies = performancePass && falsePass;
  return {
    dimension,
    scenario,
    bestBaseline: best.method,
    mbcdMae: primary.meanMae,
    bestBaselineMae: best.meanMae,
    mbcdOracleCalls: primary.meanOracleCalls,
    bestComparableBaselineCalls: bestComparableCalls,
    oracleCallReduction: callReduction,
    maeReduction,
    mbcdReconstructionCompleteRate: primary.reconstructionCompleteRate,
    bestBaselineReconstructionCompleteRate: best.reconstructionCompleteRate,
    mbcdFalseAcceptedRunRate: primary.falseAcceptedRunRate,
    baselineFalseAcceptedRunRate: baselineFalseRate,
    falseAcceptedIncrease: falseIncrease,
    performancePass,
    falseAcceptancePass: falsePass,
    qualifies,
    reason: qualifies
      ? maePass
        ? 'mae_improved_by_at_least_25_percent'
        : 'at_least_30_percent_fewer_calls_with_mae_within_5_percent'
      : !performancePass
        ? 'neither_predeclared_performance_threshold_met'
        : 'false_accepted_run_rate_increased_by_more_than_1_percentage_point'
  };
}

function run(): void {
  const projectRoot = path.resolve(__dirname, '..');
  const reportFiles = fs.readdirSync(path.join(projectRoot, 'docs/research'))
    .filter(file => /^\d{4}-\d{2}-\d{2}-mbcd-experimental-report\.md$/.test(file))
    .sort();
  assert(reportFiles.length > 0, 'No MBCD experimental report was found');
  const reportPath = path.join(projectRoot, 'docs/research', reportFiles[reportFiles.length - 1]);
  const report = fs.readFileSync(reportPath, 'utf8');
  const runId = report.match(/^\*\*Run:\*\* (.+)$/m)?.[1].trim();
  assert(runId, 'MBCD report does not name a run id');
  const resultsDirectory = path.join(projectRoot, 'scratch/results/mbcd');
  const summaryPath = path.join(resultsDirectory, `summary-${runId}.json`);
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as CampaignSummary;
  assert(summary.rawFile === path.basename(summary.rawFile)
    && /^mbcd-.*-raw\.jsonl$/.test(summary.rawFile),
  'MBCD raw-file name is not a campaign artifact basename');
  const records = readJsonLines<RawRun>(path.join(resultsDirectory, summary.rawFile));

  assert(summary.runId === runId, 'MBCD report and summary run ids differ');
  assert(summary.seedsPerCell >= 100, 'MBCD acceptance campaign requires at least 100 seeds per cell');
  assert(summary.dimensions.length === REQUIRED_DIMENSIONS.length
    && REQUIRED_DIMENSIONS.every(dimension => summary.dimensions.includes(dimension)),
  'MBCD campaign must include dimensions 16 and 32');
  assert(summary.scenarios.length === REQUIRED_SCENARIOS.length
    && REQUIRED_SCENARIOS.every(scenario => summary.scenarios.includes(scenario)),
  'MBCD campaign omits a required focus scenario');
  assert(summary.methods.length === REQUIRED_METHODS.length
    && REQUIRED_METHODS.every(method => summary.methods.includes(method)),
  'MBCD campaign omits an existing method or required ablation');
  assert(summary.evaluationPoints === EVALUATION_POINTS, 'MBCD evaluation point count changed');
  assert(summary.verdict.seedsPerCell === summary.seedsPerCell, 'Verdict seed count differs from campaign seed count');
  assert(summary.verdict.primaryMethod === 'mbcd-full', 'Primary MBCD method changed');
  assert(summary.verdict.falseAcceptanceTolerance === FALSE_ACCEPTANCE_TOLERANCE,
    'False-acceptance threshold changed after campaign');
  assert(summary.verdict.maeComparisonTolerance === MAE_TOLERANCE,
    'MAE comparability threshold changed after campaign');
  assert(summary.verdict.absoluteMaeFloor === ASAD_MAE_ABSOLUTE_FLOOR,
    'MAE absolute floor differs from the declared shared metric');

  const expectedTotal = REQUIRED_DIMENSIONS.length
    * REQUIRED_SCENARIOS.length
    * summary.seedsPerCell
    * REQUIRED_METHODS.length;
  assert(records.length === expectedTotal && summary.totalRuns === expectedTotal,
    `Expected ${expectedTotal} raw rows; found ${records.length}`);
  const uniqueKeys = new Set<string>();
  const pairedFixtures = new Map<string, { model: string; evaluation: string; methods: Set<Method> }>();
  let actualBudgetOverruns = 0;
  let diagnosticMismatches = 0;
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.seed}|${record.method}`;
    assert(!uniqueKeys.has(key), `Duplicate run ${key}`);
    uniqueKeys.add(key);
    assert(record.schemaVersion === 1, `Unsupported raw schema in ${key}`);
    assert(REQUIRED_DIMENSIONS.includes(record.dimension), `Unexpected dimension in ${key}`);
    assert(REQUIRED_SCENARIOS.includes(record.scenario), `Unexpected scenario in ${key}`);
    assert(REQUIRED_METHODS.includes(record.method), `Unexpected method in ${key}`);
    assert(Number.isSafeInteger(record.seed)
      && record.seed >= 1 && record.seed <= summary.seedsPerCell, `Unexpected seed in ${key}`);
    assert(Number.isFinite(record.actualOracleCalls) && record.actualOracleCalls >= 0,
      `Invalid actual oracle count in ${key}`);
    assert(Number.isFinite(record.oracleBudget) && record.oracleBudget >= 0,
      `Invalid oracle budget in ${key}`);
    assert(record.actualOracleCalls <= record.oracleBudget, `Actual oracle budget overrun in ${key}`);
    assert(record.oracleBudgetExceeded === (record.actualOracleCalls > record.oracleBudget),
      `Budget-overrun flag disagrees with actual count in ${key}`);
    assert(record.solverSeed === record.seed * 104729
      + record.dimension * 13007
      + SCENARIOS.indexOf(record.scenario) * 8191, `Solver seed mismatch in ${key}`);
    const scenario = makeScenario(record.scenario, record.dimension, record.seed);
    assert(record.scenarioModelHash === hash(scenario), `Teacher hash mismatch in ${key}`);
    assert(record.evaluationPointsHash === hash(randomTestPoints(record.seed, record.dimension)),
      `Evaluation-point hash mismatch in ${key}`);
    const expectedBudget = solverBudget(record.dimension, scenario.terms.length).budget;
    assert(record.oracleBudget === expectedBudget, `Oracle budget mismatch in ${key}`);
    assert(record.expectedActiveRank === scenario.expectedActiveRank, `Expected-rank metadata mismatch in ${key}`);
    assert(record.reconstructionComplete === (record.status === 'complete'),
      `Completion flag and run status disagree in ${key}`);
    if (record.diagnosticOracleCalls === null) {
      assert(record.status === 'threw', `Missing diagnostic count on a non-throwing run ${key}`);
      assert(record.oracleCountReconciled, `Throwing run did not mark its unavailable count in ${key}`);
    } else {
      assert(record.diagnosticOracleCalls === record.actualOracleCalls,
        `Actual/diagnostic oracle counts differ in ${key}`);
      assert(record.oracleCountReconciled, `Oracle count reconciliation flag is false in ${key}`);
    }
    if (record.mbcdDiagnostics) {
      assert(record.mbcdDiagnostics.totalOracleCalls === record.actualOracleCalls,
        `MBCD nested diagnostics disagree with actual calls in ${key}`);
      assert(record.mbcdDiagnostics.reconstructionComplete === record.reconstructionComplete,
        `MBCD nested completion flag disagrees in ${key}`);
      assert(record.mbcdDiagnostics.contradictionQueries >= 0
        && record.mbcdDiagnostics.predictionDisagreementQueries >= 0
        && record.mbcdDiagnostics.acquisitionQueries
          >= record.mbcdDiagnostics.contradictionQueries + record.mbcdDiagnostics.predictionDisagreementQueries,
      `MBCD acquisition counters are inconsistent in ${key}`);
      if (record.status !== 'threw') {
        assert(record.mbcdDiagnostics.gradientProbes * (2 * record.dimension + 1)
          === record.actualOracleCalls, `MBCD stencil count does not equal actual calls in ${key}`);
      }
    }
    if (record.activeRankEstimate === null) {
      assert(record.activeRankMisestimated === null, `Missing active rank has a status in ${key}`);
    } else {
      assert(record.activeRankMisestimated === (record.activeRankEstimate !== scenario.expectedActiveRank),
        `Rank-misestimate flag mismatch in ${key}`);
    }
    if (record.status === 'threw') {
      assert(record.mae === null, `Throwing run has a synthetic MAE in ${key}`);
    } else {
      assert(record.mae !== null && Number.isFinite(record.mae), `Scored run has no MAE in ${key}`);
    }
    const pairedKey = `${record.dimension}|${record.scenario}|${record.seed}`;
    const pair = pairedFixtures.get(pairedKey) ?? {
      model: record.scenarioModelHash,
      evaluation: record.evaluationPointsHash,
      methods: new Set<Method>()
    };
    assert(pair.model === record.scenarioModelHash, `Teacher differs across paired methods in ${pairedKey}`);
    assert(pair.evaluation === record.evaluationPointsHash,
      `Evaluation points differ across paired methods in ${pairedKey}`);
    assert(!pair.methods.has(record.method), `Duplicate method pairing in ${pairedKey}`);
    pair.methods.add(record.method);
    pairedFixtures.set(pairedKey, pair);
    if (record.oracleBudgetExceeded) actualBudgetOverruns++;
    if (record.diagnosticOracleCalls !== null && !record.oracleCountReconciled) diagnosticMismatches++;
  }
  assert(pairedFixtures.size === REQUIRED_DIMENSIONS.length * REQUIRED_SCENARIOS.length * summary.seedsPerCell,
    'Paired fixture cell count is incomplete');
  for (const [key, pair] of pairedFixtures) {
    assert(pair.methods.size === REQUIRED_METHODS.length, `Incomplete method pairing in ${key}`);
  }
  assert(actualBudgetOverruns === 0, 'At least one actual oracle budget was exceeded');
  assert(actualBudgetOverruns === summary.totalOracleBudgetOverruns
    && actualBudgetOverruns === summary.verdict.budgetOverruns,
  'Budget-overrun totals disagree with raw data');
  assert(diagnosticMismatches === summary.totalDiagnosticMismatches && diagnosticMismatches === 0,
    'Oracle diagnostic mismatch totals disagree with raw data');
  assert(summary.totalRuns === records.length, 'Campaign total differs from raw rows');

  const statusTotals = {
    complete: records.filter(record => record.status === 'complete').length,
    incomplete: records.filter(record => record.status === 'incomplete').length,
    threw: records.filter(record => record.status === 'threw').length
  };
    assert(statusTotals.complete === summary.statusTotals.complete
      && statusTotals.incomplete === summary.statusTotals.incomplete
      && statusTotals.threw === summary.statusTotals.threw,
    'Raw complete/incomplete/threw totals differ from summary');
  const recomputedRows = summarizeRaw(records);
  compareSummary(summary.summary, recomputedRows);
  const recomputedCells = REQUIRED_DIMENSIONS.flatMap(dimension =>
    PRIMARY_SCENARIOS.map(scenario => recomputeCell(recomputedRows, dimension, scenario)));
  assert(summary.verdict.keyCells.length === recomputedCells.length, 'Primary cell count differs from raw data');
  for (let index = 0; index < recomputedCells.length; index++) {
    const actual = summary.verdict.keyCells[index];
    const expected = recomputedCells[index];
    assert(actual.dimension === expected.dimension && actual.scenario === expected.scenario,
      `Primary-cell ordering mismatch at index ${index}`);
    for (const field of [
      'bestBaseline', 'performancePass', 'falseAcceptancePass', 'qualifies', 'reason'
    ] as const) {
      assert(actual[field] === expected[field], `${field} mismatch for ${expected.dimension}D ${expected.scenario}`);
    }
    for (const field of [
      'mbcdMae',
      'bestBaselineMae',
      'mbcdOracleCalls',
      'bestComparableBaselineCalls',
      'oracleCallReduction',
      'maeReduction',
      'mbcdReconstructionCompleteRate',
      'bestBaselineReconstructionCompleteRate',
      'mbcdFalseAcceptedRunRate',
      'baselineFalseAcceptedRunRate',
      'falseAcceptedIncrease'
    ] as const) {
      assert(nearlyEqual(actual[field], expected[field]),
        `${field} mismatch for ${expected.dimension}D ${expected.scenario}`);
    }
  }
  const winningScenarios = PRIMARY_SCENARIOS.filter(scenario =>
    REQUIRED_DIMENSIONS.every(dimension =>
      recomputedCells.some(cell =>
        cell.dimension === dimension && cell.scenario === scenario && cell.qualifies)));
  assert(JSON.stringify(winningScenarios) === JSON.stringify(summary.verdict.primaryScenarioWins),
    'Qualifying primary scenario list differs from raw-data recomputation');
  const expectedVerdict = actualBudgetOverruns > 0
    ? 'not-supported'
    : winningScenarios.length >= 2
      ? 'promising'
      : winningScenarios.length === 1
        ? 'partially-supported'
        : 'not-supported';
  assert(summary.verdict.verdict === expectedVerdict, 'Verdict differs from raw-data recomputation');
  assert(report.includes(`**Verdict:** ${expectedVerdict}`), 'Report verdict differs from verified raw-data verdict');
  assert(report.includes(`**Run:** ${runId}`), 'Report run id changed during campaign');
  assert(report.includes('reconstruction-complete rate') || report.includes('complete rate'),
    'Report omits reconstruction-completion rates');
  assert(report.includes('at least 25% lower MAE')
    && report.includes('at least 30% fewer calls')
    && report.includes('one percentage point'),
  'Report does not document the predeclared performance and false-acceptance thresholds');
  assert(report.includes('false-accepted'), 'Report omits false-acceptance metrics');
  assert(report.includes('Normal error (deg)') && report.includes('Offset error'),
    'Report omits separate normal/offset error metrics');
  assert(report.includes('not derived from test pass status'), 'Report omits the outcome-vs-test distinction');

  console.log(`MBCD_RESULT_VERIFICATION_PASSED runs=${records.length} seeds=${summary.seedsPerCell} cells=${pairedFixtures.size} overrun=${actualBudgetOverruns} diagnosticMismatches=${diagnosticMismatches} verdict=${expectedVerdict}`);
}

run();
