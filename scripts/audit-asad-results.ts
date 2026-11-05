import fs from 'node:fs';
import path from 'node:path';
import {
  maeValuesAreComparable,
  oracleCallReduction
} from './asad-benchmark-metrics';

type RawRun = {
  seed: number;
  dimension: number;
  scenario: string;
  method: string;
  oracleBudget: number;
  actualOracleCalls: number;
  diagnosticOracleCalls: number | null;
  oracleCountReconciled: boolean;
  oracleBudgetExceeded: boolean;
  mae: number | null;
};

type SummaryRow = {
  dimension: number;
  scenario: string;
  method: string;
  runs: number;
  scoredRuns: number;
  meanMae: number | null;
  meanOracleCalls: number;
  budgetOverruns: number;
  diagnosticMismatches: number;
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function equalNumber(first: number | null, second: number | null): boolean {
  if (first === null || second === null) return first === second;
  return Math.abs(first - second) <= 1e-10 * Math.max(1, Math.abs(first), Math.abs(second));
}

function parseJsonLines<T>(filePath: string): T[] {
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
  const reportPath = path.join(projectRoot, 'docs/research/2026-10-29-asad-experimental-report.md');
  const report = fs.readFileSync(reportPath, 'utf8');
  const runId = report.match(/^\*\*Run:\*\* (.+)$/m)?.[1].trim();
  assert(runId, 'ASAD report does not name its raw campaign run id');
  const resultsDirectory = path.join(projectRoot, 'scratch/results/asad');
  const summaryPath = path.join(resultsDirectory, `summary-${runId}.json`);
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as {
    runId: string;
    seedsPerCell: number;
    totalRuns: number;
    totalOracleBudgetOverruns: number;
    totalDiagnosticMismatches: number;
    rawFile: string;
    summary: SummaryRow[];
    verdict: { verdict: string; keyCellsMeetingCriteria: number };
  };
  const rawPath = path.join(resultsDirectory, summary.rawFile);
  const records = parseJsonLines<RawRun>(rawPath);
  assert(summary.runId === runId, 'Historical summary run id does not match the report');
  assert(records.length === summary.totalRuns, 'Raw record count does not match historical summary');
  assert(records.length === 11200, `Expected 11,200 historical records, found ${records.length}`);

  const recordKeys = new Set<string>();
  const grouped = new Map<string, RawRun[]>();
  let actualOverruns = 0;
  let actualDiagnosticMismatches = 0;
  for (const record of records) {
    const key = `${record.dimension}|${record.scenario}|${record.seed}|${record.method}`;
    assert(!recordKeys.has(key), `Duplicate raw record ${key}`);
    recordKeys.add(key);
    const groupKey = `${record.dimension}|${record.scenario}|${record.method}`;
    const group = grouped.get(groupKey) ?? [];
    group.push(record);
    grouped.set(groupKey, group);
    assert(record.actualOracleCalls <= record.oracleBudget, `Oracle budget exceeded in ${key}`);
    if (record.diagnosticOracleCalls !== null && record.diagnosticOracleCalls !== record.actualOracleCalls) {
      actualDiagnosticMismatches++;
    }
    if (record.oracleBudgetExceeded) actualOverruns++;
  }
  assert(actualOverruns === summary.totalOracleBudgetOverruns, 'Raw budget overrun count differs from summary');
  assert(actualDiagnosticMismatches === summary.totalDiagnosticMismatches, 'Raw diagnostic mismatch count differs from summary');
  assert(grouped.size === summary.summary.length, 'Raw and summarized cell counts differ');
  for (const row of summary.summary) {
    const group = grouped.get(`${row.dimension}|${row.scenario}|${row.method}`);
    assert(group, `Missing raw records for ${row.dimension}|${row.scenario}|${row.method}`);
    assert(group.length === row.runs && group.length === summary.seedsPerCell, 'Cell seed count mismatch');
    const scored = group.filter((record): record is RawRun & { mae: number } =>
      typeof record.mae === 'number');
    const rawMae = scored.length ? average(scored.map(record => record.mae)) : null;
    const rawCalls = average(group.map(record => record.actualOracleCalls));
    assert(row.scoredRuns === scored.length, 'Scored-run count differs from raw records');
    assert(equalNumber(row.meanMae, rawMae), `Summary MAE differs from raw data for ${row.dimension}|${row.scenario}|${row.method}`);
    assert(equalNumber(row.meanOracleCalls, rawCalls), `Summary call mean differs from raw data for ${row.dimension}|${row.scenario}|${row.method}`);
  }

  const keyScenarios = [
    'isolated-low-rank',
    'multi-crossing-low-rank',
    'near-parallel-low-rank',
    'small-amplitude-low-rank'
  ];
  const auditedCells = [];
  for (const dimension of [16, 32]) {
    for (const scenario of keyScenarios) {
      const get = (method: string): SummaryRow | undefined =>
        summary.summary.find(row => row.dimension === dimension && row.scenario === scenario && row.method === method);
      const asad = get('asad');
      assert(asad && typeof asad.meanMae === 'number', `ASAD has no scored result for ${dimension}|${scenario}`);
      const baselines = ['ransac', 'star-probe', 'adaptive']
        .map(get)
        .filter((row): row is SummaryRow => row !== undefined && typeof row.meanMae === 'number');
      const bestBaselineMae = baselines.length
        ? Math.min(...baselines.map(row => row.meanMae as number))
        : null;
      const comparableBaselines = baselines.filter(row =>
        maeValuesAreComparable(row.meanMae as number, asad.meanMae as number));
      const queryReduction = oracleCallReduction(
        asad.meanOracleCalls,
        comparableBaselines.map(row => row.meanOracleCalls)
      );
      const maeReduction = bestBaselineMae !== null && bestBaselineMae > 1e-12
        ? 1 - asad.meanMae / bestBaselineMae
        : null;
      const queryPass = queryReduction !== null && queryReduction >= 0.2;
      const maePass = maeReduction !== null && maeReduction >= 0.25;
      auditedCells.push({
        dimension,
        scenario,
        asadMae: asad.meanMae,
        bestBaselineMae,
        comparableBaselineMethods: comparableBaselines.map(row => row.method),
        queryReduction,
        maeReduction,
        qualifies: queryPass || maePass
      });
    }
  }
  const auditedKeyCells = auditedCells.filter(cell => cell.qualifies).length;
  assert(summary.verdict.keyCellsMeetingCriteria === 7, 'Historical reported key-cell count changed unexpectedly');
  assert(auditedKeyCells === 0, `Expected no key-cell passes under the stated criteria, found ${auditedKeyCells}`);
  assert(report.includes('Audited verdict under the stated criteria:** not-supported (0 of 8 key cells qualify)'), 'Report omits corrected historical verdict');
  assert(report.includes('Infinity') && report.includes('100% query reduction'), 'Report omits explanation of the historical scorer defect');

  const outputDirectory = path.join(projectRoot, 'scratch/results/bcad');
  fs.mkdirSync(outputDirectory, { recursive: true });
  const outputPath = path.join(outputDirectory, 'asad-historical-audit.json');
  fs.writeFileSync(outputPath, `${JSON.stringify({
    schemaVersion: 1,
    runId,
    rawFile: summary.rawFile,
    totalRawRecords: records.length,
    summaryCellsReconciled: grouped.size,
    actualOracleBudgetOverruns: actualOverruns,
    actualDiagnosticMismatches,
    recordedKeyCellsMeetingCriteria: summary.verdict.keyCellsMeetingCriteria,
    auditedKeyCellsMeetingCriteria: auditedKeyCells,
    auditedCells
  }, null, 2)}\n`);
  process.stdout.write(
    `ASAD_RAW_AUDIT_PASSED reported=${summary.verdict.keyCellsMeetingCriteria} audited=${auditedKeyCells} records=${records.length} cells=${grouped.size}\n`
  );
}

main();
