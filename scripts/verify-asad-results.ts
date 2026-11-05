import fs from 'node:fs';
import path from 'node:path';

const expectedDimensions = [6, 8, 16, 32];
const expectedScenarios = [
  'isolated-low-rank',
  'multi-crossing-low-rank',
  'near-parallel-low-rank',
  'small-amplitude-low-rank',
  'contaminated-local-probes',
  'orthogonal-full-rank',
  'dense-crossing-full-rank'
];
const expectedMethods = ['asad', 'ransac', 'star-probe', 'adaptive'];

function parseSeedArgument(args: string[]): number {
  const index = args.indexOf('--seeds');
  const seeds = Number(index >= 0 ? args[index + 1] : 100);
  if (!Number.isSafeInteger(seeds) || seeds < 1) throw new Error('--seeds must be a positive safe integer');
  return seeds;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function main(): void {
  const seeds = parseSeedArgument(process.argv.slice(2));
  const resultsDirectory = path.resolve('scratch/results/asad');
  const summaries = fs.readdirSync(resultsDirectory)
    .filter(filename => /^summary-.*\.json$/.test(filename))
    .map(filename => ({
      filename,
      modified: fs.statSync(path.join(resultsDirectory, filename)).mtimeMs
    }))
    .sort((a, b) => b.modified - a.modified);
  assert(summaries.length > 0, 'no ASAD benchmark summary exists');
  const summaryPath = path.join(resultsDirectory, summaries[0].filename);
  const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as {
    seedsPerCell: number;
    dimensions: number[];
    scenarios: string[];
    methods: string[];
    totalRuns: number;
    totalOracleBudgetOverruns: number;
    totalDiagnosticMismatches: number;
    rawFile: string;
    verdict: { verdict: string };
  };
  assert(summary.seedsPerCell === seeds, `summary seed count is ${summary.seedsPerCell}, expected ${seeds}`);
  assert(JSON.stringify(summary.dimensions) === JSON.stringify(expectedDimensions), 'benchmark dimensions do not match the required matrix');
  assert(JSON.stringify(summary.scenarios) === JSON.stringify(expectedScenarios), 'benchmark scenarios do not match the required matrix');
  assert(JSON.stringify(summary.methods) === JSON.stringify(expectedMethods), 'benchmark methods do not include exactly ASAD and all three baselines');
  const expectedRunCount = seeds * expectedDimensions.length * expectedScenarios.length * expectedMethods.length;
  assert(summary.totalRuns === expectedRunCount, `summary contains ${summary.totalRuns} runs, expected ${expectedRunCount}`);
  assert(summary.totalOracleBudgetOverruns === 0, `found ${summary.totalOracleBudgetOverruns} oracle budget overruns`);
  assert(summary.totalDiagnosticMismatches === 0, `found ${summary.totalDiagnosticMismatches} oracle diagnostic mismatches`);

  const rawPath = path.join(resultsDirectory, summary.rawFile);
  assert(fs.existsSync(rawPath), `raw run file is missing: ${rawPath}`);
  const rawLines = fs.readFileSync(rawPath, 'utf8').trim().split('\n');
  assert(rawLines.length === expectedRunCount, `raw file contains ${rawLines.length} rows, expected ${expectedRunCount}`);
  const cellCounts = new Map<string, number>();
  const uniqueRuns = new Set<string>();
  for (const [index, line] of rawLines.entries()) {
    const record = JSON.parse(line) as {
      seed: number;
      dimension: number;
      scenario: string;
      method: string;
      status: string;
      oracleBudget: number;
      actualOracleCalls: number;
      diagnosticOracleCalls: number | null;
      oracleCountReconciled: boolean;
      oracleBudgetExceeded: boolean;
      asadDiagnostics?: {
        fitResidualRms: number | null;
        fitResidualRelative: number | null;
        fitValidationObservations: number;
        requiredFitValidationObservations: number;
      };
    };
    const key = `${record.dimension}|${record.scenario}|${record.method}`;
    const identity = `${key}|${record.seed}`;
    assert(!uniqueRuns.has(identity), `duplicate run identity at raw row ${index + 1}: ${identity}`);
    uniqueRuns.add(identity);
    cellCounts.set(key, (cellCounts.get(key) ?? 0) + 1);
    assert(record.actualOracleCalls >= 0 && record.actualOracleCalls <= record.oracleBudget,
      `oracle budget violation at raw row ${index + 1}: ${record.actualOracleCalls}/${record.oracleBudget}`);
    assert(!record.oracleBudgetExceeded, `oracle overrun flag set at raw row ${index + 1}`);
    if (record.method === 'asad') {
      assert(record.asadDiagnostics !== undefined, `ASAD diagnostics missing at raw row ${index + 1}`);
      assert(record.asadDiagnostics.fitValidationObservations >= 0
        && record.asadDiagnostics.requiredFitValidationObservations >= 0,
      `invalid ASAD validation sample count at raw row ${index + 1}`);
      for (const residual of [
        record.asadDiagnostics.fitResidualRms,
        record.asadDiagnostics.fitResidualRelative
      ]) {
        assert(residual === null || (Number.isFinite(residual) && residual >= 0),
          `invalid ASAD fit residual at raw row ${index + 1}`);
      }
    }
    if (record.diagnosticOracleCalls !== null) {
      assert(record.diagnosticOracleCalls === record.actualOracleCalls && record.oracleCountReconciled,
        `oracle count does not reconcile at raw row ${index + 1}`);
    } else {
      assert(record.status === 'threw', `successful solver run has no oracle diagnostic at raw row ${index + 1}`);
    }
  }
  for (const dimension of expectedDimensions) {
    for (const scenario of expectedScenarios) {
      for (const method of expectedMethods) {
        const key = `${dimension}|${scenario}|${method}`;
        assert(cellCounts.get(key) === seeds, `cell ${key} has ${cellCounts.get(key) ?? 0} rows, expected ${seeds}`);
      }
    }
  }
  const reportPath = path.resolve('docs/research/2026-10-29-asad-experimental-report.md');
  assert(fs.existsSync(reportPath), 'the full ASAD research report is missing');
  const report = fs.readFileSync(reportPath, 'utf8');
  assert(report.includes(`**Verdict:** ${summary.verdict.verdict}`), 'report verdict differs from the benchmark summary');
  process.stdout.write(`ASAD_RESULT_VERIFICATION_PASSED runs=${expectedRunCount} cells=${expectedDimensions.length * expectedScenarios.length * expectedMethods.length} raw=${summary.rawFile}\n`);
}

main();
