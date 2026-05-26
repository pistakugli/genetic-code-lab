#!/usr/bin/env ts-node

import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import { resolveBenchmarkReportPath } from './benchmark-paths';

type BaselineName = 'gcl' | 'gcl-hybrid' | 'gplearn' | 'deap';

type SummaryRow = {
  taskId: string;
  baseline: BaselineName;
  runs: number;
  failedRuns: number;
  successRate: number;
  meanMae: number;
};

type OpenSourceBenchmarkReport = {
  generatedAt: string;
  benchmark: 'open-source-symbolic-regression';
  budget?: {
    taskSet?: string;
  };
  summary: SummaryRow[];
  claimAudit?: {
    pureGclVsExternal: {
      status: 'positive-smoke' | 'blocked';
      taskCount: number;
      winRate: number;
      comparableRate: number;
      failures: string[];
    };
    hybridVsPureGcl: {
      status: 'production-candidate' | 'limited-pass' | 'blocked';
      taskCount: number;
      winRate: number;
      regressionCount: number;
      successRegressionCount: number;
      solvedRate: number;
      failures: string[];
      recommendation: string;
    };
  };
};

const repoRoot = path.resolve(__dirname, '..');
const tsNode = path.join(repoRoot, 'node_modules', '.bin', 'ts-node');
const gateMode = process.env.GCL_OPEN_SOURCE_GATE_MODE === 'hybrid-production' ? 'hybrid-production' : 'pure-smoke';
const taskSet = process.env.GCL_BENCH_TASK_SET === 'wide' ? 'wide' : 'smoke';
const reportVariant = gateMode === 'hybrid-production' ? 'hybrid-wide' : taskSet;
const reportPath = resolveBenchmarkReportPath(repoRoot, 'open-source-symbolic-regression', reportVariant);

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parsePositiveRatio(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) return fallback;
  return parsed;
}

function loadReport(): OpenSourceBenchmarkReport {
  const raw = fs.readFileSync(reportPath, 'utf8');
  return JSON.parse(raw) as OpenSourceBenchmarkReport;
}

function runBenchmark(): void {
  if (process.env.GCL_OPEN_SOURCE_GATE_REUSE_REPORT === '1') return;

  const result = spawnSync(tsNode, [path.join(repoRoot, 'scripts', 'benchmark-open-source.ts')], {
    cwd: repoRoot,
    env: process.env,
    encoding: 'utf8',
    timeout: 10 * 60 * 1000,
    maxBuffer: 256 * 1024 * 1024
  });

  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.error) throw result.error;
  assert(result.status === 0, `benchmark:opensource failed with exit code ${result.status ?? 'unknown'}`);
}

function main(): void {
  runBenchmark();

  const report = loadReport();
  assert(report.benchmark === 'open-source-symbolic-regression', 'unexpected benchmark report');
  assert(Array.isArray(report.summary) && report.summary.length > 0, 'benchmark report has no summary rows');
  assert(report.claimAudit, 'benchmark report has no claim audit');

  const requiredBaselines: BaselineName[] = ['gcl', 'gplearn', 'deap'];
  const taskIds = Array.from(new Set(report.summary.map(row => row.taskId))).sort();
  assert(taskIds.length > 0, 'benchmark report has no tasks');

  let gclWinnerCount = 0;
  let gclComparableCount = 0;
  const failures: string[] = [];

  for (const taskId of taskIds) {
    const rows = report.summary.filter(row => row.taskId === taskId);
    for (const baseline of requiredBaselines) {
      const row = rows.find(candidate => candidate.baseline === baseline);
      if (!row) {
        failures.push(`${taskId}/${baseline}: missing summary row`);
        continue;
      }
      if (row.runs <= 0) failures.push(`${taskId}/${baseline}: no runs`);
      if (row.failedRuns !== 0) failures.push(`${taskId}/${baseline}: failedRuns=${row.failedRuns}`);
      if (!Number.isFinite(row.meanMae)) failures.push(`${taskId}/${baseline}: non-finite meanMae`);
    }

    const gcl = rows.find(row => row.baseline === 'gcl');
    if (!gcl) continue;
    if (gcl.successRate < 1) failures.push(`${taskId}/gcl: successRate=${gcl.successRate}`);

    const externalRows = rows.filter(row => row.baseline === 'gplearn' || row.baseline === 'deap');
    if (externalRows.length === 0) {
      failures.push(`${taskId}: no external baseline rows`);
      continue;
    }
    const bestExternalMae = Math.min(...externalRows.map(row => row.meanMae));
    if (gcl.meanMae <= bestExternalMae) gclWinnerCount++;
    if (gcl.meanMae <= bestExternalMae * 1.05 + 1e-12) gclComparableCount++;
  }

  if (gateMode === 'pure-smoke') {
    assert(failures.length === 0, `open-source benchmark smoke gate failures:\n${failures.join('\n')}`);
  }

  const minimumWinRate = parsePositiveRatio('GCL_OPEN_SOURCE_GATE_MIN_WIN_RATE', 0.75);
  const minimumComparableRate = parsePositiveRatio('GCL_OPEN_SOURCE_GATE_MIN_COMPARABLE_RATE', 1);
  const winRate = gclWinnerCount / taskIds.length;
  const comparableRate = gclComparableCount / taskIds.length;

  if (gateMode === 'pure-smoke') {
    assert(winRate >= minimumWinRate, `pure GCL win rate ${winRate} below ${minimumWinRate}`);
    assert(comparableRate >= minimumComparableRate, `pure GCL comparable rate ${comparableRate} below ${minimumComparableRate}`);
    assert(report.claimAudit.pureGclVsExternal.status === 'positive-smoke', `pure GCL claim audit is ${report.claimAudit.pureGclVsExternal.status}: ${report.claimAudit.pureGclVsExternal.failures.join('; ')}`);
  }
  if (gateMode === 'hybrid-production') {
    const hybrid = report.claimAudit.hybridVsPureGcl;
    assert((report.budget?.taskSet ?? 'smoke') === 'wide', 'hybrid production gate requires GCL_BENCH_TASK_SET=wide');
    assert(hybrid.taskCount >= 8, `hybrid production gate needs at least 8 tasks, got ${hybrid.taskCount}`);
    assert(hybrid.status === 'production-candidate', `hybrid claim audit is ${hybrid.status}: ${hybrid.failures.join('; ')}`);
    assert(hybrid.solvedRate >= 1, `hybrid solved rate ${hybrid.solvedRate} below 1`);
    assert(hybrid.regressionCount === 0, `hybrid regressionCount=${hybrid.regressionCount}`);
    assert(hybrid.successRegressionCount === 0, `hybrid successRegressionCount=${hybrid.successRegressionCount}`);
    assert(hybrid.winRate >= 0.5, `hybrid winRate ${hybrid.winRate} below 0.5`);
  }
  if (report.claimAudit.hybridVsPureGcl.status === 'blocked') {
    assert(report.claimAudit.hybridVsPureGcl.failures.length > 0 || report.claimAudit.hybridVsPureGcl.winRate < 0.75, 'hybrid blocked status lacks a blocking reason');
  }

  console.log(JSON.stringify({
    pass: true,
    benchmark: report.benchmark,
    generatedAt: report.generatedAt,
    taskCount: taskIds.length,
    gclWinnerCount,
    gclComparableCount,
    winRate,
    comparableRate,
    thresholds: {
      minimumWinRate,
      minimumComparableRate,
      gateMode
    },
    claimAudit: report.claimAudit,
    note: gateMode === 'hybrid-production'
      ? 'Hybrid production gate. This establishes the gated hybrid capability only, not a pure-GCL public superiority claim.'
      : 'Pure-GCL smoke gate only. This does not establish a public superiority claim.'
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(`\n[open-source-gate] FAILED: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
