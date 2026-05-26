#!/usr/bin/env ts-node

import path from 'path';
import { spawnSync } from 'child_process';

type BenchmarkRow = {
  proofLane: {
    laneLabel: string;
    composeChooser: string;
    composeExperimentClass: string;
    composeIncumbent?: string | null;
    composeVetoPolicy?: string | null;
  };
  composeDebug?: {
    thresholdAdaptiveDuelCount?: number;
    thresholdStaticPairedWitnessComposeCount?: number;
    thresholdAdaptivePairedWitnessIncumbentCount?: number;
    thresholdAdaptivePromotedWinnerCount?: number;
    thresholdAdaptiveShadowWinnerCount?: number;
  };
};

const repoRoot = path.resolve(__dirname, '..');
const benchmarkScript = path.join(repoRoot, 'scripts', 'benchmark-majority-seeds.ts');

const THRESHOLD_ENV_KEYS = [
  'GCL_THRESHOLD_COMPOSE_EXPERIMENT',
  'GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT',
  'GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION',
  'GCL_THRESHOLD_ADAPTIVE_DUEL_COMMIT',
  'GCL_THRESHOLD_UPPER_COMPOSE',
  'GCL_THRESHOLD_SELECTION_POLICY'
];

function parseJsonPayload(raw: string): any {
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const start = trimmed.indexOf('{');
    const end = trimmed.lastIndexOf('}');
    if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1));
    throw new Error('benchmark output did not contain parseable JSON');
  }
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function main(): void {
  const env: NodeJS.ProcessEnv = { ...process.env, BENCHMARK_SHAPE: '1' };
  for (const key of THRESHOLD_ENV_KEYS) delete env[key];

  const result = spawnSync(
    path.join(repoRoot, 'node_modules', '.bin', 'ts-node'),
    [benchmarkScript, '9', '5', '1', '1337', 'bridge'],
    {
      cwd: repoRoot,
      env,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024
    }
  );

  if (result.status !== 0) {
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);
    throw new Error(`benchmark-majority-seeds failed with code ${result.status ?? 'unknown'}`);
  }

  const row = parseJsonPayload(result.stdout).summary.bridge.rows[0] as BenchmarkRow;
  const debug = row.composeDebug || {};
  const failures: string[] = [];

  if (row.proofLane.laneLabel !== 'compose-adaptive-duel') failures.push(`lane=${row.proofLane.laneLabel}`);
  if (row.proofLane.composeChooser !== 'duel') failures.push(`chooser=${row.proofLane.composeChooser}`);
  if (row.proofLane.composeExperimentClass !== 'adaptive-duel') failures.push(`class=${row.proofLane.composeExperimentClass}`);
  if (row.proofLane.composeIncumbent !== 'paired-witness') failures.push(`incumbent=${row.proofLane.composeIncumbent}`);
  if (row.proofLane.composeVetoPolicy !== 'paired-witness-incumbent') failures.push(`vetoPolicy=${row.proofLane.composeVetoPolicy}`);
  if ((debug.thresholdAdaptiveDuelCount || 0) <= 0) failures.push('adaptive-duel never executed');
  if ((debug.thresholdAdaptivePairedWitnessIncumbentCount || 0) <= 0) failures.push('paired-witness incumbent was not exercised');
  if ((debug.thresholdStaticPairedWitnessComposeCount || 0) > 0) failures.push('default fell back to static paired-witness compose');

  const report = {
    pass: failures.length === 0,
    proofLane: row.proofLane,
    adaptiveDuelCount: debug.thresholdAdaptiveDuelCount || 0,
    pairedWitnessIncumbentCount: debug.thresholdAdaptivePairedWitnessIncumbentCount || 0,
    promotedWinnerCount: debug.thresholdAdaptivePromotedWinnerCount || 0,
    shadowWinnerCount: debug.thresholdAdaptiveShadowWinnerCount || 0,
    failures
  };

  console.log(JSON.stringify(report, null, 2));
  assert(failures.length === 0, failures.join('; '));
}

try {
  main();
} catch (error) {
  console.error(`\n[default-threshold-policy] FAILED: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
