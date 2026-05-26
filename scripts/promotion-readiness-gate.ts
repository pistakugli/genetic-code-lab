#!/usr/bin/env ts-node

import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';

type PromotionQualityReport = {
  recommendation: 'hold-regression' | 'promote-candidate' | 'keep-opt-in' | 'keep-shadow';
  status: 'positive-use-case' | 'tie/no-regression' | 'regression';
  deltas: {
    exactAccuracy: number;
    boundaryTotal: number;
    fitness: number;
    fp4?: number;
    fn5?: number;
    boundaryGap?: number;
  };
  promotionTelemetry: {
    duelCount: number;
    shadowTotal: number;
    promotedTotal: number;
    championBlocks: number;
    promotedRate?: number;
    promotedByKind?: Record<string, number>;
  };
};

const repoRoot = path.resolve(__dirname, '..');
const tsNode = path.join(repoRoot, 'node_modules', '.bin', 'ts-node');
const workDir = process.env.GCL_PROMOTION_GATE_WORKDIR || path.join(os.tmpdir(), 'gcl-promotion-gate');
const calibrationPath = process.env.GCL_PROMOTION_GATE_CALIBRATION || path.join(workDir, 'promotion-calibration.json');
const qualityPath = path.join(workDir, 'promotion-quality.json');

function arg(name: string, fallback: string): string {
  return process.env[name] || fallback;
}

function parseJsonPayload<T>(raw: string): T {
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    const start = trimmed.indexOf('{');
    const end = trimmed.lastIndexOf('}');
    if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1)) as T;
    throw new Error('command output did not contain parseable JSON');
  }
}

function buildStepEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const merged = { ...process.env, ...env };
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) delete merged[key];
  }
  return merged;
}

function runStep(name: string, command: string, args: string[], env: NodeJS.ProcessEnv = {}): string {
  console.error(`[promotion-gate] ${name}`);
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    env: buildStepEnv(env),
    encoding: 'utf8',
    timeout: Number(process.env.GCL_PROMOTION_GATE_TIMEOUT_MS || 10 * 60 * 1000),
    maxBuffer: 256 * 1024 * 1024
  });

  if (result.stderr) process.stderr.write(result.stderr);
  if (result.error) throw result.error;
  if (result.status !== 0) {
    if (result.stdout) process.stdout.write(result.stdout);
    throw new Error(`${name} failed with exit code ${result.status ?? 'unknown'}`);
  }
  return result.stdout || '';
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function runCalibration(): void {
  if (process.env.GCL_PROMOTION_GATE_REUSE_CALIBRATION === '1') {
    assert(fs.existsSync(calibrationPath), `missing reused calibration file: ${calibrationPath}`);
    return;
  }

  const defaultPromotionEnv: NodeJS.ProcessEnv = {
    GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION: undefined,
    GCL_THRESHOLD_ADAPTIVE_DUEL_COMMIT: undefined
  };
  const raw = runStep('default promotion calibration', tsNode, [
    path.join(repoRoot, 'scripts', 'compose-routing-calibration.ts'),
    arg('GCL_PROMOTION_GATE_N', '9'),
    arg('GCL_PROMOTION_GATE_THRESHOLD', '5'),
    arg('GCL_PROMOTION_GATE_HORIZONS', '5'),
    arg('GCL_PROMOTION_GATE_SEEDS', '1337,2026,4242'),
    arg('GCL_PROMOTION_GATE_MODES', 'bridge'),
    arg('GCL_PROMOTION_GATE_STRATEGIES', 'adaptive,adaptive-duel'),
    arg('GCL_PROMOTION_GATE_SELECTION', 'weighted')
  ], defaultPromotionEnv);
  fs.writeFileSync(calibrationPath, raw);
}

function main(): void {
  fs.mkdirSync(workDir, { recursive: true });
  runCalibration();

  const qualityRaw = runStep('promotion quality report', tsNode, [
    path.join(repoRoot, 'scripts', 'promotion-quality-report.ts'),
    calibrationPath
  ]);
  fs.writeFileSync(qualityPath, qualityRaw);
  const quality = parseJsonPayload<PromotionQualityReport>(qualityRaw);

  const failures: string[] = [];
  if (quality.recommendation !== 'promote-candidate') {
    failures.push(`recommendation=${quality.recommendation}, expected promote-candidate`);
  }
  if (quality.status !== 'positive-use-case') {
    failures.push(`status=${quality.status}, expected positive-use-case`);
  }
  if (quality.deltas.exactAccuracy < -0.000001) {
    failures.push(`exact accuracy regressed: ${quality.deltas.exactAccuracy}`);
  }
  if (quality.deltas.boundaryTotal > 0.000001) {
    failures.push(`boundary total regressed: ${quality.deltas.boundaryTotal}`);
  }
  if (quality.deltas.fitness < -0.000001) {
    failures.push(`fitness regressed: ${quality.deltas.fitness}`);
  }
  if (quality.promotionTelemetry.duelCount <= 0) failures.push('no adaptive-duel decisions');
  if (quality.promotionTelemetry.promotedTotal <= 0) failures.push('no promoted winners under default promotion');
  if (quality.promotionTelemetry.championBlocks <= 0) failures.push('champion safety block was not exercised');

  const payload = {
    pass: failures.length === 0,
    artifacts: {
      calibration: calibrationPath,
      quality: qualityPath
    },
    quality,
    failures,
    note: failures.length === 0
      ? 'Adaptive-duel default promotion proof passed.'
      : 'Adaptive-duel default promotion failed its proof.'
  };

  console.log(JSON.stringify(payload, null, 2));
  if (failures.length > 0) process.exit(1);
}

try {
  main();
} catch (error) {
  console.error(`\n[promotion-gate] FAILED: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
