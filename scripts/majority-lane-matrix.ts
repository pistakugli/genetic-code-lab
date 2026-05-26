#!/usr/bin/env ts-node

import path from 'path';
import { spawn } from 'child_process';
import readline from 'readline';

type DiagnosePayload = {
  n?: number;
  threshold?: number;
  mode?: string;
  seed?: number | null;
  selectionMethod?: string;
  fitness?: number;
  length?: number;
  byWeight?: Array<{ ones: number; total: number; accuracy: number }>;
  histogram?: Record<string, number>;
  boundaryErrors?: Array<{
    ones: number;
    mask: string;
    expected: number;
    predicted: number;
  }>;
};

type MatrixRow = {
  lane: 'preflight-60' | 'seeded-replay-140' | 'unseeded-replay-140';
  generations: number;
  seed: number | null;
  command: string;
  fitness: number;
  length: number;
  weight4Accuracy: number | null;
  weight5Accuracy: number | null;
  boundaryErrors: number;
  fp4: number;
  fn5: number;
};

const repoRoot = path.resolve(__dirname, '..');
const diagnoseScript = path.join(repoRoot, 'scripts', 'diagnose-threshold.ts');
const tsNodeBin = path.join(repoRoot, 'node_modules', '.bin', 'ts-node');

function parseSeedArg(value: string | undefined): number {
  if (value === undefined) return 1337;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 1337;
}

function parseJsonFromMixedOutput(raw: string): DiagnosePayload {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error('diagnose-threshold produced empty output');

  try {
    return JSON.parse(trimmed) as DiagnosePayload;
  } catch {
    const start = trimmed.indexOf('{');
    const end = trimmed.lastIndexOf('}');
    if (start >= 0 && end > start) {
      return JSON.parse(trimmed.slice(start, end + 1)) as DiagnosePayload;
    }
    throw new Error('diagnose-threshold output did not contain parseable JSON');
  }
}

function boundaryStats(payload: DiagnosePayload): { total: number; fp4: number; fn5: number } {
  const boundaryErrors = Array.isArray(payload.boundaryErrors) ? payload.boundaryErrors : [];
  let fp4 = 0;
  let fn5 = 0;

  for (const err of boundaryErrors) {
    if (err.ones === 4 && err.expected === 0 && err.predicted === 1) fp4++;
    if (err.ones === 5 && err.expected === 1 && err.predicted === 0) fn5++;
  }

  return { total: boundaryErrors.length, fp4, fn5 };
}

function buildCommand(generations: number, seed: number | null): string {
  const base = ['./node_modules/.bin/ts-node', '--transpile-only', 'scripts/diagnose-threshold.ts', '9', '5', String(generations), 'bridge'];
  if (seed !== null) {
    base.push(String(seed), 'lexicase', 'dump-errors');
  } else {
    base.push('lexicase', 'dump-errors');
  }
  return base.join(' ');
}

function runDiagnose(generations: number, seed: number | null, lane: string): Promise<DiagnosePayload> {
  const args = ['ts-node', diagnoseScript, '9', '5', String(generations), 'bridge'];
  if (seed !== null) {
    args.push(String(seed), 'lexicase', 'dump-errors');
  } else {
    args.push('lexicase', 'dump-errors');
  }

  return new Promise((resolve, reject) => {
    const child = spawn(tsNodeBin, ['--transpile-only', ...args.slice(1)], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');

    child.stdout.on('data', chunk => {
      stdout += chunk;
    });

    child.stderr.on('data', chunk => {
      stderr += chunk;
    });

    const prefixStream = (stream: NodeJS.ReadableStream, label: string): void => {
      const rl = readline.createInterface({ input: stream });
      rl.on('line', line => {
        process.stderr.write(`[matrix][${label}] ${line}\n`);
      });
    };

    prefixStream(child.stderr, lane);

    child.on('error', error => reject(error));
    child.on('close', code => {
      if (code !== 0) {
        reject(new Error(`[${lane}] diagnose-threshold exited with code ${code ?? 'unknown'}`));
        return;
      }
      resolve(parseJsonFromMixedOutput(`${stdout}\n${stderr}`));
    });
  });
}

function summarizeRow(lane: MatrixRow['lane'], generations: number, seed: number | null, payload: DiagnosePayload): MatrixRow {
  const byWeight4 = payload.byWeight?.find(row => row.ones === 4);
  const byWeight5 = payload.byWeight?.find(row => row.ones === 5);
  const stats = boundaryStats(payload);

  return {
    lane,
    generations,
    seed,
    command: buildCommand(generations, seed),
    fitness: Number((payload.fitness ?? 0).toFixed(8)),
    length: Number(payload.length ?? 0),
    weight4Accuracy: typeof byWeight4?.accuracy === 'number' ? Number(byWeight4.accuracy.toFixed(4)) : null,
    weight5Accuracy: typeof byWeight5?.accuracy === 'number' ? Number(byWeight5.accuracy.toFixed(4)) : null,
    boundaryErrors: stats.total,
    fp4: stats.fp4,
    fn5: stats.fn5
  };
}

async function runLane(lane: MatrixRow['lane'], generations: number, seed: number | null): Promise<MatrixRow> {
  const seedLabel = seed === null ? 'none' : String(seed);
  console.error(`[matrix] ${lane} start gens=${generations} seed=${seedLabel}`);
  const payload = await runDiagnose(generations, seed, lane);
  const row = summarizeRow(lane, generations, seed, payload);
  console.error(
    `[matrix] ${lane} done fitness=${row.fitness.toFixed(8)} `
    + `length=${row.length} w4=${row.weight4Accuracy?.toFixed(4) ?? 'n/a'} `
    + `w5=${row.weight5Accuracy?.toFixed(4) ?? 'n/a'} boundary=${row.boundaryErrors}`
  );
  return row;
}

async function main(): Promise<void> {
  const seed = parseSeedArg(process.argv[2]);
  const laneSpecs: Array<Pick<MatrixRow, 'lane' | 'generations' | 'seed'>> = [
    { lane: 'preflight-60', generations: 60, seed },
    { lane: 'seeded-replay-140', generations: 140, seed },
    { lane: 'unseeded-replay-140', generations: 140, seed: null }
  ];

  const settled = await Promise.allSettled(
    laneSpecs.map(spec => runLane(spec.lane, spec.generations, spec.seed))
  );

  const rejected = settled.find(result => result.status === 'rejected') as PromiseRejectedResult | undefined;
  if (rejected) {
    throw rejected.reason instanceof Error ? rejected.reason : new Error(String(rejected.reason));
  }

  const rows = settled.map(result => (result as PromiseFulfilledResult<MatrixRow>).value);

  console.log(JSON.stringify({
    problem: {
      n: 9,
      threshold: 5,
      mode: 'bridge'
    },
    seed,
    selectionMethod: 'lexicase',
    rows
  }, null, 2));
}

main().catch(error => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
