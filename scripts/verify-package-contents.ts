#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';

type PackedFile = { path: string; size: number; mode: number };
type PackReport = { files: PackedFile[]; entryCount: number; size: number; unpackedSize: number };

const forbiddenPatterns = [
  /^audit\//,
  /^REPORTS\//,
  /^proof_pack(?:\/|$)/,
  /^tests\//,
  /^src\//,
  /^dist\/tools\//,
  /^dist\/diag\//,
  /^node_modules\//,
  /^runs\//,
  /^result_island_.*\.json$/,
  /^audit_.*\.json$/,
  /^output.*\.log$/,
  /^reauth_.*\.log$/,
  /\.tar\.gz$/,
  /\.log$/
];

const requiredFiles = [
  'package.json',
  'README.md',
  'dist/index.js',
  'dist/index.d.ts',
  'dist/cli/index.js',
  'dist/server/index.js',
  'dist/server/dashboard.html',
  'configs/math-quadratic.json'
];

function main(): void {
  const result = spawnSync('npm', ['pack', '--dry-run', '--json'], {
    cwd: process.cwd(),
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024
  });

  if (result.status !== 0) {
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);
    throw new Error(`npm pack --dry-run failed with exit code ${result.status ?? 'unknown'}`);
  }

  const reports = JSON.parse(result.stdout) as PackReport[];
  const report = reports[0];
  if (!report) throw new Error('npm pack produced no report');

  const paths = report.files.map(file => file.path).sort();
  const missing = requiredFiles.filter(file => !paths.includes(file));
  const forbidden = paths.filter(file => forbiddenPatterns.some(pattern => pattern.test(file)));

  if (missing.length > 0 || forbidden.length > 0) {
    console.error(JSON.stringify({
      pass: false,
      missing,
      forbidden,
      entryCount: report.entryCount,
      packageSize: report.size,
      unpackedSize: report.unpackedSize
    }, null, 2));
    process.exit(1);
  }

  console.log(JSON.stringify({
    pass: true,
    entryCount: report.entryCount,
    packageSize: report.size,
    unpackedSize: report.unpackedSize,
    files: process.env.GCL_PACKAGE_CHECK_VERBOSE === '1' ? paths : undefined
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
