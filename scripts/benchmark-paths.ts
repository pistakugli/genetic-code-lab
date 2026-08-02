import { spawnSync } from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

export type BenchmarkOutputPaths = {
  rootDir: string;
  outputDir: string;
  jsonOutputPath: string;
  markdownOutputPath: string;
  manifestOutputPath: string;
  variant: string;
};

export type BenchmarkManifestInput = {
  lane: string;
  status: 'pass' | 'fail' | 'generated';
  summary?: unknown;
  notes?: string[];
};

export function resolveBenchmarkOutput(
  repoRoot: string,
  lane: string,
  defaultVariant: string
): BenchmarkOutputPaths {
  const rootDir = resolveBenchmarkRoot(repoRoot);
  const safeLane = sanitizePathSegment(lane, 'benchmark lane');
  const variant = sanitizePathSegment(process.env.GCL_BENCHMARK_VARIANT || defaultVariant, 'benchmark variant');
  const outputDir = path.join(rootDir, safeLane, variant);
  return {
    rootDir,
    outputDir,
    jsonOutputPath: path.join(outputDir, 'report.json'),
    markdownOutputPath: path.join(outputDir, 'report.md'),
    manifestOutputPath: path.join(outputDir, 'manifest.json'),
    variant
  };
}

export function resolveBenchmarkReportPath(repoRoot: string, lane: string, defaultVariant: string): string {
  return resolveBenchmarkOutput(repoRoot, lane, defaultVariant).jsonOutputPath;
}

export function writeBenchmarkManifest(
  repoRoot: string,
  output: BenchmarkOutputPaths,
  input: BenchmarkManifestInput
): void {
  const manifest = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    lane: input.lane,
    variant: output.variant,
    status: input.status,
    command: {
      argv: process.argv,
      npmLifecycleEvent: process.env.npm_lifecycle_event,
      cwd: process.cwd(),
      node: process.version
    },
    env: collectBenchmarkEnv(),
    git: collectGitInfo(repoRoot),
    outputs: {
      reportJson: path.relative(repoRoot, output.jsonOutputPath),
      reportMarkdown: path.relative(repoRoot, output.markdownOutputPath),
      manifest: path.relative(repoRoot, output.manifestOutputPath)
    },
    summary: input.summary,
    notes: input.notes || []
  };
  fs.writeFileSync(output.manifestOutputPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

function resolveBenchmarkRoot(repoRoot: string): string {
  const configuredRoot = process.env.GCL_BENCHMARK_ROOT;
  if (!configuredRoot) return path.join(repoRoot, 'benchmarks', 'release');
  return path.isAbsolute(configuredRoot)
    ? configuredRoot
    : path.join(repoRoot, configuredRoot);
}

function sanitizePathSegment(value: string, label: string): string {
  if (!value || value === '.' || value === '..') throw new Error(`${label} must be a non-empty path segment`);
  if (value.includes('/') || value.includes('\\') || value.includes('..')) {
    throw new Error(`${label} must not contain path separators or traversal`);
  }
  return value;
}

function collectBenchmarkEnv(): Record<string, string> {
  const entries = Object.entries(process.env)
    .filter(([key, value]) => value !== undefined && (key.startsWith('GCL_') || key === 'NODE_ENV'))
    .sort(([a], [b]) => a.localeCompare(b));
  return Object.fromEntries(entries) as Record<string, string>;
}

function collectGitInfo(repoRoot: string): {
  commit: string | null;
  statusShort: string | null;
  uncommittedDiffSha256: string | null;
} {
  return {
    commit: runGit(repoRoot, ['rev-parse', 'HEAD']),
    statusShort: normalizeGitStatusShort(runGit(repoRoot, ['status', '--short', '--', '.'])),
    // The commit alone does not identify the code that ran. Measured on this tree: 153
    // variants across 51 groups share a commit and an environment with a differently named
    // peer, and every one of those runs happened over a dirty working tree. `statusShort`
    // recorded which files were modified and nothing about how, so a run named adam-refine
    // and one named orthogonal-cover, minutes apart over different edits to the same files,
    // left byte-identical provenance. Hashing the diff makes those runs distinguishable
    // afterwards, which is what lets `verify-variant-distinctness.ts` tell a real difference
    // from a naming claim nothing backs.
    uncommittedDiffSha256: hashUncommittedDiff(repoRoot)
  };
}

function hashUncommittedDiff(repoRoot: string): string | null {
  const tracked = runGit(repoRoot, ['diff', 'HEAD']);
  if (tracked === null) return null;
  if (!tracked) return null; // A clean tree needs no digest: the commit already identifies it.
  return crypto.createHash('sha256').update(tracked).digest('hex');
}

function normalizeGitStatusShort(status: string | null): string | null {
  if (status === null) return null;
  return status
    .split('\n')
    .filter(line => line.trim() && !line.includes('__pycache__/'))
    .join('\n');
}

function runGit(repoRoot: string, args: string[]): string | null {
  const result = spawnSync('git', args, {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 5000,
    maxBuffer: 2 * 1024 * 1024
  });
  if (result.status !== 0) return null;
  return result.stdout.trim();
}
