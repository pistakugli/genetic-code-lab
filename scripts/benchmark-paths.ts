import { spawnSync } from 'child_process';
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

function collectGitInfo(repoRoot: string): { commit: string | null; statusShort: string | null } {
  return {
    commit: runGit(repoRoot, ['rev-parse', 'HEAD']),
    statusShort: runGit(repoRoot, ['status', '--short', '--', '.'])
  };
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
