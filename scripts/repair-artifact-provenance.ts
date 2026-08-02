#!/usr/bin/env ts-node

/**
 * Resolve what `verify-artifact-provenance.ts` reports, one directory at a time.
 *
 * Two states, two different honest answers:
 *
 * Empty variant directories. Ten of them, all in nn-live-blind-swiglu-layer-expansion, named
 * after methods nobody can now run: dev-scale035-mixture, quadratic-start-smoke, and so on.
 * There is no evidence in them to preserve, but the fact that the variant was once created
 * is itself information, so the names are written to `removed-empty-variants.json` before the
 * directories go. Removing the directory and keeping the record is not the same as deleting
 * evidence; leaving an empty directory that reads as a lane that ran is worse than either.
 *
 * Reports with no manifest. Two diagnostics artifacts carry measurements with nothing saying
 * what produced them. The original command and environment are gone and will not be invented
 * here: `writeBenchmarkManifest` would stamp this script's own argv into the record, which
 * would be a lie about who ran what. What can be recovered is the commit that introduced the
 * file, so the manifest says exactly that and no more - status `unprovenanced`, provenance
 * reconstructed from git, and no summary, because a verdict was never recorded either.
 *
 * `--check` is the default and writes nothing. `--write` performs both actions.
 */

import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const repoRoot = path.resolve(__dirname, '..');
const releaseRoot = path.join(repoRoot, 'benchmarks', 'release');
const write = process.argv.includes('--write');
const removedRegistryPath = path.join(releaseRoot, 'removed-empty-variants.json');

type RemovedRegistry = {
  note: string;
  removed: Array<{ variant: string; removedAt: string }>;
};

function gitOrigin(relativePath: string): { commit: string; date: string; subject: string } | null {
  try {
    const line = execFileSync('git', ['log', '--diff-filter=A', '--format=%H%x1f%cI%x1f%s', '-1', '--', relativePath], {
      cwd: repoRoot,
      encoding: 'utf8'
    }).trim();
    if (!line) return null;
    const [commit, date, subject] = line.split('');
    return { commit, date, subject };
  } catch {
    return null;
  }
}

/** Same in-flight guard as the verifier: a run deletes its directory before rewriting it. */
const IN_FLIGHT_WINDOW_MS = 10 * 60 * 1000;
const isRecent = (dir: string): boolean => Date.now() - fs.statSync(dir).mtimeMs < IN_FLIGHT_WINDOW_MS;

function coveredByParentManifest(dir: string): boolean {
  let current = path.dirname(dir);
  while (current.startsWith(releaseRoot) && current !== releaseRoot) {
    if (fs.existsSync(path.join(current, 'manifest.json'))) return true;
    current = path.dirname(current);
  }
  return false;
}

const emptyDirectories: string[] = [];
const unmanifested: string[] = [];

function walk(dir: string): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  if (!entries.length) {
    if (!isRecent(dir)) emptyDirectories.push(dir);
    return;
  }
  const files = entries.filter(entry => entry.isFile());
  if (files.some(entry => entry.name === 'report.json') && !files.some(entry => entry.name === 'manifest.json')) {
    if (!coveredByParentManifest(dir)) unmanifested.push(dir);
  }
  for (const entry of entries.filter(item => item.isDirectory())) walk(path.join(dir, entry.name));
}

for (const lane of fs.readdirSync(releaseRoot, { withFileTypes: true })) {
  if (lane.isDirectory()) walk(path.join(releaseRoot, lane.name));
}

const removedAt = new Date().toISOString();
const registry: RemovedRegistry = fs.existsSync(removedRegistryPath)
  ? (JSON.parse(fs.readFileSync(removedRegistryPath, 'utf8')) as RemovedRegistry)
  : {
      note:
        'Variant directories that existed in the release tree with no files in them. The directory carried no ' +
        'evidence, but the fact that the variant was once created is recorded here rather than lost with it.',
      removed: []
    };

for (const dir of emptyDirectories) {
  const relative = path.relative(repoRoot, dir);
  console.log(`empty variant: ${relative}`);
  if (!write) continue;
  registry.removed.push({ variant: relative, removedAt });
  fs.rmdirSync(dir);
}

if (write && emptyDirectories.length) {
  registry.removed.sort((a, b) => a.variant.localeCompare(b.variant));
  fs.writeFileSync(removedRegistryPath, `${JSON.stringify(registry, null, 2)}\n`);
}

for (const dir of unmanifested) {
  const relative = path.relative(repoRoot, dir);
  const reportRelative = path.join(relative, 'report.json');
  const origin = gitOrigin(reportRelative);
  console.log(`unmanifested report: ${relative} (git origin: ${origin ? origin.commit.slice(0, 7) : 'unknown'})`);
  if (!write) continue;

  const segments = path.relative(releaseRoot, dir).split(path.sep);
  const manifest = {
    schemaVersion: 1,
    lane: segments[0],
    variant: segments.slice(1).join('/'),
    status: 'unprovenanced',
    // Deliberately no `command` block. The run that wrote report.json did not record one,
    // and this script's own argv is not that command.
    provenance: {
      state: origin ? 'reconstructed-from-git' : 'unknown',
      note:
        'This artifact was published without a manifest, so the command, environment and run time ' +
        'that produced it were never recorded and cannot be recovered. The commit below is the one ' +
        'that first introduced the file, which is the only provenance that still exists.',
      introducedByCommit: origin?.commit ?? null,
      introducedAt: origin?.date ?? null,
      introducedBySubject: origin?.subject ?? null,
      recordedAt: removedAt
    },
    outputs: { reportJson: path.join(relative, 'report.json') },
    // No summary: a verdict was never recorded for this run, and inventing one here would be
    // the same defect as inventing the command.
    summary: null,
    notes: [
      'Diagnostic measurement, not a gated lane result.',
      'Rerunning the diagnostic under a manifest-writing entrypoint is the only way to give it real provenance.'
    ]
  };
  fs.writeFileSync(path.join(dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

console.log(
  JSON.stringify(
    {
      mode: write ? 'write' : 'check',
      emptyDirectories: emptyDirectories.length,
      unmanifestedReports: unmanifested.length
    },
    null,
    2
  )
);
