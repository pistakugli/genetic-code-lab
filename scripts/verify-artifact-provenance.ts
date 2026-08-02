#!/usr/bin/env ts-node

/**
 * Every published artifact must say where it came from, and every directory in the release
 * tree must be an artifact.
 *
 * An audit found 10 variant directories that were completely empty and 2 that carried a
 * report.json with no manifest.json and no report.md. An empty directory reads as a lane
 * that was run; an unmanifested report reads as evidence while recording neither the command
 * that produced it, nor the environment, nor when. Neither state was detectable, because
 * every existing check starts from report.json and asks about its contents.
 *
 * Two rules, both about the shape of the tree rather than the numbers in it:
 *
 *   - a directory under a lane holds either artifacts or further directories, never nothing
 *   - a report.json is accompanied by a manifest.json, unless it is a per-case report inside
 *     a variant whose own manifest covers the run that wrote it
 *
 * Read-only.
 */

import fs from 'fs';
import path from 'path';

type Violation = { kind: 'empty-directory' | 'missing-manifest'; artifact: string; detail: string };

const repoRoot = path.resolve(__dirname, '..');
// Honours GCL_BENCHMARK_ROOT like the writers do, which is also what lets this be tested
// against a scratch tree instead of whatever the real release tree happens to hold.
const configuredRoot = process.env.GCL_BENCHMARK_ROOT;
const releaseRoot = configuredRoot
  ? (path.isAbsolute(configuredRoot) ? configuredRoot : path.join(repoRoot, configuredRoot))
  : path.join(repoRoot, 'benchmarks', 'release');

/**
 * A run in flight deletes its output directory before rewriting it, so a scan racing one
 * sees an empty directory that is neither abandoned nor a defect. Directories touched in
 * the last few minutes are reported separately rather than counted against the tree.
 */
const IN_FLIGHT_WINDOW_MS = 10 * 60 * 1000;

function isRecent(dir: string): boolean {
  try {
    return Date.now() - fs.statSync(dir).mtimeMs < IN_FLIGHT_WINDOW_MS;
  } catch {
    return false;
  }
}

/** Does a parent variant directory already account for this per-case report? */
function coveredByParentManifest(dir: string): boolean {
  let current = path.dirname(dir);
  while (current.startsWith(releaseRoot) && current !== releaseRoot) {
    if (fs.existsSync(path.join(current, 'manifest.json'))) return true;
    current = path.dirname(current);
  }
  return false;
}

const violations: Violation[] = [];
const inFlight: string[] = [];
let directoriesChecked = 0;

function walk(dir: string): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const subdirectories = entries.filter(entry => entry.isDirectory());
  const files = entries.filter(entry => entry.isFile());
  directoriesChecked++;

  if (!entries.length) {
    const relative = path.relative(repoRoot, dir);
    if (isRecent(dir)) inFlight.push(relative);
    else violations.push({ kind: 'empty-directory', artifact: relative, detail: 'directory holds no files and no subdirectories' });
    return;
  }

  if (files.some(entry => entry.name === 'report.json') && !files.some(entry => entry.name === 'manifest.json')) {
    if (!coveredByParentManifest(dir)) {
      violations.push({
        kind: 'missing-manifest',
        artifact: path.relative(repoRoot, dir),
        detail: 'report.json with no manifest.json here or above it, so the run that wrote it is unrecorded'
      });
    }
  }

  for (const entry of subdirectories) walk(path.join(dir, entry.name));
}

for (const lane of fs.readdirSync(releaseRoot, { withFileTypes: true })) {
  if (lane.isDirectory()) walk(path.join(releaseRoot, lane.name));
}

console.log(
  JSON.stringify(
    {
      directoriesChecked,
      violations: violations.length,
      emptyDirectories: violations.filter(item => item.kind === 'empty-directory').length,
      missingManifest: violations.filter(item => item.kind === 'missing-manifest').length,
      skippedAsInFlight: inFlight.length
    },
    null,
    2
  )
);

for (const item of inFlight) console.log(`  in flight, not counted: ${item}`);

if (!violations.length) process.exit(0);

console.error(`\nartifact provenance check failed: ${violations.length} directory/directories`);
for (const item of violations) {
  console.error(`  [${item.kind}] ${item.artifact}`);
  console.error(`     ${item.detail}`);
}
process.exit(1);
