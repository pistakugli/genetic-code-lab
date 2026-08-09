#!/usr/bin/env ts-node

// The whole ladder, checked against artifacts that already exist.
//
// The roadmap says this lane is release-gated from 16 to 512 units. It was gated at 64, and
// only at 64: the other five variants had a command and an artifact and nothing that would
// notice if either stopped agreeing with the code. That is the same shape as a released
// variant with no command - a claim whose surface nobody was watching.
//
// Re-running the model six times is an hour that a release gate cannot spend, so this runs the
// lane in its no-regeneration mode, where each variant validates the committed artifact against
// the current solver and its own thresholds rather than producing a new one. That is what a
// gate is for: the artifacts are the claim, and this asks whether they still hold.

import { spawnSync } from 'child_process';
import path from 'path';

const repoRoot = path.resolve(__dirname, '..');
const variants = ['16unit', '32unit', '64unit', '128unit', '256unit', '512unit'];
const failed: string[] = [];

for (const variant of variants) {
  const started = Date.now();
  const result = spawnSync(
    'npx',
    ['ts-node', 'scripts/benchmark-nn-live-jacobian-boundary-recovery.ts'],
    {
      cwd: repoRoot,
      encoding: 'utf8',
      env: {
        ...process.env,
        GCL_JACOBIAN_VARIANT: variant,
        GCL_JACOBIAN_REGENERATE: '0',
        GCL_PYTHON: process.env.GCL_PYTHON ?? '.gcl-python-gpu/bin/python',
      },
    },
  );
  const seconds = ((Date.now() - started) / 1000).toFixed(1);
  if (result.status !== 0) {
    failed.push(variant);
    // The lane's own failure list is the useful part; printing it here keeps the reason with
    // the variant rather than in a log the caller has to go find.
    process.stderr.write(`${variant} failed after ${seconds}s\n${result.stdout ?? ''}${result.stderr ?? ''}\n`);
  } else {
    process.stdout.write(`${variant} ok in ${seconds}s\n`);
  }
}

console.log(JSON.stringify({ variants: variants.length, failed }, null, 2));
if (failed.length) process.exit(1);
