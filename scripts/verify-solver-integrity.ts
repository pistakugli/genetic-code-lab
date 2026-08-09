#!/usr/bin/env ts-node

// Four checks against four classes of defect, each of which reached a commit during a single
// day of work on the Jacobian route and none of which any existing verifier could see. The
// instances are repaired; this is here because repairing instances of a class you have no
// check for is how you get the same class again next week.
//
//   1. A failure swallowed by `except Exception: pass`, so a broken card looks like a slow one.
//   2. An artifact whose generator has moved on, still reading `pass`. This lane's manifests
//      went stale within hours of the lane being written.
//   3. An access audit made of literals the tool writes about itself, checked by a gate that
//      confirms it spelled them correctly.
//   4. A run that writes outside its own lane. One diagnostic pass rewrote 169 release
//      artifacts in two unrelated lanes and nothing noticed for a day.

import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

const repoRoot = path.resolve(__dirname, '..');
const violations: string[] = [];

function walk(dir: string, match: (name: string) => boolean): string[] {
  if (!fs.existsSync(dir)) return [];
  const found: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) found.push(...walk(full, match));
    else if (match(entry.name)) found.push(full);
  }
  return found;
}

// 1. Swallowed failures. A comment describing the anti-pattern is fine; code performing it is
// not, and the distinction matters because the repair deliberately left its explanation behind.
//
// Enforced on the files that have been through this audit, counted elsewhere. The core
// decompiler has four such sites and they are real, but they sit in thirteen thousand lines
// that today's verification does not cover: changing what an exception handler catches there
// would let failures escape in paths I cannot re-measure now, which trades a known silence for
// an unknown break. They are reported so the number is visible.
const auditedSolvers = new Set([
  'tools/llm_reader/jacobian_boundary_recovery.py',
  // Added once its four sites were repaired rather than counted. The deferral was that
  // narrowing handlers in a file nothing here covers trades a known silence for an unknown
  // break - but this file stopped being uncovered the day a real defect was found and fixed
  // in it, so the reason expired before the entry did.
  'tools/llm_reader/blind_swiglu_oracle_decompiler.py',
]);
let unauditedSwallowedFailures = 0;
for (const file of walk(path.join(repoRoot, 'tools/llm_reader'), (name) => name.endsWith('.py'))) {
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  lines.forEach((line, index) => {
    if (!line.includes('except Exception') || line.trim().startsWith('#')) return;
    // A bare re-raise or a recorded reason is a handled failure, not a swallowed one.
    const body = lines.slice(index + 1, index + 4).join('\n');
    if (/\braise\b|notes\.append|_record_fallback|logger|print\(/.test(body)) return;
    const relative = path.relative(repoRoot, file);
    if (!auditedSolvers.has(relative)) {
      unauditedSwallowedFailures += 1;
      return;
    }
    violations.push(`${relative}:${index + 1} swallows a failure`);
  });
}

// 2. Artifacts that outlived the code that made them. Only lanes that opted in by recording a
// generator hash are checked, so this grows lane by lane rather than failing everything at once.
const generators: Record<string, string> = {
  'nn-live-jacobian-boundary-recovery': 'tools/llm_reader/jacobian_boundary_recovery.py',
};
for (const [lane, generator] of Object.entries(generators)) {
  const current = crypto
    .createHash('sha256')
    .update(fs.readFileSync(path.join(repoRoot, generator)))
    .digest('hex');
  for (const manifest of walk(path.join(repoRoot, 'benchmarks/release', lane), (name) => name === 'manifest.json')) {
    const parsed = JSON.parse(fs.readFileSync(manifest, 'utf8')) as { summary?: Record<string, unknown> };
    const recorded = parsed.summary?.solverSha256;
    if (typeof recorded !== 'string') {
      violations.push(`${path.relative(repoRoot, manifest)} records no generator hash`);
    } else if (recorded !== current) {
      violations.push(`${path.relative(repoRoot, manifest)} was produced by a different ${generator}`);
    }
  }
}

// 3. Access audits that assert rather than evidence. A report claiming `oracle-only` has to
// carry something a reader could check: which stage unlocked the post-hoc truth, at which
// query count, against the count the solver spent.
//
// Only lanes that have adopted the evidence fields are held to it. Every other lane in the
// repository claims `oracle-only` on its own say-so, which is the same defect and is not
// today's work to fix - and a verifier that fails on a hundred pre-existing reports is a
// verifier somebody disables, which is worse than not having one. Those are counted instead,
// so the number is visible and can be worked down lane by lane.
const evidenceLanes = new Set(['nn-live-jacobian-boundary-recovery']);
let unevidencedLegacyReports = 0;
for (const report of walk(path.join(repoRoot, 'benchmarks/release'), (name) => name === 'report.json')) {
  const parsed = JSON.parse(fs.readFileSync(report, 'utf8')) as { accessAudit?: Record<string, unknown> };
  const audit = parsed.accessAudit;
  if (!audit || audit.solverAccess !== 'oracle-only') continue;
  const lane = path.relative(repoRoot, report).split('/')[2];
  // Two tiers, because one rule for both would either let the weak case through or demand of
  // it a comparison it cannot honestly make. A lane that keeps its own call count can be held
  // to two independently maintained counters agreeing, which is what rules out a query outside
  // the accounted path. A lane with only the client's counter cannot: filling both sides from
  // `oracle.query_count` compares a number with itself. What that lane can still show is that
  // the truth was locked until a named stage and at which query count it opened, so that is
  // what is required of it - real evidence, and not dressed up as the stronger kind.
  const strong = audit.solverOracleCalls !== undefined;
  const evidence = strong
    ? ['truthUnlockStage', 'truthUnlockQueryCount', 'solverOracleCalls']
    : ['truthUnlockStage', 'truthUnlockQueryCount', 'queryCountAtReport'];
  const missing = evidence.filter((key) => audit[key] === undefined || audit[key] === null);
  if (!evidenceLanes.has(lane)) {
    if (missing.length) unevidencedLegacyReports += 1;
    continue;
  }
  if (missing.length) {
    violations.push(`${path.relative(repoRoot, report)} claims oracle-only but is missing ${missing.join(', ')}`);
  } else if (strong && audit.truthUnlockQueryCount !== audit.solverOracleCalls) {
    violations.push(`${path.relative(repoRoot, report)} unlocked truth at a different count than the solver spent`);
  } else if (strong && audit.counterAgreementChecked === false) {
    // Carrying the field while declaring the comparison unavailable is a report claiming the
    // strong form on a number it just said was not independent.
    violations.push(`${path.relative(repoRoot, report)} reports solverOracleCalls but marks the counters as not independently checked`);
  }
}

// 4. Uncommitted writes outside the lane under test. Cheap and blunt: if the working tree
// holds modified release artifacts, say which lanes, because that is the state in which every
// measurement taken afterwards carries a polluted provenance stamp.
const dirty = spawnSync('git', ['status', '--porcelain', 'benchmarks/release'], { cwd: repoRoot, encoding: 'utf8' });
const touchedLanes = new Set(
  (dirty.stdout || '')
    .split('\n')
    .filter(Boolean)
    .map((line) => line.trim().split(/\s+/).pop() ?? '')
    .map((file) => file.split('/')[2])
    .filter(Boolean),
);
if (touchedLanes.size) {
  violations.push(`uncommitted release artifacts in: ${[...touchedLanes].join(', ')}`);
}

// 5. Artifact paths cited in prose. Every other claim surface here has a checker - manifests
// carry generator hashes, reports carry access evidence, delegated citations are compared
// field by field - and the documentation has none, which makes it the one place a claim can
// rot indefinitely. It did: the roadmap cited an artifact that had been deleted for being
// unreproducible, and nothing noticed.
const citedPaths = new Set<string>();
for (const doc of walk(path.join(repoRoot, 'docs'), (name) => name.endsWith('.md'))) {
  const body = fs.readFileSync(doc, 'utf8');
  for (const match of body.matchAll(/benchmarks\/release\/[A-Za-z0-9._/-]+/g)) {
    const cited = match[0].replace(/[.,`)]+$/, '');
    if (!fs.existsSync(path.join(repoRoot, cited))) {
      citedPaths.add(`${path.relative(repoRoot, doc)} cites a path that does not exist: ${cited}`);
    }
  }
}
for (const missing of citedPaths) violations.push(missing);

// 6. Released variants with no command that makes them. The roadmap's own rule is that a lane
// becomes a release claim only with a command, artifact, gate and boundary - and the 512-unit
// variant had been shipped, cited and gated with no npm script behind it, found by accident
// while writing a regeneration loop that asked for a script that was not there. An artifact
// nobody can re-run is a number, not a result.
const jacobianVariants = new Set<string>();
const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf8')) as {
  scripts?: Record<string, string>;
};
for (const command of Object.values(pkg.scripts ?? {})) {
  const match = /GCL_JACOBIAN_VARIANT=(\w+)/.exec(command);
  if (match) jacobianVariants.add(match[1]);
}
const jacobianLane = path.join(repoRoot, 'benchmarks/release/nn-live-jacobian-boundary-recovery');
if (fs.existsSync(jacobianLane)) {
  for (const entry of fs.readdirSync(jacobianLane, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const named = /layer15-(\w+)-[a-z]+-gate$/.exec(entry.name);
    if (named && !jacobianVariants.has(named[1])) {
      violations.push(`benchmarks/release/nn-live-jacobian-boundary-recovery/${entry.name} has no npm script that produces it`);
    }
  }
}

console.log(JSON.stringify({
  swallowedFailures: violations.filter((v) => v.includes('swallows')).length,
  unauditedSwallowedFailures,
  staleArtifacts: violations.filter((v) => v.includes('different') || v.includes('no generator')).length,
  unevidencedAccessClaims: violations.filter((v) => v.includes('oracle-only') || v.includes('unlocked truth')).length,
  // Not a failure: lanes that have not adopted the evidence fields yet. Reported so the
  // number is visible rather than assumed to be zero.
  unevidencedLegacyReports,
  strayLaneWrites: violations.filter((v) => v.startsWith('uncommitted')).length,
  deadDocCitations: violations.filter((v) => v.includes('cites a path that does not exist')).length,
  unreproducibleVariants: violations.filter((v) => v.includes('no npm script that produces it')).length,
  violations,
}, null, 2));
if (violations.length) process.exit(1);
