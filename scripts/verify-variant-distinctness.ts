#!/usr/bin/env ts-node

/**
 * A variant name claims something. Its own artifact has to back that claim.
 *
 * `nn-live-blind-swiglu-layer-expansion` holds directories named adam-refine, gatedict-normaware,
 * cubic-gradient-field, structured-residual-smoke and dozens more. An audit reported that
 * nothing in those artifacts distinguished the method their names advertise. That turned out to
 * be too strong: measured here, 231 variants produce 29 distinct `config` blocks but 218 distinct
 * manifest environments, so for most of them the method is recorded - in the manifest, not in
 * report.json. `config.methodKnobs` now closes that gap for new runs.
 *
 * What the measurement does confirm is narrower and worse: 41 variants across 12 groups share an
 * identical environment *and* an identical git commit with at least one differently named peer.
 * For those, two names promise two methods and the whole recorded state of both runs is the same.
 * Either the difference lived somewhere nothing captured, or there was no difference.
 *
 * This check fails when two variants in a lane are indistinguishable that way and neither
 * declares it. As with the citation baseline, a known set is recorded rather than hidden, so the
 * count is visible and can only shrink; regenerate it deliberately with
 * GCL_VARIANT_DISTINCTNESS_BASELINE_WRITE=1.
 *
 * Read-only. Comparing names is not the job here - `expansion-fixednorm` and `expansion-scaleinit`
 * may well have been the same run twice - so this reports the collision and leaves the judgement
 * of which name was wrong to whoever knows what they meant to run.
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

const repoRoot = path.resolve(__dirname, '..');
const configuredRoot = process.env.GCL_BENCHMARK_ROOT;
const releaseRoot = configuredRoot
  ? (path.isAbsolute(configuredRoot) ? configuredRoot : path.join(repoRoot, configuredRoot))
  : path.join(repoRoot, 'benchmarks', 'release');
const baselinePath = path.join(releaseRoot, 'variant-distinctness-baseline.json');

type Group = { lane: string; fingerprint: string; variants: string[]; commit: string | null };

/** Keys that name the run rather than describe it, so they cannot make two runs distinct. */
const IDENTITY_KEYS = new Set(['GCL_BENCHMARK_VARIANT', 'GCL_BLIND_SWIGLU_SWEEP_VARIANT']);

function readJson(filePath: string): Record<string, unknown> | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function fingerprint(manifest: Record<string, unknown>): { key: string; commit: string | null } | null {
  const env = manifest.env;
  if (!env || typeof env !== 'object') return null;
  const describing = Object.entries(env as Record<string, string>)
    .filter(([key]) => key.startsWith('GCL_') && !IDENTITY_KEYS.has(key))
    .sort(([a], [b]) => a.localeCompare(b));
  const git = manifest.git as Record<string, unknown> | undefined;
  const commit = (git?.commit as string | undefined) ?? null;
  // Runs over a dirty tree share a commit while running different code. The diff digest is
  // what separates them; artifacts written before it existed have none, which is exactly why
  // their collisions cannot be resolved retroactively and sit in the baseline instead.
  const uncommittedDiffSha256 = (git?.uncommittedDiffSha256 as string | undefined) ?? null;
  return {
    key: crypto
      .createHash('sha256')
      .update(JSON.stringify({ describing, commit, uncommittedDiffSha256 }))
      .digest('hex')
      .slice(0, 16),
    commit
  };
}

/** Which level of a run this artifact is: the sweep aggregate, or one case inside it. */
function artifactKind(variantDir: string): 'aggregate' | 'case' | 'other' {
  const report = readJson(path.join(variantDir, 'report.json'));
  if (!report) return 'other';
  if (Array.isArray(report.cases)) return 'aggregate';
  if (report.solver) return 'case';
  return 'other';
}

/**
 * What the run was pointed at, as the report records it.
 *
 * A randomized sweep picks its layer and units from a seed, so two cases of one sweep share
 * every environment variable and differ only in what they were run against. That difference
 * is real and is written down - in the report's own summary - so it belongs in the key.
 * Without it the check reports mistral layer 1 and layer 19 as the same run under two names,
 * which is a defect in the check rather than in those artifacts.
 */
function subjectIdentity(variantDir: string): string {
  const report = readJson(path.join(variantDir, 'report.json'));
  if (!report) return '';
  const summary = report.summary as Record<string, unknown> | undefined;
  const config = report.config as Record<string, unknown> | undefined;
  return JSON.stringify({
    model: summary?.model ?? null,
    layer: summary?.layer ?? null,
    units: summary?.units ?? summary?.unit ?? null,
    atomCount: summary?.atomCount ?? null,
    outputChannel: summary?.outputChannel ?? null,
    caseSpec: config?.caseSpec ?? null,
    randomSeed: config?.randomSeed ?? null
  });
}

const groups = new Map<string, Group>();

for (const laneEntry of fs.readdirSync(releaseRoot, { withFileTypes: true })) {
  if (!laneEntry.isDirectory()) continue;
  const lanePath = path.join(releaseRoot, laneEntry.name);
  for (const variantEntry of fs.readdirSync(lanePath, { withFileTypes: true })) {
    if (!variantEntry.isDirectory()) continue;
    const variantDir = path.join(lanePath, variantEntry.name);
    const manifest = readJson(path.join(variantDir, 'manifest.json'));
    if (!manifest) continue;
    // A variant that says it is indistinguishable is disclosed, not hidden, so it is not a finding.
    if (manifest.variantDistinctness) continue;
    const print = fingerprint(manifest);
    if (!print) continue;
    // A sweep aggregate and the per-case report it summarises come from one run and share
    // everything this check compares. They are two levels of the same run, not two methods,
    // so the artifact kind joins the key rather than letting them collide with each other.
    const subjectKey = crypto.createHash('sha256').update(subjectIdentity(variantDir)).digest('hex').slice(0, 12);
    const groupKey = `${laneEntry.name}:${artifactKind(variantDir)}:${subjectKey}:${print.key}`;
    const existing = groups.get(groupKey);
    if (existing) existing.variants.push(variantEntry.name);
    else groups.set(groupKey, { lane: laneEntry.name, fingerprint: print.key, variants: [variantEntry.name], commit: print.commit });
  }
}

const collisions = [...groups.values()]
  .filter(group => group.variants.length > 1)
  .map(group => ({ ...group, variants: group.variants.sort() }))
  .sort((a, b) => b.variants.length - a.variants.length);

const collisionKey = (group: Group): string => `${group.lane}: ${group.variants.join(' == ')}`;
const collisionKeys = collisions.map(collisionKey).sort();

console.log(
  JSON.stringify(
    {
      variantsWithManifest: [...groups.values()].reduce((total, group) => total + group.variants.length, 0),
      collidingGroups: collisions.length,
      collidingVariants: collisions.reduce((total, group) => total + group.variants.length, 0)
    },
    null,
    2
  )
);

for (const group of collisions.slice(0, 20)) {
  console.log(`\n  ${group.lane} (${group.variants.length} variants, commit ${group.commit?.slice(0, 7) ?? 'unknown'}):`);
  for (const variant of group.variants) console.log(`    ${variant}`);
}

type Baseline = { recordedAt: string; note: string; collisions: string[] };

if (process.env.GCL_VARIANT_DISTINCTNESS_BASELINE_WRITE === '1') {
  const written: Baseline = {
    recordedAt: new Date().toISOString(),
    note:
      'Groups of differently named variants whose recorded environment and git commit are identical, so ' +
      'nothing on disk shows the difference their names claim. This list may only shrink.',
    collisions: collisionKeys
  };
  fs.writeFileSync(baselinePath, `${JSON.stringify(written, null, 2)}\n`);
  console.log(`\nbaseline written with ${collisionKeys.length} known collision group(s)`);
  process.exit(0);
}

const baseline: Baseline = fs.existsSync(baselinePath)
  ? (JSON.parse(fs.readFileSync(baselinePath, 'utf8')) as Baseline)
  : { recordedAt: '', note: '', collisions: [] };
const known = new Set(baseline.collisions);
const newlyColliding = collisionKeys.filter(key => !known.has(key));
const resolved = baseline.collisions.filter(key => !collisionKeys.includes(key));

console.log(
  JSON.stringify(
    { knownCollisions: baseline.collisions.length, collisionsNow: collisionKeys.length, newlyColliding: newlyColliding.length, resolvedSinceBaseline: resolved.length },
    null,
    2
  )
);

if (newlyColliding.length) {
  console.error(`\nvariant distinctness check failed: ${newlyColliding.length} new group(s) of variants that only differ by name`);
  for (const key of newlyColliding.slice(0, 20)) console.error(`  ${key}`);
  process.exit(1);
}
if (resolved.length) {
  console.error(
    `\nvariant distinctness check failed: ${resolved.length} baseline group(s) no longer collide. ` +
    'Re-record with GCL_VARIANT_DISTINCTNESS_BASELINE_WRITE=1 so the count reflects reality.'
  );
  process.exit(1);
}
