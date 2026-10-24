#!/usr/bin/env ts-node

/**
 * Check every delegated citation in the release tree against what it cites today.
 *
 * Some lanes do not carry their own evidence. A layer-expansion variant runs the
 * oracle-decompiler sweep and then copies that sweep's summary into its own report,
 * recording the source as `sweepReportPath`. The target is a normal, mutable file, so a
 * later run of a different variant overwrites it and every earlier citation silently
 * starts pointing at numbers that were never the ones it published.
 *
 * An audit found 209 of 241 such citations no longer matching, and two variants publishing
 * `pass` against a witness that had since been rewritten to `fail`. Nothing detected it,
 * because a citation to a path proves only that the path exists.
 *
 * Artifacts written from now on record `sweepReportSha256`, the digest of the exact bytes
 * that were read, which makes drift provable. This verifier reports three states:
 *
 *   verified  - the citation carries a digest and the target still matches it
 *   drifted   - the citation carries a digest and the target no longer matches
 *   unpinned  - the citation predates the digest, so it can only be compared by value
 *
 * For unpinned citations the published summary is compared field by field against the
 * target's current summary, which is weaker than a digest but is what is available for
 * artifacts already on disk. That comparison is what surfaced the two `pass`-against-`fail`
 * cases, so it is worth doing even though it cannot be made exact retroactively.
 *
 * Read-only. It never rewrites an artifact; a stale citation is a fact to report, not
 * something to paper over by regenerating the file underneath it.
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

type CitationState = 'verified' | 'drifted' | 'unpinned-matching' | 'unpinned-diverged' | 'missing-target';

type Citation = {
  artifact: string;
  target: string;
  state: CitationState;
  detail: string[];
  labelled?: boolean;
};

const repoRoot = path.resolve(__dirname, '..');
const releaseRoot = path.join(repoRoot, 'benchmarks', 'release');

/** Summary fields worth comparing for an unpinned citation: the ones a reader would quote. */
const COMPARED_FIELDS = [
  'pass',
  'caseCount',
  'passedCases',
  'failedCases',
  'minR2',
  'maxMae',
  'maxWorstError',
  'minSubspacePrincipalCosine',
  'minPairedAtomJointCosine',
  'maxRawOracleCallsPerCase',
  'totalRawOracleCalls',
  'accessInvariant'
] as const;

function readJson(filePath: string): Record<string, unknown> | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function sha256(filePath: string): string {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function walkReports(dir: string, found: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkReports(full, found);
    else if (entry.name === 'report.json') found.push(full);
  }
  return found;
}

function compareSummaries(published: unknown, current: unknown): string[] {
  if (!published || typeof published !== 'object') return ['published summary is missing'];
  if (!current || typeof current !== 'object') return ['target summary is missing'];
  const a = published as Record<string, unknown>;
  const b = current as Record<string, unknown>;
  const differences: string[] = [];
  for (const field of COMPARED_FIELDS) {
    if (!(field in a) && !(field in b)) continue;
    const left = a[field];
    const right = b[field];
    if (JSON.stringify(left) !== JSON.stringify(right)) {
      differences.push(`${field}: published ${JSON.stringify(left)} vs current ${JSON.stringify(right)}`);
    }
  }
  return differences;
}

function collectCitations(): Citation[] {
  const citations: Citation[] = [];
  for (const reportPath of walkReports(releaseRoot)) {
    const report = readJson(reportPath);
    if (!report) continue;
    const cases = Array.isArray(report.cases) ? report.cases : [];
    for (const entry of cases) {
      if (!entry || typeof entry !== 'object') continue;
      const record = entry as Record<string, unknown>;
      const relativeTarget = record.sweepReportPath;
      if (typeof relativeTarget !== 'string' || !relativeTarget) continue;

      const artifact = path.relative(repoRoot, reportPath);
      const target = path.join(repoRoot, relativeTarget);
      if (!fs.existsSync(target)) {
        citations.push({ artifact, target: relativeTarget, state: 'missing-target', detail: ['target file does not exist'] });
        continue;
      }

      const pinned = record.sweepReportSha256;
      if (typeof pinned === 'string' && pinned.length === 64) {
        let actual = sha256(target);
        let verifiedByCopy = false;
        if (actual !== pinned && typeof record.citedSweepReportCopy === 'string') {
          const copyPath = path.join(path.dirname(reportPath), record.citedSweepReportCopy);
          if (fs.existsSync(copyPath) && sha256(copyPath) === pinned) {
            actual = pinned;
            verifiedByCopy = true;
          }
        }
        citations.push({
          artifact,
          target: relativeTarget,
          state: actual === pinned ? 'verified' : 'drifted',
          detail: actual === pinned
            ? (verifiedByCopy ? ['verified against local immutable copy'] : [])
            : [`pinned ${pinned} but target is ${actual}`]
        });
        continue;
      }

      const targetReport = readJson(target);
      const differences = compareSummaries(record.sweepSummary, targetReport?.summary);
      citations.push({
        artifact,
        target: relativeTarget,
        state: differences.length === 0 ? 'unpinned-matching' : 'unpinned-diverged',
        detail: differences,
        // A broken citation that the artifact itself admits to is still broken and still
        // counted below. The label only decides whether a reader of that artifact is told;
        // before it existed, only this verifier knew.
        labelled: typeof record.citationEvidence === 'object' && record.citationEvidence !== null
      });
    }
  }
  return citations;
}

const citations = collectCitations();
const byState = (state: CitationState): Citation[] => citations.filter(citation => citation.state === state);

const summary = {
  citationCount: citations.length,
  verified: byState('verified').length,
  drifted: byState('drifted').length,
  unpinnedMatching: byState('unpinned-matching').length,
  unpinnedDiverged: byState('unpinned-diverged').length,
  missingTarget: byState('missing-target').length
};

console.log(JSON.stringify(summary, null, 2));

for (const state of ['drifted', 'missing-target', 'unpinned-diverged'] as CitationState[]) {
  const rows = byState(state);
  if (!rows.length) continue;
  console.log(`\n${state} (${rows.length}):`);
  for (const row of rows.slice(0, 40)) {
    console.log(`  ${row.artifact}`);
    console.log(`    -> ${row.target}`);
    for (const line of row.detail.slice(0, 3)) console.log(`       ${line}`);
  }
  if (rows.length > 40) console.log(`  ... and ${rows.length - 40} more`);
}

/**
 * Verdict, as a ratchet against a recorded baseline.
 *
 * An earlier version of this file counted only drifted digests and missing targets, and
 * excluded unpinned divergence unless an opt-in env var was set. Since nothing was pinned
 * yet, that meant the check detected 210 broken citations and then reported success, which
 * is the same defect it exists to catch.
 *
 * Failing outright on all 210 would be equally useless: they are already-published
 * artifacts whose repair needs GPU reruns, so a permanently red check would just get
 * ignored. Instead the known-broken set is recorded explicitly in
 * `benchmarks/release/delegated-citation-baseline.json` and this check fails when
 *
 *   - a citation is broken and is not in the baseline, i.e. something new broke, or
 *   - the baseline does not shrink to match reality, i.e. a repair was made and not banked.
 *
 * So the count is visible, bounded, and can only go down. Regenerate the baseline
 * deliberately with GCL_CITATION_BASELINE_WRITE=1 after a repair; there is no flag that
 * makes a broken citation stop counting.
 */
type Baseline = { recordedAt: string; note: string; broken: string[] };

const baselinePath = path.join(releaseRoot, 'delegated-citation-baseline.json');
const brokenKey = (citation: Citation): string => `${citation.artifact} -> ${citation.target}`;
const brokenNow = citations
  .filter(citation => citation.state === 'drifted' || citation.state === 'missing-target' || citation.state === 'unpinned-diverged')
  .map(brokenKey)
  .sort();

if (process.env.GCL_CITATION_BASELINE_WRITE === '1') {
  const written: Baseline = {
    recordedAt: new Date().toISOString(),
    note: 'Citations known to be broken. This list may only shrink; verify-delegated-citations.ts fails on anything not in it.',
    broken: brokenNow
  };
  fs.writeFileSync(baselinePath, `${JSON.stringify(written, null, 2)}\n`);
  console.log(`\nbaseline written with ${brokenNow.length} known-broken citation(s): ${path.relative(repoRoot, baselinePath)}`);
  process.exit(0);
}

const baseline: Baseline = fs.existsSync(baselinePath)
  ? (JSON.parse(fs.readFileSync(baselinePath, 'utf8')) as Baseline)
  : { recordedAt: '', note: '', broken: [] };
const baselineSet = new Set(baseline.broken);
const newlyBroken = brokenNow.filter(key => !baselineSet.has(key));
const repaired = baseline.broken.filter(key => !brokenNow.includes(key));

console.log(JSON.stringify({
  knownBroken: baseline.broken.length,
  brokenNow: brokenNow.length,
  newlyBroken: newlyBroken.length,
  repairedSinceBaseline: repaired.length
}, null, 2));

if (newlyBroken.length) {
  console.error(`\ndelegated citation check failed: ${newlyBroken.length} citation(s) broke and are not in the baseline`);
  for (const key of newlyBroken.slice(0, 20)) console.error(`  ${key}`);
  process.exit(1);
}

/**
 * The baseline bounds how many citations may be broken. It says nothing about whether the
 * artifacts making them admit it, and for a year that was the whole problem: the count lived
 * here, while a reader opening one of those reports saw a citation to a path that exists and
 * read it as evidence. A broken citation must now carry `citationEvidence` in its own case
 * record, written by `label-unverifiable-citations.ts`.
 *
 * This adds a requirement; it does not remove one. The label does not shrink `brokenNow`,
 * is not consulted by the baseline comparison above, and cannot turn a broken citation into
 * a passing one. Only a rerun that produces a digest-pinned citation does that.
 */
const unlabelledBroken = citations.filter(
  citation =>
    (citation.state === 'drifted' || citation.state === 'unpinned-diverged') && citation.labelled !== true
);
if (unlabelledBroken.length) {
  console.error(
    `\ndelegated citation check failed: ${unlabelledBroken.length} broken citation(s) carry no citationEvidence, ` +
    'so the artifacts making them do not disclose that their source cannot be verified. ' +
    'Run `npm run label:unverifiable-citations` to write the record, or rerun the lane to earn a pinned citation.'
  );
  for (const citation of unlabelledBroken.slice(0, 20)) console.error(`  ${citation.artifact} -> ${citation.target}`);
  process.exit(1);
}
if (repaired.length) {
  console.error(
    `\ndelegated citation check failed: ${repaired.length} citation(s) are no longer broken but are still listed as known-broken. ` +
    `Re-record the baseline with GCL_CITATION_BASELINE_WRITE=1 so the count reflects reality.`
  );
  process.exit(1);
}
