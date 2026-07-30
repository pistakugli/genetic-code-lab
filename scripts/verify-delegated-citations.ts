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
        const actual = sha256(target);
        citations.push({
          artifact,
          target: relativeTarget,
          state: actual === pinned ? 'verified' : 'drifted',
          detail: actual === pinned ? [] : [`pinned ${pinned} but target is ${actual}`]
        });
        continue;
      }

      const targetReport = readJson(target);
      const differences = compareSummaries(record.sweepSummary, targetReport?.summary);
      citations.push({
        artifact,
        target: relativeTarget,
        state: differences.length === 0 ? 'unpinned-matching' : 'unpinned-diverged',
        detail: differences
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

// A drifted digest is unambiguous: this artifact published numbers from bytes that are
// gone. That fails. Unpinned divergence is reported loudly but does not fail by default,
// because every artifact predating the digest is unpinned and failing on all of them would
// say nothing new; set GCL_CITATION_STRICT=1 to treat it as a failure too.
const strict = process.env.GCL_CITATION_STRICT === '1';
const failing = summary.drifted + summary.missingTarget + (strict ? summary.unpinnedDiverged : 0);
if (failing > 0) {
  console.error(`\ndelegated citation check failed: ${failing} citation(s) do not match what they cite`);
  process.exit(1);
}
