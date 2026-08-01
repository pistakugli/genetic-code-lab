#!/usr/bin/env ts-node

/**
 * Fail when a release artifact disagrees with itself about whether it passed.
 *
 * Every lane writes two files side by side: `report.json`, produced by the tool that did
 * the work, and `manifest.json`, produced by the wrapper, which embeds a copy of the
 * report's summary and a `status` field derived from it. Three artifacts in
 * `nn-live-blind-swiglu-oracle-decompiler` carry `manifest.status = "fail"` while both
 * `manifest.summary.pass` and `report.summary.pass` read true, so the manifest contradicts
 * the summary it is carrying. Their bodies are honest, 14 of 16 paired atoms and a minimum
 * joint cosine of 0.295, but anyone reading `report.md` alone sees a failing variant
 * headlined as passing.
 *
 * Three separate agreements are checked, because they fail for different reasons:
 *
 *   - manifest.status against manifest.summary.pass, which is a contradiction inside one
 *     file and can only be a writer defect,
 *   - manifest.summary.pass against report.summary.pass, which means the two files came
 *     from different runs,
 *   - report.md's headline `pass:` line against report.json, which means the prose and the
 *     data diverged.
 *
 * Read-only. A disagreement is reported, never silently rewritten: which of the two sides is
 * right is a question about what actually ran, and this script cannot answer it.
 */

import fs from 'fs';
import path from 'path';

type Disagreement = {
  variant: string;
  kind: 'manifest-status-vs-manifest-summary' | 'manifest-summary-vs-report' | 'markdown-vs-report';
  detail: string;
};

const repoRoot = path.resolve(__dirname, '..');
const releaseRoot = path.join(repoRoot, 'benchmarks', 'release');

function readJson(filePath: string): Record<string, unknown> | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function summaryPass(container: Record<string, unknown> | null): boolean | null {
  if (!container) return null;
  const summary = container.summary;
  if (!summary || typeof summary !== 'object') return null;
  const pass = (summary as Record<string, unknown>).pass;
  return typeof pass === 'boolean' ? pass : null;
}

/** The `- pass: \`true\`` bullet that report.md files lead with, when present. */
function markdownPass(markdownPath: string): boolean | null {
  if (!fs.existsSync(markdownPath)) return null;
  const match = fs.readFileSync(markdownPath, 'utf8').match(/^-\s*pass:\s*`?(true|false)`?/im);
  // The regex is case-insensitive but the comparison must be too: several lanes render the
  // Python boolean, so the captured text is `True`, and comparing that to 'true' directly
  // reported every one of them as a failing markdown against a passing report. That was a
  // defect in this check, not in those artifacts.
  return match ? match[1].toLowerCase() === 'true' : null;
}

function walkVariants(dir: string, found: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const full = path.join(dir, entry.name);
    if (fs.existsSync(path.join(full, 'report.json'))) found.push(full);
    else walkVariants(full, found);
  }
  return found;
}

const disagreements: Disagreement[] = [];
let checked = 0;

for (const variantDir of walkVariants(releaseRoot)) {
  const variant = path.relative(repoRoot, variantDir);
  const report = readJson(path.join(variantDir, 'report.json'));
  const manifest = readJson(path.join(variantDir, 'manifest.json'));
  const reportPass = summaryPass(report);
  checked++;

  if (manifest) {
    const status = manifest.status;
    const manifestPass = summaryPass(manifest);
    if (typeof status === 'string' && manifestPass !== null) {
      // `generated` is a deliberate third state for runs that completed below the lane's
      // release bars, so it is not a contradiction.
      const statusSaysPass = status === 'pass';
      if (status !== 'generated' && statusSaysPass !== manifestPass) {
        disagreements.push({
          variant,
          kind: 'manifest-status-vs-manifest-summary',
          detail: `manifest.status=${status} but manifest.summary.pass=${manifestPass}`
        });
      }
    }
    if (manifestPass !== null && reportPass !== null && manifestPass !== reportPass) {
      disagreements.push({
        variant,
        kind: 'manifest-summary-vs-report',
        detail: `manifest.summary.pass=${manifestPass} but report.summary.pass=${reportPass}`
      });
    }
  }

  const mdPass = markdownPass(path.join(variantDir, 'report.md'));
  if (mdPass !== null && reportPass !== null && mdPass !== reportPass) {
    disagreements.push({
      variant,
      kind: 'markdown-vs-report',
      detail: `report.md says pass:${mdPass} but report.json says ${reportPass}`
    });
  }
}

const byKind = (kind: Disagreement['kind']): number => disagreements.filter(item => item.kind === kind).length;

console.log(JSON.stringify({
  variantsChecked: checked,
  disagreements: disagreements.length,
  manifestStatusVsManifestSummary: byKind('manifest-status-vs-manifest-summary'),
  manifestSummaryVsReport: byKind('manifest-summary-vs-report'),
  markdownVsReport: byKind('markdown-vs-report')
}, null, 2));

if (disagreements.length) {
  console.error(`\nartifact verdict consistency failed: ${disagreements.length} artifact(s) disagree with themselves`);
  for (const item of disagreements) {
    console.error(`  ${item.variant}`);
    console.error(`    ${item.kind}: ${item.detail}`);
  }
  process.exit(1);
}
