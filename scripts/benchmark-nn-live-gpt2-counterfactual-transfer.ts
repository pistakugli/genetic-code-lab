#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(
  repoRoot,
  'nn-live-gpt2-counterfactual-circuit-transfer',
  'pilot'
);
if (![
  'pilot',
  'pilot-verified',
  'pilot-calibrated',
  'pilot-final',
  'pilot-archive-corrected',
  'all-pairs'
].includes(output.variant)) {
  throw new Error('GCL_BENCHMARK_VARIANT must be a pilot variant or all-pairs');
}
const protocolVariant = output.variant.startsWith('pilot') ? 'pilot' : 'all-pairs';
const python = process.env.GCL_PYTHON
  || path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
const device = process.env.GCL_GPT2_COUNTERFACTUAL_DEVICE || 'cuda';
if (!['cuda', 'cpu'].includes(device)) {
  throw new Error('GCL_GPT2_COUNTERFACTUAL_DEVICE must be cuda or cpu');
}
const sourceReport = path.join(
  repoRoot,
  'benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder',
  'gpt2-50-facts-ordered-residual-path',
  'report.json'
);
const runner = path.join(repoRoot, 'tools', 'llm_reader', 'gpt2_counterfactual_transfer.py');
const finalizer = path.join(repoRoot, 'tools', 'llm_reader', 'finalize_gpt2_counterfactual_transfer.py');
if (!fs.existsSync(python)) throw new Error(`Missing configured GCL Python interpreter: ${python}`);
if (!fs.existsSync(sourceReport)) throw new Error(`Missing frozen source artifact: ${sourceReport}`);
if (!fs.existsSync(runner)) throw new Error(`Missing counterfactual runner: ${runner}`);
if (!fs.existsSync(finalizer)) throw new Error(`Missing counterfactual report finalizer: ${finalizer}`);

const result = spawnSync(python, [
  runner,
  '--source-report', sourceReport,
  '--output-dir', output.outputDir,
  '--model', process.env.GCL_GPT2_COUNTERFACTUAL_MODEL || 'gpt2',
  '--device', device,
  '--variant', protocolVariant,
  '--pilot-pairs', '10',
  '--control-count', '100',
  '--control-magnitude-tolerance', '0.2',
  '--bootstrap-replicates', '5000'
], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(process.env.GCL_GPT2_COUNTERFACTUAL_TIMEOUT_MS || '3600000'),
  maxBuffer: 128 * 1024 * 1024
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);

const reportPath = path.join(output.outputDir, 'report.json');
const archivePath = path.join(output.outputDir, 'raw-counterfactual.npz');
if (!fs.existsSync(reportPath) || !fs.existsSync(archivePath)) {
  throw new Error('Counterfactual runner exited successfully but report or raw archive is missing');
}
const finalized = spawnSync(python, [
  finalizer,
  '--report', reportPath,
  '--bootstrap-replicates', '5000'
], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
  timeout: Number(process.env.GCL_GPT2_COUNTERFACTUAL_TIMEOUT_MS || '3600000'),
  maxBuffer: 128 * 1024 * 1024
});
if (finalized.error) throw finalized.error;
if (finalized.status !== 0) process.exit(finalized.status ?? 1);
const report = JSON.parse(fs.readFileSync(reportPath, 'utf8')) as {
  kind?: string;
  model?: { id?: string; device?: string };
  pairInventory?: { eligibleUnorderedPairCount?: number };
  protocolConfig?: { selectedPairCount?: number };
  acceptanceCriteria?: Record<string, unknown>;
  pairedStatistics?: unknown;
};
if (report.kind !== 'gpt2-counterfactual-circuit-transfer') {
  throw new Error('Counterfactual runner emitted an unexpected report kind');
}

writeBenchmarkManifest(repoRoot, output, {
  lane: 'nn-live-gpt2-counterfactual-circuit-transfer',
  status: 'generated',
  summary: {
    protocolVariant,
    model: report.model,
    eligiblePairCount: report.pairInventory?.eligibleUnorderedPairCount,
    testedPairCount: report.protocolConfig?.selectedPairCount,
    acceptanceCriteria: report.acceptanceCriteria,
    pairedStatistics: report.pairedStatistics,
    reportJson: path.relative(repoRoot, reportPath),
    rawArchive: path.relative(repoRoot, archivePath)
  },
  notes: [
    'Instrumented local GPT-2 activation-patching experiment; not black-box decompilation.',
    'Only source-reported factual prompts, answers, and discovered paths are used.',
    'Prompt slots are excluded from patch positions; static context tokens require exact token-ID alignment.',
    'Transfer evidence is not equated with complete circuit sufficiency.',
    'Magnitude-invalid controls leave specificity unresolved.'
  ]
});
console.log('GPT2_COUNTERFACTUAL_TRANSFER_EXPERIMENT_WRITTEN');
