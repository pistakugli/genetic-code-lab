#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

type CourtCase = {
  id: string;
  family: 'mistral' | 'llama';
  model: string;
  dimension: number;
  layer: number;
  units: number[];
  outputChannel: number;
  seed: number;
};

type CourtRow = {
  id: string;
  family: CourtCase['family'];
  dimension: number;
  layer: number;
  oracleMode: string;
  runId: string;
  pass: boolean;
  r2: number;
  mae: number;
  worstError: number;
  subspaceMinPrincipalCosine: number;
  pairedAtomMinJointCosine: number;
  pairedAtomCountAbove099: number;
  rawOracleCalls: number;
  traceLineCount: number;
  reportPath: string;
  failures: string[];
};

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-isolated-cold-start-court';
const regenerate = process.env.GCL_ISOLATED_COLD_START_COURT_REGENERATE !== '0';
const canonicalVariant = 'mistral4096d-llama3072d-16unit-q000001-opaque-oracle';
const variant = regenerate ? canonicalVariant : `${canonicalVariant}-artifact-audit`;
const canonical = resolveBenchmarkOutput(repoRoot, lane, canonicalVariant);
const output = resolveBenchmarkOutput(repoRoot, lane, variant);
const python = process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python';
const maxOracleCalls = 1_500_000;

const cases: CourtCase[] = [
  {
    id: 'mistral-4096d-layer15-output2048',
    family: 'mistral',
    model: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
    dimension: 4096,
    layer: 15,
    units: [7885, 2037, 13787, 2761, 5013, 3496, 5314, 9185, 2434, 2487, 223, 5617, 12804, 5572, 12373, 10618],
    outputChannel: 2048,
    seed: 260723,
  },
  {
    id: 'llama-3072d-layer9-selected-sum',
    family: 'llama',
    model: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    dimension: 3072,
    layer: 9,
    units: [958, 1532, 1647, 1765, 2755, 3182, 3856, 4191, 4371, 4373, 5442, 7098, 7145, 7305, 7679, 7685],
    outputChannel: -1,
    seed: 260728,
  },
];

const runtimeFiles = {
  decompilerSha256: 'tools/llm_reader/blind_swiglu_oracle_decompiler.py',
  orchestratorSha256: 'tools/llm_reader/isolated_swiglu_cold_start.py',
  oracleServerSha256: 'tools/llm_reader/swiglu_oracle_server.py',
  solverWorkerSha256: 'tools/llm_reader/blind_swiglu_solver_worker.py',
  protocolSha256: 'tools/llm_reader/swiglu_oracle_protocol.py',
};

function record(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value;
}

function finite(value: unknown, label: string, failures: string[]): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    failures.push(`${label} is not finite`);
    return Number.NaN;
  }
  return value;
}

function sha256(filePath: string): string {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function childEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of ['PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']) {
    if (process.env[key] !== undefined) env[key] = process.env[key];
  }
  return env;
}

function casePaths(testCase: CourtCase): { dir: string; json: string; markdown: string } {
  const dir = path.join(canonical.outputDir, 'cases', testCase.id);
  return { dir, json: path.join(dir, 'report.json'), markdown: path.join(dir, 'report.md') };
}

function runCase(testCase: CourtCase): void {
  const paths = casePaths(testCase);
  fs.rmSync(paths.dir, { recursive: true, force: true });
  fs.mkdirSync(paths.dir, { recursive: true });
  const result = spawnSync(python, [
    'tools/llm_reader/isolated_swiglu_cold_start.py',
    '--model', testCase.model,
    '--layer', String(testCase.layer),
    '--units', testCase.units.join(','),
    '--output-channel', String(testCase.outputChannel),
    '--dimension', String(testCase.dimension),
    '--atom-count', String(testCase.units.length),
    '--active-rank', String(testCase.units.length * 2),
    '--max-oracle-calls', String(maxOracleCalls),
    '--quantization-step', '0.000001',
    '--seed', String(testCase.seed),
    '--output-json', paths.json,
    '--output-md', paths.markdown,
  ], {
    cwd: repoRoot,
    env: childEnv(),
    stdio: 'inherit',
    timeout: 60 * 60 * 1000,
    maxBuffer: 128 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${testCase.id} exited with status ${String(result.status)}`);
}

function validateCase(testCase: CourtCase): CourtRow {
  const paths = casePaths(testCase);
  const failures: string[] = [];
  if (!fs.existsSync(paths.json)) throw new Error(`missing ${testCase.id} report: ${paths.json}`);
  const report = record(JSON.parse(fs.readFileSync(paths.json, 'utf8')), `${testCase.id} report`);
  const summary = record(report.summary, `${testCase.id} summary`);
  const witness = record(report.postHocWitness, `${testCase.id} postHocWitness`);
  const processIsolation = record(report.processIsolation, `${testCase.id} processIsolation`);
  const access = record(report.accessAudit, `${testCase.id} accessAudit`);
  const frozen = record(report.frozenRecovery, `${testCase.id} frozenRecovery`);
  const provenance = record(report.provenance, `${testCase.id} provenance`);
  const artifacts = record(report.artifacts, `${testCase.id} artifacts`);
  const r2 = finite(summary.r2, `${testCase.id} R2`, failures);
  const mae = finite(summary.mae, `${testCase.id} MAE`, failures);
  const worstError = finite(summary.worstError, `${testCase.id} worstError`, failures);
  const subspace = finite(summary.subspaceMinPrincipalCosine, `${testCase.id} subspace`, failures);
  const pairedMin = finite(summary.pairedAtomMinJointCosine, `${testCase.id} pairedMin`, failures);
  const pairedCount = finite(summary.pairedAtomCountAbove099, `${testCase.id} pairedCount`, failures);
  const rawOracleCalls = finite(summary.rawOracleCalls, `${testCase.id} rawOracleCalls`, failures);
  const traceLineCount = finite(processIsolation.solverFileAccessTraceLineCount, `${testCase.id} traceLineCount`, failures);
  if (report.status !== 'pass' || summary.pass !== true) failures.push('report is not pass');
  if (summary.dimension !== testCase.dimension || summary.unitCount !== testCase.units.length) failures.push('public shape mismatch');
  if (summary.quantizationStep !== 1e-6) failures.push('quantization mismatch');
  if (witness.model !== testCase.model || witness.layer !== testCase.layer) failures.push('post-hoc model/layer mismatch');
  if (JSON.stringify(witness.units) !== JSON.stringify(testCase.units)) failures.push('post-hoc source-unit mismatch');
  const expectedOutput = testCase.outputChannel < 0 ? null : testCase.outputChannel;
  if (witness.outputChannel !== expectedOutput) failures.push('post-hoc output-channel mismatch');
  if (r2 < 0.999 || mae > 0.001 || worstError > 0.005 || subspace < 0.999 || pairedMin < 0.99) failures.push('quality threshold failed');
  if (pairedCount !== testCase.units.length) failures.push(`paired count=${pairedCount}/${testCase.units.length}`);
  if (rawOracleCalls > maxOracleCalls) failures.push('oracle budget exceeded');
  if (processIsolation.orchestratorOwnsTeacherManifest !== true || processIsolation.oracleProtocol !== 'gcl-swiglu-eval-v1') failures.push('orchestrator/protocol invariant failed');
  if (processIsolation.postHocStartedAfterSolverExit !== true || processIsolation.queryTranscriptMatch !== true) failures.push('ordering/transcript invariant failed');
  if (processIsolation.solverFileAccessAuditComplete !== true || traceLineCount <= 0) failures.push('file-access trace incomplete');
  if (array(processIsolation.solverForbiddenTeacherFlags, 'solverForbiddenTeacherFlags').length) failures.push('solver teacher flags present');
  if (array(processIsolation.solverManifestEnvKeys, 'solverManifestEnvKeys').length) failures.push('solver manifest env present');
  if (array(processIsolation.solverSourceLocatorMatches, 'solverSourceLocatorMatches').length) failures.push('solver source-locator hint present');
  if (array(processIsolation.solverModelAccessMatches, 'solverModelAccessMatches').length) failures.push('solver model/weight access present');
  if (array(processIsolation.solverControlFileAccessMatches, 'solverControlFileAccessMatches').length) failures.push('solver control-file access present');
  if (access.solverAccess !== 'opaque-eval-only-oracle' || access.weightsUsedBySolver !== false || access.hooksUsedBySolver !== false || access.sourceIdsUsedBySolver !== false) failures.push('solver access invariant failed');
  if (access.fitSnapshotUsedBySolver !== false || access.nonlinearInitialFitUsed !== false || access.truthCapabilityAvailableToSolver !== false) failures.push('cold-start/truth invariant failed');
  if (access.postHocTruthAfterSolverExit !== true) failures.push('post-hoc truth ordering failed');
  if (frozen.immutableAcrossVerification !== true || frozen.sha256BeforeVerification !== frozen.sha256AfterVerification) failures.push('frozen recovery changed');
  for (const [hashKey, relativePath] of Object.entries(runtimeFiles)) {
    if (provenance[hashKey] !== sha256(path.join(repoRoot, relativePath))) failures.push(`${hashKey} is stale`);
  }
  const tracePath = path.join(paths.dir, String(artifacts.solverFileAccessTrace || ''));
  if (!fs.existsSync(tracePath)) failures.push('solver trace artifact missing');
  else if (sha256(tracePath) !== processIsolation.solverFileAccessTraceSha256) failures.push('solver trace hash mismatch');
  const frozenPath = path.join(paths.dir, String(frozen.path || ''));
  if (!fs.existsSync(frozenPath)) failures.push('frozen recovery artifact missing');
  else if (sha256(frozenPath) !== frozen.sha256BeforeVerification) failures.push('frozen recovery hash mismatch');
  return {
    id: testCase.id,
    family: testCase.family,
    dimension: testCase.dimension,
    layer: testCase.layer,
    oracleMode: String(witness.oracleMode || ''),
    runId: String(report.runId || ''),
    pass: failures.length === 0,
    r2,
    mae,
    worstError,
    subspaceMinPrincipalCosine: subspace,
    pairedAtomMinJointCosine: pairedMin,
    pairedAtomCountAbove099: pairedCount,
    rawOracleCalls,
    traceLineCount,
    reportPath: path.relative(repoRoot, paths.json),
    failures,
  };
}

if (regenerate) {
  fs.mkdirSync(canonical.outputDir, { recursive: true });
  for (const testCase of cases) runCase(testCase);
}
const rows = cases.map(validateCase);
const failures = rows.flatMap(row => row.failures.map(failure => `${row.id}: ${failure}`));
const summary = {
  pass: failures.length === 0,
  regenerated: regenerate,
  caseCount: rows.length,
  passedCases: rows.filter(row => row.pass).length,
  modelFamilies: [...new Set(rows.map(row => row.family))].sort(),
  dimensions: [...new Set(rows.map(row => row.dimension))].sort((a, b) => a - b),
  unitCount: 16,
  quantizationStep: 1e-6,
  minR2: Math.min(...rows.map(row => row.r2)),
  maxMae: Math.max(...rows.map(row => row.mae)),
  maxWorstError: Math.max(...rows.map(row => row.worstError)),
  minSubspacePrincipalCosine: Math.min(...rows.map(row => row.subspaceMinPrincipalCosine)),
  minPairedAtomJointCosine: Math.min(...rows.map(row => row.pairedAtomMinJointCosine)),
  minPairedAtomCountAbove099: Math.min(...rows.map(row => row.pairedAtomCountAbove099)),
  maxRawOracleCallsPerCase: Math.max(...rows.map(row => row.rawOracleCalls)),
  totalRawOracleCalls: rows.reduce((sum, row) => sum + row.rawOracleCalls, 0),
  minTraceLineCount: Math.min(...rows.map(row => row.traceLineCount)),
  failures,
};

fs.mkdirSync(output.outputDir, { recursive: true });
const aggregate = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  kind: regenerate ? 'isolated-swiglu-cold-start-cross-family-court' : 'isolated-swiglu-cold-start-cross-family-artifact-audit',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  cases: rows,
  provenance: {
    courtScriptSha256: sha256(__filename),
    runtimeSha256: Object.fromEntries(Object.entries(runtimeFiles).map(([key, relativePath]) => [key, sha256(path.join(repoRoot, relativePath))])),
  },
};
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(aggregate, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Isolated SwiGLU Cold-Start Cross-Family Court',
  '',
  `- status: \`${aggregate.status}\``,
  `- cases: \`${summary.passedCases}/${summary.caseCount}\``,
  `- min R2: \`${summary.minR2}\``,
  `- min paired atom cosine: \`${summary.minPairedAtomJointCosine}\``,
  `- max oracle calls per case: \`${summary.maxRawOracleCallsPerCase}/${maxOracleCalls}\``,
  '',
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Each dedicated solver receives only an opaque eval endpoint plus public shape and budget metadata.',
    'Source alignment is performed by a separate verifier only after solver exit and frozen-artifact hashing.',
  ],
});
console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
