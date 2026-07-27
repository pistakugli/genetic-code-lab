#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-isolated-cold-start';
const regenerate = process.env.GCL_ISOLATED_COLD_START_REGENERATE !== '0';
const canonicalVariant = 'mistral4096d-layer15-16unit-q000001-opaque-oracle';
const variant = regenerate ? canonicalVariant : `${canonicalVariant}-artifact-audit`;
const canonical = resolveBenchmarkOutput(repoRoot, lane, canonicalVariant);
const output = resolveBenchmarkOutput(repoRoot, lane, variant);

function record(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value;
}

function finiteNumber(value: unknown, label: string, failures: string[]): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    failures.push(`${label} is not a finite number`);
    return Number.NaN;
  }
  return value;
}

function sha256(relativePath: string): string {
  return crypto.createHash('sha256').update(fs.readFileSync(path.join(repoRoot, relativePath))).digest('hex');
}

function childEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of ['PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']) {
    if (process.env[key] !== undefined) env[key] = process.env[key];
  }
  return env;
}

if (regenerate) {
  fs.mkdirSync(canonical.outputDir, { recursive: true });
  const python = process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python';
  const result = spawnSync(python, [
    'tools/llm_reader/isolated_swiglu_cold_start.py',
    '--output-json', canonical.jsonOutputPath,
    '--output-md', canonical.markdownOutputPath,
  ], {
    cwd: repoRoot,
    env: childEnv(),
    stdio: 'inherit',
    timeout: 60 * 60 * 1000,
    maxBuffer: 128 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`isolated cold-start court exited with status ${String(result.status)}`);
}

if (!fs.existsSync(canonical.jsonOutputPath)) throw new Error(`missing canonical isolated cold-start report: ${canonical.jsonOutputPath}`);
const report = record(JSON.parse(fs.readFileSync(canonical.jsonOutputPath, 'utf8')), 'report');
const summary = record(report.summary, 'summary');
const processIsolation = record(report.processIsolation, 'processIsolation');
const access = record(report.accessAudit, 'accessAudit');
const frozen = record(report.frozenRecovery, 'frozenRecovery');
const provenance = record(report.provenance, 'provenance');
const failures: string[] = [];
const r2 = finiteNumber(summary.r2, 'R2', failures);
const mae = finiteNumber(summary.mae, 'MAE', failures);
const worstError = finiteNumber(summary.worstError, 'worstError', failures);
const subspaceMin = finiteNumber(summary.subspaceMinPrincipalCosine, 'subspaceMinPrincipalCosine', failures);
const pairedMin = finiteNumber(summary.pairedAtomMinJointCosine, 'pairedAtomMinJointCosine', failures);
const rawOracleCalls = finiteNumber(summary.rawOracleCalls, 'rawOracleCalls', failures);
if (report.status !== 'pass' || summary.pass !== true) failures.push('canonical report is not pass');
if (typeof report.runId !== 'string' || report.runId.length < 16) failures.push('missing runId');
if (typeof report.generatedAt !== 'string' || !report.generatedAt) failures.push('missing generatedAt');
if (r2 < 0.999) failures.push(`R2=${String(summary.r2)}<0.999`);
if (mae > 0.001) failures.push(`MAE=${String(summary.mae)}>0.001`);
if (worstError > 0.005) failures.push(`worstError=${String(summary.worstError)}>0.005`);
if (subspaceMin < 0.999) failures.push('subspace cosine below 0.999');
if (pairedMin < 0.99) failures.push('paired atom minimum below 0.99');
if (summary.pairedAtomCountAbove099 !== 16) failures.push('paired atom count is not 16/16');
if (rawOracleCalls > 1_500_000) failures.push('oracle budget exceeded');
if (processIsolation.orchestratorOwnsTeacherManifest !== true) failures.push('orchestrator manifest ownership missing');
if (processIsolation.oracleProtocol !== 'gcl-swiglu-eval-v1') failures.push('wrong oracle protocol');
if (processIsolation.postHocStartedAfterSolverExit !== true) failures.push('post-hoc verifier ordering failed');
if (processIsolation.queryTranscriptMatch !== true) failures.push('query transcript mismatch');
if (processIsolation.solverFileAccessAuditComplete !== true) failures.push('solver file-access audit is incomplete');
if (typeof processIsolation.solverFileAccessTraceSha256 !== 'string' || processIsolation.solverFileAccessTraceSha256.length !== 64) failures.push('solver file-access trace hash missing');
if (finiteNumber(processIsolation.solverFileAccessTraceLineCount, 'solverFileAccessTraceLineCount', failures) <= 0) failures.push('solver file-access trace is empty');
if (array(processIsolation.solverForbiddenTeacherFlags, 'solverForbiddenTeacherFlags').length) failures.push('solver argv contains teacher flags');
if (array(processIsolation.solverManifestEnvKeys, 'solverManifestEnvKeys').length) failures.push('solver env contains teacher manifest keys');
if (array(processIsolation.solverSourceLocatorMatches, 'solverSourceLocatorMatches').length) failures.push('solver argv contains source-locator hints');
if (array(processIsolation.solverModelAccessMatches, 'solverModelAccessMatches').length) failures.push('solver opened model/cache paths');
if (array(processIsolation.solverControlFileAccessMatches, 'solverControlFileAccessMatches').length) failures.push('solver opened control files');
const solverEnv = array(processIsolation.solverEnvironmentKeys, 'solverEnvironmentKeys').map(String);
for (const forbidden of ['HOME', 'HF_HOME', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']) {
  if (solverEnv.includes(forbidden)) failures.push(`solver environment contains ${forbidden}`);
}
if (access.solverAccess !== 'opaque-eval-only-oracle') failures.push('solver access mode is not opaque eval-only');
if (access.weightsUsedBySolver !== false || access.hooksUsedBySolver !== false || access.sourceIdsUsedBySolver !== false) failures.push('solver access invariant failed');
if (access.fitSnapshotUsedBySolver !== false || access.nonlinearInitialFitUsed !== false) failures.push('cold-start input/initial-fit invariant failed');
if (access.truthCapabilityAvailableToSolver !== false || access.postHocTruthAfterSolverExit !== true) failures.push('truth isolation invariant failed');
if (frozen.immutableAcrossVerification !== true || frozen.sha256BeforeVerification !== frozen.sha256AfterVerification) failures.push('frozen recovery hash changed');
const expectedHashes: Record<string, string> = {
  decompilerSha256: sha256('tools/llm_reader/blind_swiglu_oracle_decompiler.py'),
  orchestratorSha256: sha256('tools/llm_reader/isolated_swiglu_cold_start.py'),
  oracleServerSha256: sha256('tools/llm_reader/swiglu_oracle_server.py'),
  solverWorkerSha256: sha256('tools/llm_reader/blind_swiglu_solver_worker.py'),
  protocolSha256: sha256('tools/llm_reader/swiglu_oracle_protocol.py'),
};
for (const [key, expected] of Object.entries(expectedHashes)) {
  if (provenance[key] !== expected) failures.push(`${key} does not match current code`);
}

if (!regenerate) {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const audit = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    kind: 'isolated-cold-start-artifact-audit',
    status: failures.length ? 'fail' : 'pass',
    sourceRunId: report.runId,
    sourceReport: path.relative(repoRoot, canonical.jsonOutputPath),
    summary: { pass: failures.length === 0, failures },
  };
  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(audit, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, `# Isolated Cold-Start Artifact Audit\n\n- status: \`${audit.status}\`\n- source run: \`${String(report.runId)}\`\n`);
}
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: failures.length ? 'fail' : 'pass',
  summary: { pass: failures.length === 0, regenerated: regenerate, runId: report.runId, failures },
  notes: [
    'The solver uses an opaque eval-only Unix binary protocol and receives no model/layer/unit/output locator.',
    'Source truth is evaluated only by a separate verifier process after solver exit and frozen-artifact hashing.',
  ],
});
console.log(JSON.stringify({ pass: failures.length === 0, regenerated: regenerate, runId: report.runId, ...summary, failures }, null, 2));
if (failures.length) process.exit(1);
