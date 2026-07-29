#!/usr/bin/env ts-node

/**
 * Randomized stratified cross-family process-separated cold-start court.
 *
 * The existing `nn-live-blind-swiglu-isolated-cold-start-court` proves the opaque solver
 * protocol works on two hand-picked blocks (Mistral layer 15, LLaMA layer 9). That is
 * evidence of family independence, not of selection independence: hand-picked blocks can
 * be the blocks that happened to work.
 *
 * This court removes the hand-picking. Layers, source units, oracle mode, and solver seed
 * are derived from one fixed master seed, stratified across early/middle/late layer bands
 * in both model families. The solver protocol, solver worker, oracle server, and query
 * budget are unchanged; only the teacher-side selection moves.
 *
 * A randomly selected block is allowed to be harder than a hand-picked one. When a case
 * fails, the court records that case's own failures plus its diagnostics and reports the
 * aggregate as failed. It must not be repaired by widening thresholds or reselecting the
 * block, per docs/claim-policy.md: an exposed weakness becomes a gate or a visible
 * limitation, never a quieter setting.
 */

import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

type LayerBand = 'early' | 'middle' | 'late';
type Family = 'mistral' | 'llama';

type FamilySpec = {
  family: Family;
  model: string;
  /** Residual-stream width; also the oracle input dimension. */
  hiddenSize: number;
  /** SwiGLU expansion width; source units are drawn from this range. */
  intermediateSize: number;
  numHiddenLayers: number;
};

type CourtCase = {
  id: string;
  family: Family;
  band: LayerBand;
  model: string;
  dimension: number;
  layer: number;
  units: number[];
  outputChannel: number;
  oracleMode: 'down-proj-output-channel' | 'selected-unit-sum';
  seed: number;
};

type CourtRow = {
  id: string;
  family: Family;
  band: LayerBand;
  dimension: number;
  layer: number;
  oracleMode: string;
  runId: string;
  generatedAt: string;
  pass: boolean;
  /** 'ok' when the orchestrator produced a readable report, 'error' when it did not. */
  outcome: 'ok' | 'error';
  /** Attributed cause for cases with no report; null when a report exists. */
  failureCause: string | null;
  /** Solver stderr witness for cases with no report; null when a report exists. */
  diagnostics: CaseDiagnostics | null;
  processExitCode: number | null;
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
const lane = 'nn-live-blind-swiglu-randomized-cold-start-court';
const regenerate = process.env.GCL_RANDOMIZED_COLD_START_COURT_REGENERATE !== '0';
const canonicalVariant = 'mistral4096d-llama3072d-16unit-q000001-randomized-stratified-opaque-oracle';
const variant = regenerate ? canonicalVariant : `${canonicalVariant}-artifact-audit`;
const canonical = resolveBenchmarkOutput(repoRoot, lane, canonicalVariant);
const output = resolveBenchmarkOutput(repoRoot, lane, variant);
const python = process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python';

const masterSeed = Number(process.env.GCL_RANDOMIZED_COLD_START_COURT_SEED || 260729);
const atomCount = 16;
const activeRank = atomCount * 2;
const quantizationStep = 1e-6;
const maxOracleCalls = 1_500_000;
const perCaseTimeoutMs = 60 * 60 * 1000;

/**
 * Shapes verified against the local model configs rather than assumed:
 * mistral-7b-instruct-v0.3 hidden 4096 / intermediate 14336 / 32 layers,
 * llama-3.2-3b-instruct hidden 3072 / intermediate 8192 / 28 layers.
 * The orchestrator rejects out-of-range units, so a wrong constant fails loudly.
 */
const families: FamilySpec[] = [
  {
    family: 'mistral',
    model: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
    hiddenSize: 4096,
    intermediateSize: 14336,
    numHiddenLayers: 32
  },
  {
    family: 'llama',
    model: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    hiddenSize: 3072,
    intermediateSize: 8192,
    numHiddenLayers: 28
  }
];

const bands: LayerBand[] = ['early', 'middle', 'late'];

const runtimeFiles: Record<string, string> = {
  decompilerSha256: 'tools/llm_reader/blind_swiglu_oracle_decompiler.py',
  orchestratorSha256: 'tools/llm_reader/isolated_swiglu_cold_start.py',
  oracleServerSha256: 'tools/llm_reader/swiglu_oracle_server.py',
  solverWorkerSha256: 'tools/llm_reader/blind_swiglu_solver_worker.py',
  protocolSha256: 'tools/llm_reader/swiglu_oracle_protocol.py'
};

/** mulberry32: small deterministic PRNG, so the generated case set is reproducible from the seed alone. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function intInRange(rand: () => number, minInclusive: number, maxInclusive: number): number {
  return minInclusive + Math.floor(rand() * (maxInclusive - minInclusive + 1));
}

function pickUniqueUnits(rand: () => number, count: number, exclusiveMax: number): number[] {
  const chosen = new Set<number>();
  while (chosen.size < count) chosen.add(intInRange(rand, 0, exclusiveMax - 1));
  return [...chosen];
}

/**
 * Splits usable layers into three bands. Layer 0 and the final layer are excluded because
 * they sit at the embedding and output boundaries and are not representative of the
 * mid-network SwiGLU blocks this lane targets.
 */
function bandRange(spec: FamilySpec, band: LayerBand): { min: number; max: number } {
  const first = 1;
  const last = spec.numHiddenLayers - 2;
  const span = last - first + 1;
  const third = Math.floor(span / 3);
  if (band === 'early') return { min: first, max: first + third - 1 };
  if (band === 'middle') return { min: first + third, max: first + 2 * third - 1 };
  return { min: first + 2 * third, max: last };
}

/**
 * Units whose down-proj coefficient for the chosen channel is zero contribute nothing to
 * the oracle, so a block containing them asks the solver to recover atoms it can never
 * observe. On a 4-bit Mistral layer roughly 8.8% of units are zero for a given channel,
 * which makes a blind 16-unit draw ill-posed about three times out of four. Drawing from
 * the contributing pool removes an impossible task; it does not soften any threshold.
 *
 * The lookup runs entirely on the teacher side and is cached, since it costs a model load.
 */
function contributingUnitPool(spec: FamilySpec, layer: number, outputChannel: number): number[] {
  const cacheDir = path.join(repoRoot, 'benchmarks', 'research', 'contributing-units');
  const cacheKey = `${spec.family}-l${layer}-c${outputChannel}.json`;
  const cachePath = path.join(cacheDir, cacheKey);
  if (fs.existsSync(cachePath)) {
    const cached = JSON.parse(fs.readFileSync(cachePath, 'utf8')) as { contributingUnits?: number[] };
    if (Array.isArray(cached.contributingUnits) && cached.contributingUnits.length) return cached.contributingUnits;
  }
  fs.mkdirSync(cacheDir, { recursive: true });
  const result = spawnSync(python, [
    'tools/llm_reader/list_contributing_units.py',
    '--model', spec.model,
    '--layer', String(layer),
    '--output-channel', String(outputChannel),
    '--output-json', cachePath,
  ], { cwd: repoRoot, env: childEnv(), stdio: 'inherit', timeout: 20 * 60 * 1000 });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`contributing-unit lookup failed for ${spec.family} layer ${layer} channel ${outputChannel}`);
  const payload = JSON.parse(fs.readFileSync(cachePath, 'utf8')) as { contributingUnits?: number[] };
  if (!Array.isArray(payload.contributingUnits) || payload.contributingUnits.length < atomCount) {
    throw new Error(`fewer than ${atomCount} contributing units for ${spec.family} layer ${layer} channel ${outputChannel}`);
  }
  return payload.contributingUnits;
}

function pickUniqueFromPool(rand: () => number, count: number, pool: number[]): number[] {
  const chosen = new Set<number>();
  while (chosen.size < count) chosen.add(pool[intInRange(rand, 0, pool.length - 1)]);
  return [...chosen];
}

function buildCases(): CourtCase[] {
  const cases: CourtCase[] = [];
  for (const spec of families) {
    for (const [bandIndex, band] of bands.entries()) {
      // Per-case stream so adding a family or band does not renumber unrelated cases.
      const rand = mulberry32(masterSeed + bandIndex * 1013 + (spec.family === 'mistral' ? 0 : 7919));
      const range = bandRange(spec, band);
      const layer = intInRange(rand, range.min, range.max);
      // Alternate oracle mode by band so both modes appear in both families.
      const useOutputChannel = bandIndex % 2 === 0;
      const outputChannel = useOutputChannel ? intInRange(rand, 0, spec.hiddenSize - 1) : -1;
      // In selected-unit-sum mode every unit carries weight one, so no pool filter applies.
      const units = useOutputChannel
        ? pickUniqueFromPool(rand, atomCount, contributingUnitPool(spec, layer, outputChannel))
        : pickUniqueUnits(rand, atomCount, spec.intermediateSize);
      cases.push({
        id: `${spec.family}-${spec.hiddenSize}d-${band}-layer${layer}`,
        family: spec.family,
        band,
        model: spec.model,
        dimension: spec.hiddenSize,
        layer,
        units,
        outputChannel,
        oracleMode: useOutputChannel ? 'down-proj-output-channel' : 'selected-unit-sum',
        seed: intInRange(rand, 100000, 999999)
      });
    }
  }
  return cases;
}

function record(value: unknown): JsonRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonRecord) : {};
}

type CaseDiagnostics = {
  lastPhase: string | null;
  runtimeError: string | null;
  observedR2: number | null;
  observedWorstError: number | null;
  observedRawOracleCalls: number | null;
};

/**
 * When the orchestrator aborts on a recovery failure it raises before writing report.json,
 * so the only surviving witness is the solver's stderr progress stream. Reading it turns
 * "no report" into an attributable cause, which is the difference between a recorded
 * negative result and an unexplained blank.
 *
 * These values are diagnostics for the failure record only. They never contribute to a
 * pass decision: a case without report.json always fails.
 */
function readCaseDiagnostics(caseDir: string): CaseDiagnostics {
  const empty: CaseDiagnostics = { lastPhase: null, runtimeError: null, observedR2: null, observedWorstError: null, observedRawOracleCalls: null };
  const stderrPath = path.join(caseDir, 'solver-stderr.log');
  if (!fs.existsSync(stderrPath)) return empty;
  let text: string;
  try {
    text = fs.readFileSync(stderrPath, 'utf8');
  } catch {
    return empty;
  }
  const diagnostics: CaseDiagnostics = { ...empty };
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('RuntimeError:')) diagnostics.runtimeError = trimmed.slice('RuntimeError:'.length).trim();
    if (!trimmed.startsWith('{')) continue;
    let event: unknown;
    try {
      event = JSON.parse(trimmed);
    } catch {
      continue;
    }
    const progress = record(event);
    if (progress.type !== 'progress') continue;
    if (typeof progress.phase === 'string') diagnostics.lastPhase = progress.phase;
    if (typeof progress.r2 === 'number' && Number.isFinite(progress.r2)) diagnostics.observedR2 = progress.r2;
    if (typeof progress.worstError === 'number' && Number.isFinite(progress.worstError)) diagnostics.observedWorstError = progress.worstError;
    if (typeof progress.rawOracleCalls === 'number' && Number.isFinite(progress.rawOracleCalls)) diagnostics.observedRawOracleCalls = progress.rawOracleCalls;
  }
  return diagnostics;
}

/**
 * Classifies why a case without a report failed, from its diagnostics alone.
 * The labels separate "recovery never produced a structure" from "recovery produced a
 * structure that missed the quality bar", because those need different follow-up work.
 */
function classifyFailure(diagnostics: CaseDiagnostics): string {
  if (diagnostics.runtimeError && diagnostics.runtimeError.includes('structure-court-rejected')) return 'structure-not-recovered';
  if (diagnostics.lastPhase === 'validation-complete') {
    const r2Short = diagnostics.observedR2 !== null && diagnostics.observedR2 < 0.999;
    const worstShort = diagnostics.observedWorstError !== null && diagnostics.observedWorstError > 0.005;
    if (r2Short && worstShort) return 'recovered-but-inaccurate';
    if (r2Short) return 'recovered-but-low-r2';
    if (worstShort) return 'recovered-but-worst-error-over-bar';
    return 'recovered-but-rejected-downstream';
  }
  if (diagnostics.runtimeError) return 'runtime-error';
  return 'unclassified';
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

/**
 * Runs one case and returns the orchestrator exit code. A non-zero exit is not thrown:
 * the orchestrator writes its report before returning 1, so a genuine recovery failure
 * must stay visible as a failed case instead of aborting the whole court.
 */
function runCase(testCase: CourtCase): number | null {
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
    '--atom-count', String(atomCount),
    '--active-rank', String(activeRank),
    '--max-oracle-calls', String(maxOracleCalls),
    '--quantization-step', String(quantizationStep),
    '--seed', String(testCase.seed),
    '--output-json', paths.json,
    '--output-md', paths.markdown,
  ], {
    cwd: repoRoot,
    env: childEnv(),
    stdio: 'inherit',
    timeout: perCaseTimeoutMs,
    maxBuffer: 128 * 1024 * 1024,
  });
  if (result.error) {
    console.error(`[randomized-court] ${testCase.id} process error: ${result.error.message}`);
    return null;
  }
  return result.status;
}

function validateCase(testCase: CourtCase, exitCode: number | null): CourtRow {
  const paths = casePaths(testCase);
  const failures: string[] = [];
  const base = {
    id: testCase.id,
    family: testCase.family,
    band: testCase.band,
    dimension: testCase.dimension,
    layer: testCase.layer,
    oracleMode: testCase.oracleMode,
    processExitCode: exitCode,
    reportPath: path.relative(repoRoot, paths.json)
  };

  if (!fs.existsSync(paths.json)) {
    const diagnostics = readCaseDiagnostics(paths.dir);
    const cause = classifyFailure(diagnostics);
    const detail = [
      `orchestrator produced no report (exit ${String(exitCode)})`,
      `cause=${cause}`,
      `lastPhase=${String(diagnostics.lastPhase)}`,
      diagnostics.runtimeError ? `runtimeError=${diagnostics.runtimeError}` : null,
      diagnostics.observedR2 !== null ? `observedR2=${diagnostics.observedR2}` : null,
      diagnostics.observedWorstError !== null ? `observedWorstError=${diagnostics.observedWorstError}` : null
    ].filter(Boolean).join('; ');
    return {
      ...base,
      runId: '',
      generatedAt: '',
      pass: false,
      outcome: 'error',
      failureCause: cause,
      diagnostics,
      r2: Number.NaN,
      mae: Number.NaN,
      worstError: Number.NaN,
      subspaceMinPrincipalCosine: Number.NaN,
      pairedAtomMinJointCosine: Number.NaN,
      pairedAtomCountAbove099: Number.NaN,
      rawOracleCalls: Number.NaN,
      traceLineCount: Number.NaN,
      failures: [detail]
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(paths.json, 'utf8'));
  } catch (error) {
    return {
      ...base,
      runId: '',
      generatedAt: '',
      pass: false,
      outcome: 'error',
      failureCause: 'report-not-parseable',
      diagnostics: readCaseDiagnostics(paths.dir),
      r2: Number.NaN,
      mae: Number.NaN,
      worstError: Number.NaN,
      subspaceMinPrincipalCosine: Number.NaN,
      pairedAtomMinJointCosine: Number.NaN,
      pairedAtomCountAbove099: Number.NaN,
      rawOracleCalls: Number.NaN,
      traceLineCount: Number.NaN,
      failures: [`report is not valid JSON: ${error instanceof Error ? error.message : String(error)}`]
    };
  }

  const report = record(parsed);
  const summary = record(report.summary);
  const witness = record(report.postHocWitness);
  const isolation = record(report.processIsolation);
  const access = record(report.accessAudit);
  const frozen = record(report.frozenRecovery);
  const provenance = record(report.provenance);
  const artifacts = record(report.artifacts);

  const num = (value: unknown, label: string): number => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      failures.push(`${label} is not a finite number`);
      return Number.NaN;
    }
    return value;
  };
  const emptyList = (value: unknown, label: string): void => {
    if (!Array.isArray(value)) failures.push(`${label} is missing or not an array`);
    else if (value.length) failures.push(`${label} is not empty: ${JSON.stringify(value).slice(0, 200)}`);
  };

  const r2 = num(summary.r2, 'r2');
  const mae = num(summary.mae, 'mae');
  const worstError = num(summary.worstError, 'worstError');
  const subspace = num(summary.subspaceMinPrincipalCosine, 'subspaceMinPrincipalCosine');
  const pairedMin = num(summary.pairedAtomMinJointCosine, 'pairedAtomMinJointCosine');
  const pairedCount = num(summary.pairedAtomCountAbove099, 'pairedAtomCountAbove099');
  const rawOracleCalls = num(summary.rawOracleCalls, 'rawOracleCalls');
  const traceLineCount = num(isolation.solverFileAccessTraceLineCount, 'solverFileAccessTraceLineCount');

  // Recovery quality. Same thresholds as the hand-picked court; a randomized block does not get an easier bar.
  if (report.status !== 'pass' || summary.pass !== true) failures.push('report is not pass');
  if (typeof report.runId !== 'string' || report.runId.length < 16) failures.push('runId is missing or too short');
  if (typeof report.generatedAt !== 'string' || !report.generatedAt) failures.push('generatedAt is missing');
  if (r2 < 0.999) failures.push(`R2=${String(summary.r2)} < 0.999`);
  if (mae > 0.001) failures.push(`MAE=${String(summary.mae)} > 0.001`);
  if (worstError > 0.005) failures.push(`worstError=${String(summary.worstError)} > 0.005`);
  if (subspace < 0.999) failures.push(`subspace cosine=${String(summary.subspaceMinPrincipalCosine)} < 0.999`);
  if (pairedMin < 0.99) failures.push(`paired atom minimum=${String(summary.pairedAtomMinJointCosine)} < 0.99`);
  if (pairedCount !== atomCount) failures.push(`paired atom count is not ${atomCount}/${atomCount}: ${String(summary.pairedAtomCountAbove099)}`);
  if (rawOracleCalls > maxOracleCalls) failures.push(`oracle budget exceeded: ${String(summary.rawOracleCalls)}`);

  // Public shape the solver was allowed to know.
  if (summary.dimension !== testCase.dimension) failures.push(`dimension is not ${testCase.dimension}`);
  if (summary.unitCount !== atomCount) failures.push(`unitCount is not ${atomCount}`);
  if (summary.quantizationStep !== quantizationStep) failures.push(`quantizationStep is not ${quantizationStep}`);

  // Post-hoc witness must match exactly the block this court selected, not some other block.
  if (witness.model !== testCase.model) failures.push('post-hoc model mismatch');
  if (witness.layer !== testCase.layer) failures.push('post-hoc layer mismatch');
  if (JSON.stringify(witness.units) !== JSON.stringify(testCase.units)) failures.push('post-hoc source-unit mismatch');
  const expectedChannel = testCase.outputChannel < 0 ? null : testCase.outputChannel;
  if (witness.outputChannel !== expectedChannel) failures.push('post-hoc output-channel mismatch');
  if (witness.oracleMode !== testCase.oracleMode) failures.push(`post-hoc oracle mode is not ${testCase.oracleMode}`);

  // Process separation.
  if (isolation.orchestratorOwnsTeacherManifest !== true) failures.push('orchestrator teacher-manifest ownership missing');
  if (isolation.oracleProtocol !== 'gcl-swiglu-eval-v1') failures.push('wrong oracle protocol');
  if (isolation.postHocStartedAfterSolverExit !== true) failures.push('post-hoc verifier did not start after solver exit');
  if (isolation.queryTranscriptMatch !== true) failures.push('solver/oracle query transcript mismatch');
  if (isolation.solverFileAccessAuditComplete !== true) failures.push('solver file-access audit incomplete');
  if (typeof isolation.solverFileAccessTraceSha256 !== 'string' || !/^[0-9a-f]{64}$/.test(isolation.solverFileAccessTraceSha256)) {
    failures.push('solver file-access trace hash missing or malformed');
  }
  if (traceLineCount <= 0) failures.push('solver file-access trace is empty');
  emptyList(isolation.solverForbiddenTeacherFlags, 'solver argv teacher flags');
  emptyList(isolation.solverManifestEnvKeys, 'solver environment teacher-manifest keys');
  emptyList(isolation.solverSourceLocatorMatches, 'solver argv source-locator hints');
  emptyList(isolation.solverModelAccessMatches, 'solver model/weight access');
  emptyList(isolation.solverControlFileAccessMatches, 'solver control-file access');
  if (!Array.isArray(isolation.solverEnvironmentKeys)) {
    failures.push('solverEnvironmentKeys is missing or not an array');
  } else {
    const solverEnv = isolation.solverEnvironmentKeys.map(String);
    for (const forbidden of ['HOME', 'HF_HOME', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']) {
      if (solverEnv.includes(forbidden)) failures.push(`solver environment exposes ${forbidden}`);
    }
  }

  // Blind-access invariant.
  if (access.solverAccess !== 'opaque-eval-only-oracle') failures.push('solver access is not opaque eval-only');
  if (access.weightsUsedBySolver !== false || access.hooksUsedBySolver !== false || access.sourceIdsUsedBySolver !== false) {
    failures.push('solver weights/hooks/source-id invariant failed');
  }
  if (access.fitSnapshotUsedBySolver !== false || access.nonlinearInitialFitUsed !== false) failures.push('cold-start invariant failed');
  if (access.truthCapabilityAvailableToSolver !== false) failures.push('solver had a truth capability');
  if (access.postHocTruthAfterSolverExit !== true) failures.push('post-hoc truth ordering failed');

  // Frozen result and provenance.
  if (frozen.immutableAcrossVerification !== true || frozen.sha256BeforeVerification !== frozen.sha256AfterVerification) {
    failures.push('frozen recovery changed across verification');
  }
  for (const [hashKey, relativePath] of Object.entries(runtimeFiles)) {
    if (provenance[hashKey] !== sha256(path.join(repoRoot, relativePath))) failures.push(`${hashKey} does not match current code`);
  }

  const safeName = (value: unknown): string | null => {
    if (typeof value !== 'string') return null;
    const trimmed = value.trim();
    if (!trimmed || trimmed.includes('/') || trimmed.includes('\\') || trimmed.includes('..')) return null;
    return trimmed;
  };
  const traceName = safeName(artifacts.solverFileAccessTrace);
  const tracePath = traceName ? path.join(paths.dir, traceName) : null;
  if (!tracePath || !fs.existsSync(tracePath)) failures.push('solver file-access trace artifact missing');
  else if (sha256(tracePath) !== isolation.solverFileAccessTraceSha256) failures.push('solver file-access trace hash mismatch');
  const frozenName = safeName(frozen.path);
  const frozenPath = frozenName ? path.join(paths.dir, frozenName) : null;
  if (!frozenPath || !fs.existsSync(frozenPath)) failures.push('frozen recovery artifact missing');
  else if (sha256(frozenPath) !== frozen.sha256BeforeVerification) failures.push('frozen recovery hash mismatch');

  // A clean recovery must also have exited zero. A mismatch means the report and the process disagree.
  if (exitCode !== null && exitCode !== 0 && failures.length === 0) {
    failures.push(`orchestrator exited ${exitCode} while reporting no failures`);
  }

  return {
    ...base,
    runId: typeof report.runId === 'string' ? report.runId : '',
    generatedAt: typeof report.generatedAt === 'string' ? report.generatedAt : '',
    pass: failures.length === 0,
    outcome: 'ok',
    failureCause: failures.length === 0 ? null : 'report-failed-court-checks',
    diagnostics: null,
    r2,
    mae,
    worstError,
    subspaceMinPrincipalCosine: subspace,
    pairedAtomMinJointCosine: pairedMin,
    pairedAtomCountAbove099: pairedCount,
    rawOracleCalls,
    traceLineCount,
    failures
  };
}

const cases = buildCases();

/**
 * Plan-only mode prints the derived case set and exits. Reviewing which blocks the seed
 * selected is worth doing before committing GPU time to a full court run.
 */
if (process.env.GCL_RANDOMIZED_COLD_START_COURT_PLAN_ONLY === '1') {
  console.log(JSON.stringify({ masterSeed, atomCount, activeRank, quantizationStep, maxOracleCalls, cases }, null, 2));
  process.exit(0);
}

const exitCodes = new Map<string, number | null>();

if (regenerate) {
  fs.mkdirSync(canonical.outputDir, { recursive: true });
  fs.writeFileSync(
    path.join(canonical.outputDir, 'case-plan.json'),
    `${JSON.stringify({ masterSeed, atomCount, activeRank, quantizationStep, maxOracleCalls, cases }, null, 2)}\n`
  );
  for (const testCase of cases) {
    console.log(`[randomized-court] running ${testCase.id} (${testCase.oracleMode}, seed ${testCase.seed})`);
    exitCodes.set(testCase.id, runCase(testCase));
  }
} else {
  for (const testCase of cases) exitCodes.set(testCase.id, null);
}

const rows = cases.map(testCase => validateCase(testCase, exitCodes.get(testCase.id) ?? null));
const failures = rows.flatMap(row => row.failures.map(failure => `${row.id}: ${failure}`));
const passedRows = rows.filter(row => row.pass);
const finite = (values: number[]): number[] => values.filter(value => Number.isFinite(value));
const minOf = (values: number[]): number | null => (finite(values).length ? Math.min(...finite(values)) : null);
const maxOf = (values: number[]): number | null => (finite(values).length ? Math.max(...finite(values)) : null);

const summary = {
  pass: failures.length === 0 && passedRows.length === rows.length,
  regenerated: regenerate,
  masterSeed,
  caseCount: rows.length,
  passedCases: passedRows.length,
  failedCases: rows.length - passedRows.length,
  /** Failure causes with counts, so the negative result stays attributable in the aggregate alone. */
  failureCauses: rows
    .filter(row => !row.pass)
    .reduce<Record<string, number>>((counts, row) => {
      const cause = row.failureCause || 'unclassified';
      counts[cause] = (counts[cause] || 0) + 1;
      return counts;
    }, {}),
  modelFamilies: [...new Set(rows.map(row => row.family))].sort(),
  layerBands: [...new Set(rows.map(row => row.band))].sort(),
  oracleModes: [...new Set(rows.map(row => row.oracleMode))].sort(),
  selectedLayers: Object.fromEntries(
    families.map(spec => [spec.family, rows.filter(row => row.family === spec.family).map(row => row.layer).sort((a, b) => a - b)])
  ),
  dimensions: [...new Set(rows.map(row => row.dimension))].sort((a, b) => a - b),
  unitCount: atomCount,
  quantizationStep,
  // These aggregates only cover cases that produced a gated report. A case that dies
  // before writing one carries NaN, and dropping those silently made a 1-of-6 run
  // headline the single passing case's numbers as if they were court-wide: minR2 read
  // 0.99999995 while five cases sat between 0.9927 and 0.99999. The coverage fields
  // below make that impossible to read by accident, and the observed variants carry the
  // solver's own progress values for the cases that have no gated report, so the real
  // spread is visible even when the gated set is incomplete.
  measuredCaseCount: finite(rows.map(row => row.r2)).length,
  metricsCoverAllCases: finite(rows.map(row => row.r2)).length === rows.length,
  minR2: minOf(rows.map(row => row.r2)),
  maxMae: maxOf(rows.map(row => row.mae)),
  maxWorstError: maxOf(rows.map(row => row.worstError)),
  minSubspacePrincipalCosine: minOf(rows.map(row => row.subspaceMinPrincipalCosine)),
  minPairedAtomJointCosine: minOf(rows.map(row => row.pairedAtomMinJointCosine)),
  minPairedAtomCountAbove099: minOf(rows.map(row => row.pairedAtomCountAbove099)),
  maxRawOracleCallsPerCase: maxOf(rows.map(row => row.rawOracleCalls)),
  totalRawOracleCalls: finite(rows.map(row => row.rawOracleCalls)).reduce((sum, value) => sum + value, 0),
  minTraceLineCount: minOf(rows.map(row => row.traceLineCount)),
  minR2IncludingUngatedCases: minOf(rows.map(row => (Number.isFinite(row.r2) ? row.r2 : (row.diagnostics?.observedR2 ?? Number.NaN)))),
  maxWorstErrorIncludingUngatedCases: maxOf(rows.map(row => (Number.isFinite(row.worstError) ? row.worstError : (row.diagnostics?.observedWorstError ?? Number.NaN)))),
  totalRawOracleCallsIncludingUngatedCases: finite(
    rows.map(row => (Number.isFinite(row.rawOracleCalls) ? row.rawOracleCalls : (row.diagnostics?.observedRawOracleCalls ?? Number.NaN)))
  ).reduce((sum, value) => sum + value, 0),
  failures
};

fs.mkdirSync(output.outputDir, { recursive: true });
const aggregate = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  kind: regenerate ? 'randomized-stratified-cold-start-court' : 'randomized-stratified-cold-start-artifact-audit',
  status: summary.pass ? 'pass' : 'fail',
  summary,
  cases: rows,
  provenance: {
    courtScriptSha256: sha256(__filename),
    runtimeSha256: Object.fromEntries(Object.entries(runtimeFiles).map(([key, relativePath]) => [key, sha256(path.join(repoRoot, relativePath))]))
  }
};
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(aggregate, null, 2)}\n`);
fs.writeFileSync(output.markdownOutputPath, [
  '# Randomized Stratified SwiGLU Cold-Start Court',
  '',
  `- status: \`${aggregate.status}\``,
  `- master seed: \`${masterSeed}\``,
  `- cases: \`${summary.passedCases}/${summary.caseCount}\``,
  `- families: \`${summary.modelFamilies.join(', ')}\``,
  `- layer bands: \`${summary.layerBands.join(', ')}\``,
  `- oracle modes: \`${summary.oracleModes.join(', ')}\``,
  `- metric coverage: \`${summary.measuredCaseCount}/${summary.caseCount}\` cases produced a gated report`,
  `- min R2 over gated cases: \`${String(summary.minR2)}\``,
  `- min R2 including ungated cases: \`${String(summary.minR2IncludingUngatedCases)}\``,
  `- min paired atom cosine: \`${String(summary.minPairedAtomJointCosine)}\``,
  `- max oracle calls per case: \`${String(summary.maxRawOracleCallsPerCase)}/${maxOracleCalls}\``,
  '',
  '| case | family | band | dim | layer | oracle mode | R2 | worst error | pairs | calls | cause | pass |',
  '| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |',
  ...rows.map(row => {
    // A gated metric is only reported when the report exists. For a case that died before
    // writing one, the solver's own progress value is shown and marked 'observed' so it is
    // never mistaken for a verified measurement.
    const cell = (gated: number, observed: number | null | undefined): string => {
      if (Number.isFinite(gated)) return String(gated);
      return observed === null || observed === undefined ? 'n/a' : `${observed} (observed)`;
    };
    const pairsCell = Number.isFinite(row.pairedAtomCountAbove099) ? `${row.pairedAtomCountAbove099}/${atomCount}` : 'n/a';
    return `| ${row.id} | ${row.family} | ${row.band} | ${row.dimension} | ${row.layer} | ${row.oracleMode} | ${cell(row.r2, row.diagnostics?.observedR2)} | ${cell(row.worstError, row.diagnostics?.observedWorstError)} | ${pairsCell} | ${cell(row.rawOracleCalls, row.diagnostics?.observedRawOracleCalls)} | ${row.failureCause || 'pass'} | ${row.pass} |`;
  }),
  ...(failures.length ? ['', '## Failures', '', ...failures.map(failure => `- ${failure}`)] : []),
  ''
].join('\n'));
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: [
    'Layers, source units, oracle mode, and solver seed are derived from one master seed, not hand-picked.',
    'The solver protocol, solver worker, oracle server, and query budget are identical to the hand-picked court.',
    'A failing randomly selected block is reported as a failed case, never repaired by widening thresholds.'
  ]
});
console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
