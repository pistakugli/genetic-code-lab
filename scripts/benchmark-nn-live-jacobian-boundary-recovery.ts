#!/usr/bin/env ts-node

// The Jacobian boundary route had artifacts under `benchmarks/release/` and no generator.
// Nothing could reproduce them, nothing checked them, and their numbers had gone stale by
// three fixes - the 128-unit manifest still recorded 106 of 128 up directions long after the
// route reached 128 of 128. `scripts/verify-artifact-provenance.ts` exists precisely for that
// shape of problem, and this lane was invisible to it because it had no script to point at.
//
// The thresholds below are asserted, not observed. A lane whose status is `generated` cannot
// catch a regression, which is the only thing a lane is for.

import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-jacobian-boundary-recovery';

// Per unit count, the probe settings and the call budget the route is expected to stay
// inside. `scale` is the one that decides whether the boundaries can be told apart at all: a
// `silu` knee is `1/(S ||g|| |g.d|)` wide along the line while boundaries sit about `2/k`
// apart, so a line stops separating them once `k` approaches `2 S ||g|| / sqrt(dim)` - about
// 770 units at the 2e4 used here, on this model.
const variants = {
  // Budgets are the measured cost plus a tenth, so a regression in the amplitude sweep's
  // stopping rule shows up as a gate failure rather than as a slower green run.
  // The smallest rungs. They existed as hand-made artifacts with no generator and no saved
  // unit list, so nothing could reproduce them; they are proper variants now.
  //
  // A null-space variant was among them and is not here. Measured on the same 32 units it
  // reaches 31 of 32 at 600 lines and 1047784 calls, against 32 of 32 at 167560 for the
  // amplitude route - six times the cost for a worse result. That is worth knowing and is in
  // the roadmap; it is not worth asserting as a passing gate, and lowering a bar to let it
  // through would have been fitting the threshold to the measurement.
  '16unit': { units: 16, lines: 24, basisPoints: 6, upSamples: 8_000, budget: 150_000, gateMode: 'amplitude', minR2: 0.9985 },
  '32unit': { units: 32, lines: 40, basisPoints: 8, upSamples: 15_000, budget: 220_000, gateMode: 'amplitude', minR2: 0.999 },
  '64unit': { units: 64, lines: 80, basisPoints: 12, upSamples: 30_000, budget: 380_000, gateMode: 'amplitude', minR2: 0.999 },
  '128unit': { units: 128, lines: 160, basisPoints: 20, upSamples: 60_000, budget: 880_000, gateMode: 'amplitude', minR2: 0.999 },
  // Budgets and floors from the ladder run, not from a guess: 256 spent 2074336 calls at R2 0.999417,
  // 512 spent 6155552 at 0.999241.
  '256unit': { units: 256, lines: 320, basisPoints: 36, upSamples: 120_000, budget: 2281769, gateMode: 'amplitude', minR2: 0.999 },
  '512unit': { units: 512, lines: 640, basisPoints: 68, upSamples: 80_000, budget: 6771107, gateMode: 'amplitude', minR2: 0.999 },
} as const;

const requested = process.env.GCL_JACOBIAN_VARIANT ?? '64unit';
if (!(requested in variants)) {
  throw new Error(`unknown variant ${requested}; expected one of ${Object.keys(variants).join(', ')}`);
}
const settings = variants[requested as keyof typeof variants];
const variant = `llama-3072d-layer15-${requested}-${settings.gateMode}-gate`;
const output = resolveBenchmarkOutput(repoRoot, lane, variant);
const regenerate = process.env.GCL_JACOBIAN_REGENERATE !== '0';

function record(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function finiteNumber(value: unknown, label: string, failures: string[]): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    failures.push(`${label} is not a finite number`);
    return Number.NaN;
  }
  return value;
}

function childEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of ['PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME']) {
    if (process.env[key] !== undefined) env[key] = process.env[key];
  }
  // The host's NVIDIA kernel module and userspace library are on different point releases,
  // so NVML refuses while CUDA compute is fine. Without this torch aborts on a check it does
  // not need to make.
  env.PYTORCH_NVML_BASED_CUDA_CHECK = '0';
  return env;
}

const recoveredNpz = path.join(os.tmpdir(), `gcl-jacobian-${requested}-recovered.npz`);
const unitsFile = path.join(repoRoot, 'benchmarks/fixtures/jacobian-boundary-units', `${settings.units}.txt`);
if (!fs.existsSync(unitsFile)) throw new Error(`missing unit fixture: ${unitsFile}`);
const units = fs.readFileSync(unitsFile, 'utf8').trim();
if (units.split(',').length !== settings.units) throw new Error(`unit fixture does not hold ${settings.units} units`);

if (regenerate) {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const python = process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python';
  const result = spawnSync(python, [
    'tools/llm_reader/jacobian_boundary_recovery.py',
    '--model', 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    '--layer', '15',
    '--units', units,
    '--line-count', String(settings.lines),
    '--samples', '801',
    '--basis-points', String(settings.basisPoints),
    '--basis-scale', '1',
    '--scale', '2e4',
    '--step', '1',
    '--cluster-min-cosine', '0.93',
    '--gate-mode', settings.gateMode,
    '--gate-residual-tolerance', '5e-3',
    '--amplitude-crossings', '3',
    '--min-cluster-size', '4',
    '--up-sample-count', String(settings.upSamples),
    '--up-sample-scale', '2',
    '--max-oracle-calls', String(settings.budget * 4),
    '--output-json', output.jsonOutputPath,
    // Not in the output directory: this is six to twelve megabytes of recovered parameters
    // per variant, which is working material for the intervention check and not an artifact
    // anyone reads.
    '--output-directions-npz', recoveredNpz,
  ], {
    cwd: repoRoot,
    env: childEnv(),
    stdio: ['ignore', 'ignore', 'inherit'],
    timeout: 4 * 60 * 60 * 1000,
    maxBuffer: 128 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  // Recorded whatever else happened. A run that dies on a signal used to be reported only
  // when nothing else had failed, which reads as a pass.
  if (result.signal) throw new Error(`recovery killed by signal ${result.signal}`);
  if (result.status !== 0) throw new Error(`recovery exited with status ${String(result.status)}`);
}

// The held-out R2 above is measured on the probe cloud, which is not where the model lives:
// probe points carry `||x|| ~ 111` while real hidden states put `g.x` at a standard deviation
// of 0.265. A recovery could interpolate the probe distribution well and be wrong in the
// regime that matters - audit finding F24. The intervention check runs the recovered units
// against real sentences and compares against a hooked forward pass, so the gate covers both.
const interventionPath = path.join(output.outputDir, 'intervention.json');
if (regenerate) {
  const python = process.env.GCL_PYTHON || '.gcl-python-gpu/bin/python';
  const checked = spawnSync(python, [
    'tools/llm_reader/unit_intervention_test.py',
    '--recovered-npz', recoveredNpz,
    '--model', 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    '--layer', '15',
    '--text', 'The capital of France is Paris, a city known for its architecture.',
    '--text', 'Photosynthesis converts light energy into chemical energy stored in glucose.',
    '--text', 'def quicksort(items): pivot = items[0]; return sorted(items)',
    '--max-units', String(settings.units),
    '--output-json', interventionPath,
  ], { cwd: repoRoot, env: childEnv(), stdio: ['ignore', 'ignore', 'inherit'], timeout: 30 * 60 * 1000 });
  if (checked.error) throw checked.error;
  if (checked.signal) throw new Error(`intervention check killed by signal ${checked.signal}`);
  if (checked.status !== 0) throw new Error(`intervention check exited with status ${String(checked.status)}`);
}

if (!fs.existsSync(output.jsonOutputPath)) throw new Error(`missing report: ${output.jsonOutputPath}`);
const report = record(JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8')), 'report');
const identifiability = record(report.identifiability, 'identifiability');
const posthoc = record(report.posthoc, 'posthoc');
const access = record(report.accessAudit, 'accessAudit');
const telemetry = record(report.telemetry, 'telemetry');
const failures: string[] = [];

// The verdict first, because a report whose structure claim is not allowed must not be read
// metric first.
if (identifiability.structureClaimAllowed !== true) failures.push('structure claim not allowed');
if (identifiability.verdict !== 'recovered') failures.push(`verdict is ${String(identifiability.verdict)}, not recovered`);

// Directions with the norm, matched to one source unit. The joint direction bar reported 64
// of 64 on a run whose gate norms were wrong by four orders of magnitude, which is a measure
// a bad solution also satisfies.
const fullyRecovered = finiteNumber(posthoc.unitsFullyRecovered, 'unitsFullyRecovered', failures);
if (fullyRecovered !== settings.units) failures.push(`unitsFullyRecovered=${String(fullyRecovered)}, expected ${settings.units}`);
if (finiteNumber(posthoc.spuriousSolutions, 'spuriousSolutions', failures) !== 0) failures.push('spurious solutions returned');
if (finiteNumber(posthoc.solutionCount, 'solutionCount', failures) !== settings.units) failures.push('solution count does not match the block');

// The one number a caller without the model's weights gets. Held-out points, so a missed unit
// shows up here even though every recovered unit looks clean.
const reconstruction = record(report.reconstruction, 'reconstruction');
const rSquared = finiteNumber(reconstruction.rSquared, 'reconstruction.rSquared', failures);
// Per variant, and set from measurement rather than from a round number. A 16-unit block
// reconstructs to 0.99868 and stays there from 8000 probe samples to 80000, so that is the
// block's floor and not a sampling artefact: with fewer units, each unit's parameter error is
// a larger share of the total signal. Applying the 64-unit bar to it would have been
// extrapolation, and lowering one global bar to fit would have hidden the effect. This is a
// ratchet, the same shape the rest of the repository uses for baselines.
if (rSquared < settings.minR2) failures.push(`held-out R2=${String(rSquared)}<${settings.minR2}`);
if (finiteNumber(reconstruction.relativeResidual, 'reconstruction.relativeResidual', failures) > 0.05) {
  failures.push('held-out relative residual above 0.05');
}

const intervention = record(JSON.parse(fs.readFileSync(interventionPath, 'utf8')), 'intervention');
const deltaCosine = finiteNumber(intervention.deltaCosineMedian, 'intervention.deltaCosineMedian', failures);
if (deltaCosine < 0.99) failures.push(`silencing delta cosine on real text ${String(deltaCosine)}<0.99`);
if (finiteNumber(intervention.unitsTested, 'intervention.unitsTested', failures) !== settings.units) {
  failures.push('intervention check did not cover the block');
}

if (access.solverAccess !== 'oracle-only') failures.push('solver access mode is not oracle-only');
if (access.weightsUsedBySolver !== false || access.hooksUsedBySolver !== false) failures.push('solver access invariant failed');
if (access.postHocWitnessOnly !== true) failures.push('post-hoc witness invariant failed');

// The four fields above are literals the tool writes about itself, so checking them proves
// only that it spelled them correctly. Audit finding F17. What follows is evidence: the
// oracle refuses `truth()` until it is unlocked and records the query count at that moment,
// so if the unlock count and the solver's own call count agree, no query was made outside the
// accounted path and none preceded the freeze.
if (typeof access.truthUnlockStage !== 'string' || !access.truthUnlockStage) {
  failures.push('no post-hoc truth unlock stage recorded');
}
const unlockAt = finiteNumber(access.truthUnlockQueryCount, 'accessAudit.truthUnlockQueryCount', failures);
const solverCalls = finiteNumber(access.solverOracleCalls, 'accessAudit.solverOracleCalls', failures);
if (unlockAt !== solverCalls) {
  failures.push(`truth unlocked at query ${String(unlockAt)} but the solver counted ${String(solverCalls)}`);
}

const oracleCalls = finiteNumber(telemetry.oracleCalls, 'telemetry.oracleCalls', failures);
if (oracleCalls !== solverCalls) failures.push('telemetry and access audit disagree on the call count');

// Which code produced this artifact. Without it a manifest keeps reading `pass` long after
// the tool it describes has moved on - which is what happened here within hours of the lane
// being written, and is audit finding F02. The pattern is
// `benchmark-nn-live-blind-swiglu-isolated-cold-start.ts`, which hashes its tools and refuses
// artifacts produced by different ones.
const solverSha256 = crypto
  .createHash('sha256')
  .update(fs.readFileSync(path.join(repoRoot, 'tools/llm_reader/jacobian_boundary_recovery.py')))
  .digest('hex');
const manifestPath = path.join(output.outputDir, 'manifest.json');
if (!regenerate && fs.existsSync(manifestPath)) {
  const previous = record(JSON.parse(fs.readFileSync(manifestPath, 'utf8')), 'manifest');
  const summary = record(previous.summary, 'manifest.summary');
  if (summary.solverSha256 !== solverSha256) {
    failures.push('artifact was produced by a different version of the recovery tool');
  }
}
if (oracleCalls > settings.budget) failures.push(`oracleCalls=${String(oracleCalls)}>${settings.budget}`);

writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: failures.length ? 'fail' : 'pass',
  summary: {
    pass: failures.length === 0,
    regenerated: regenerate,
    unitCount: settings.units,
    unitsFullyRecovered: fullyRecovered,
    spuriousSolutions: posthoc.spuriousSolutions,
    heldOutRSquared: rSquared,
    realTextDeltaCosine: deltaCosine,
    oracleCalls,
    verdict: identifiability.verdict,
    solverSha256,
    truthUnlockQueryCount: access.truthUnlockQueryCount,
    failures,
  },
  notes: [
    'unitsFullyRecovered requires the gate direction, the up direction and the gate norm together, matched to one source unit.',
    'heldOutRSquared is measured without the weights, on points the solve never saw.',
    'Oracle response width is 3072: one call carries 3072 numbers, not 1.',
  ],
});

console.log(JSON.stringify({
  pass: failures.length === 0,
  variant,
  verdict: identifiability.verdict,
  unitsFullyRecovered: fullyRecovered,
  unitCount: settings.units,
  spuriousSolutions: posthoc.spuriousSolutions,
  heldOutRSquared: rSquared,
  realTextDeltaCosine: deltaCosine,
  oracleCalls,
  failures,
}, null, 2));
if (failures.length) process.exit(1);
