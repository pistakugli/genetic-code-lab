#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type SweepCase = {
  id: string;
  layer: number;
  unit: number;
  units: number[];
  activeRank: number;
  atomCount: number;
  minR2: number;
  maxMae: number;
  maxWorstError: number;
  minSubspaceCosine: number;
};

type CaseReport = {
  summary: {
    pass: boolean;
    model: string;
    modelType: string;
    dimension: number;
    layer: number;
    unit: number;
    units?: number[];
    unitCount?: number;
    oracleMode?: string;
    outputChannel?: number | null;
    activeRank: number;
    atomCount?: number;
    rawOracleCalls: number;
    metrics: {
      mae: number;
      mse: number;
      r2: number;
      worstError: number;
    };
    sourceAlignment: {
      gateMaxAbsCosine: number;
      upMaxAbsCosine: number;
      subspaceMinPrincipalCosine: number;
      subspaceCosines: number[];
      pairedAtomMinJointCosine?: number;
      pairedAtomMeanJointCosine?: number;
      pairedAtomCountAbove095?: number;
      pairedAtomCountAbove099?: number;
      sourceCancellation?: {
        cancellationFraction: number;
        negativePairCount: number;
        pairCountAtOrBelowMinus050: number;
        minimumPairCorrelation: number | null;
      };
    };
    selectedGradientEnergyRatio: number;
  };
  accessAudit: {
    solverAccess: string;
    weightsUsedBySolver: boolean;
    hooksUsedBySolver: boolean;
    sourceIdsUsedBySolver: boolean;
    oracleOwnsWeights: boolean;
    postHocWitnessOnly: boolean;
    processBoundary: string;
  };
};

const repoRoot = path.resolve(__dirname, '..');
const lane = 'nn-live-blind-swiglu-oracle-decompiler';

function envValue(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

function optionalEnvArg(flag: string, name: string): string[] {
  const value = process.env[name];
  return value && value.trim() ? [flag, value.trim()] : [];
}

const modelLabel = envValue('GCL_BLIND_SWIGLU_MODEL_LABEL', 'mistral-4096d');
const expectedDimension = Number(envValue('GCL_BLIND_SWIGLU_EXPECTED_DIMENSION', '4096'));
const familyLabel = envValue('GCL_BLIND_SWIGLU_FAMILY_LABEL', 'Mistral-family');
const minPairedAtomCosine = Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_PAIRED_ATOM_COSINE', '0'));
const outputChannel = Number(envValue('GCL_BLIND_SWIGLU_OUTPUT_CHANNEL', '-1'));
const minSourceCancellationFraction = Number(envValue(
  'GCL_BLIND_SWIGLU_SWEEP_MIN_SOURCE_CANCELLATION_FRACTION',
  '0'
));
const maxSourcePairCorrelation = Number(envValue(
  'GCL_BLIND_SWIGLU_SWEEP_MAX_SOURCE_PAIR_CORRELATION',
  '1'
));

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const gpuPython = path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
  if (fs.existsSync(gpuPython)) return gpuPython;
  return 'python3';
}

function caseVariant(testCase: SweepCase): string {
  const gradientMode = envValue('GCL_BLIND_SWIGLU_GRADIENT_MODE', 'coordinate');
  const quantizationStep = Number(envValue('GCL_BLIND_SWIGLU_ORACLE_QUANTIZATION_STEP', '0'));
  const cubicDirectionCount = Number(envValue('GCL_BLIND_SWIGLU_CUBIC_DIRECTION_COUNT', '0'));
  const cubicHoldoutCount = Number(envValue('GCL_BLIND_SWIGLU_CUBIC_HOLDOUT_COUNT', '0'));
  const cubicSteps = envValue('GCL_BLIND_SWIGLU_CUBIC_STEPS', envValue('GCL_BLIND_SWIGLU_CUBIC_STEP', '0.35'));
  const variantTags = [];
  if (outputChannel >= 0) variantTags.push(`output${outputChannel}`);
  if (quantizationStep > 0) variantTags.push(`q${String(quantizationStep).replace(/^0\./, '').replace(/\./g, '')}`);
  if (cubicDirectionCount > 0) {
    variantTags.push(
      `cubic-richardson-${cubicDirectionCount}h${cubicHoldoutCount}s${cubicSteps.replace(/[^0-9]+/g, '-')}`
    );
  }
  const variantTag = variantTags.length ? `${variantTags.join('-')}-` : '';
  const suffix = gradientMode === 'coordinate' ? `${variantTag}oracle-only` : `${variantTag}${gradientMode}-oracle-only`;
  if (testCase.units.length === 1) return `${modelLabel}-layer${testCase.layer}-unit${testCase.unit}-${suffix}`;
  return `${modelLabel}-layer${testCase.layer}-units${testCase.units.join('-')}-${testCase.atomCount}atom-${suffix}`;
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function parseCases(raw: string): SweepCase[] {
  return raw
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
    .map(item => {
      const [layerRaw, unitRaw, minR2Raw, maxMaeRaw, minSubspaceRaw, maxWorstRaw] = item.split(':');
      const layer = Number(layerRaw);
      const units = unitRaw
        .split(/[+|]/)
        .map(value => Number(value))
        .filter(value => Number.isInteger(value));
      const unit = units[0];
      if (!Number.isInteger(layer) || !Number.isInteger(unit) || units.length === 0) {
        throw new Error(`Invalid blind SwiGLU sweep case ${item}; expected layer:unit or layer:unit+unit[:minR2[:maxMae[:minSubspaceCosine[:maxWorstError]]]]`);
      }
      return {
        id: units.length === 1 ? `layer${layer}-unit${unit}` : `layer${layer}-units${units.join('-')}`,
        layer,
        unit,
        units,
        activeRank: Number(envValue('GCL_BLIND_SWIGLU_ACTIVE_RANK', String(units.length * 2))),
        atomCount: units.length,
        minR2: minR2Raw ? Number(minR2Raw) : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_R2', '0.995')),
        maxMae: maxMaeRaw ? Number(maxMaeRaw) : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MAX_MAE', '0.002')),
        maxWorstError: maxWorstRaw ? Number(maxWorstRaw) : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MAX_WORST_ERROR', '1e30')),
        minSubspaceCosine: minSubspaceRaw
          ? Number(minSubspaceRaw)
          : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_SUBSPACE_COSINE', '0.99'))
      };
    });
}

function generateRandomCases(count: number): SweepCase[] {
  const random = makeRandom(Number(envValue('GCL_BLIND_SWIGLU_SWEEP_RANDOM_SEED', '260621')));
  const layerCount = Number(envValue('GCL_BLIND_SWIGLU_LAYER_COUNT', modelLabel.startsWith('llama') ? '28' : '32'));
  const intermediateSize = Number(envValue('GCL_BLIND_SWIGLU_INTERMEDIATE_SIZE', modelLabel.startsWith('llama') ? '8192' : '14336'));
  const minR2 = Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_R2', '0.995'));
  const maxMae = Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MAX_MAE', '0.002'));
  const maxWorstError = Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MAX_WORST_ERROR', '1e30'));
  const minSubspaceCosine = Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_SUBSPACE_COSINE', '0.99'));
  const unitCount = Number(envValue('GCL_BLIND_SWIGLU_SWEEP_UNIT_COUNT', '1'));
  const activeRank = Number(envValue('GCL_BLIND_SWIGLU_ACTIVE_RANK', String(unitCount * 2)));
  if (!Number.isInteger(count) || count <= 0) throw new Error(`Invalid random sweep count ${count}`);
  if (!Number.isInteger(unitCount) || unitCount <= 0) throw new Error(`Invalid random sweep unit count ${unitCount}`);
  if (!Number.isInteger(layerCount) || layerCount <= 0) throw new Error(`Invalid layer count ${layerCount}`);
  if (!Number.isInteger(intermediateSize) || intermediateSize <= 0) {
    throw new Error(`Invalid intermediate size ${intermediateSize}`);
  }

  const cases: SweepCase[] = [];
  const seen = new Set<string>();
  while (cases.length < count) {
    const layer = Math.floor(random() * layerCount);
    const unitSet = new Set<number>();
    while (unitSet.size < unitCount) unitSet.add(Math.floor(random() * intermediateSize));
    const units = [...unitSet].sort((a, b) => a - b);
    const unit = units[0];
    const key = `${layer}:${units.join('+')}`;
    if (seen.has(key)) continue;
    seen.add(key);
    cases.push({
      id: unitCount === 1
        ? `random${cases.length}-layer${layer}-unit${unit}`
        : `random${cases.length}-layer${layer}-units${units.join('-')}`,
      layer,
      unit,
      units,
      activeRank,
      atomCount: unitCount,
      minR2,
      maxMae,
      maxWorstError,
      minSubspaceCosine
    });
  }
  return cases;
}

function resolveSweepCases(): { cases: SweepCase[]; selection: Record<string, unknown> } {
  const explicit = process.env.GCL_BLIND_SWIGLU_SWEEP_CASES;
  if (explicit && explicit.trim()) {
    return {
      cases: parseCases(explicit),
      selection: { kind: 'explicit', source: 'GCL_BLIND_SWIGLU_SWEEP_CASES' }
    };
  }
  const randomCount = Number(process.env.GCL_BLIND_SWIGLU_SWEEP_RANDOM_COUNT ?? '0');
  if (randomCount > 0) {
    return {
      cases: generateRandomCases(randomCount),
      selection: {
        kind: 'deterministic-random',
        seed: Number(envValue('GCL_BLIND_SWIGLU_SWEEP_RANDOM_SEED', '260621')),
        count: randomCount,
        unitCount: Number(envValue('GCL_BLIND_SWIGLU_SWEEP_UNIT_COUNT', '1')),
        layerCount: Number(envValue('GCL_BLIND_SWIGLU_LAYER_COUNT', modelLabel.startsWith('llama') ? '28' : '32')),
        intermediateSize: Number(envValue('GCL_BLIND_SWIGLU_INTERMEDIATE_SIZE', modelLabel.startsWith('llama') ? '8192' : '14336'))
      }
    };
  }
  return {
    cases: parseCases('8:123:0.995:0.002:0.99,16:2048:0.995:0.002:0.99,20:0:0.995:0.002:0.99,24:17:0.995:0.002:0.99'),
    selection: { kind: 'fixed-canonical' }
  };
}

function resolveOutputForVariant(variant: string) {
  const previous = process.env.GCL_BENCHMARK_VARIANT;
  process.env.GCL_BENCHMARK_VARIANT = variant;
  try {
    return resolveBenchmarkOutput(repoRoot, lane, variant);
  } finally {
    if (previous === undefined) delete process.env.GCL_BENCHMARK_VARIANT;
    else process.env.GCL_BENCHMARK_VARIANT = previous;
  }
}

function assertCaseReport(report: CaseReport, testCase: SweepCase): string[] {
  const failures: string[] = [];
  if (!report.summary.pass) failures.push('summary.pass=false');
  if (report.summary.dimension !== expectedDimension) failures.push(`dimension=${report.summary.dimension}`);
  if (report.summary.layer !== testCase.layer) failures.push(`layer=${report.summary.layer}`);
  if (report.summary.unit !== testCase.unit) failures.push(`unit=${report.summary.unit}`);
  const reportUnits = report.summary.units ?? [report.summary.unit];
  if (JSON.stringify(reportUnits) !== JSON.stringify(testCase.units)) failures.push(`units=${reportUnits.join(',')}`);
  if ((report.summary.unitCount ?? 1) !== testCase.units.length) failures.push(`unitCount=${report.summary.unitCount}`);
  if (outputChannel >= 0 && report.summary.outputChannel !== outputChannel) {
    failures.push(`outputChannel=${report.summary.outputChannel}`);
  }
  if ((report.summary.atomCount ?? 1) !== testCase.atomCount) failures.push(`atomCount=${report.summary.atomCount}`);
  if (report.summary.activeRank !== testCase.activeRank) failures.push(`activeRank=${report.summary.activeRank}`);
  if (report.summary.metrics.r2 < testCase.minR2) failures.push(`r2=${report.summary.metrics.r2}<${testCase.minR2}`);
  if (report.summary.metrics.mae > testCase.maxMae) failures.push(`mae=${report.summary.metrics.mae}>${testCase.maxMae}`);
  if (report.summary.metrics.worstError > testCase.maxWorstError) {
    failures.push(`worstError=${report.summary.metrics.worstError}>${testCase.maxWorstError}`);
  }
  if (report.summary.sourceAlignment.subspaceMinPrincipalCosine < testCase.minSubspaceCosine) {
    failures.push(`subspace=${report.summary.sourceAlignment.subspaceMinPrincipalCosine}<${testCase.minSubspaceCosine}`);
  }
  if (minPairedAtomCosine > 0) {
    const pairedMinimum = report.summary.sourceAlignment.pairedAtomMinJointCosine;
    const pairedCount = minPairedAtomCosine >= 0.99
      ? report.summary.sourceAlignment.pairedAtomCountAbove099
      : report.summary.sourceAlignment.pairedAtomCountAbove095;
    if (pairedMinimum === undefined) failures.push('pairedAtomMinJointCosine=missing');
    else if (pairedMinimum < minPairedAtomCosine) failures.push(`pairedAtomMinJointCosine=${pairedMinimum}<${minPairedAtomCosine}`);
    if (pairedCount === undefined) failures.push('pairedAtomCount=missing');
    else if (pairedCount < testCase.atomCount) failures.push(`pairedAtomCount=${pairedCount}<${testCase.atomCount}`);
  }
  const sourceCancellation = report.summary.sourceAlignment.sourceCancellation;
  if (minSourceCancellationFraction > 0) {
    if (!sourceCancellation) failures.push('sourceCancellation=missing');
    else if (sourceCancellation.cancellationFraction < minSourceCancellationFraction) {
      failures.push(
        `sourceCancellationFraction=${sourceCancellation.cancellationFraction}<${minSourceCancellationFraction}`
      );
    }
  }
  if (maxSourcePairCorrelation < 1) {
    if (!sourceCancellation || sourceCancellation.minimumPairCorrelation === null) {
      failures.push('sourceMinimumPairCorrelation=missing');
    } else if (sourceCancellation.minimumPairCorrelation > maxSourcePairCorrelation) {
      failures.push(
        `sourceMinimumPairCorrelation=${sourceCancellation.minimumPairCorrelation}>${maxSourcePairCorrelation}`
      );
    }
  }
  if (report.accessAudit.solverAccess !== 'oracle-only') failures.push(`solverAccess=${report.accessAudit.solverAccess}`);
  if (report.accessAudit.weightsUsedBySolver !== false) failures.push('weightsUsedBySolver!=false');
  if (report.accessAudit.hooksUsedBySolver !== false) failures.push('hooksUsedBySolver!=false');
  if (report.accessAudit.sourceIdsUsedBySolver !== false) failures.push('sourceIdsUsedBySolver!=false');
  if (report.accessAudit.postHocWitnessOnly !== true) failures.push('postHocWitnessOnly!=true');
  return failures;
}

const toolPath = path.join(repoRoot, 'tools', 'llm_reader', 'blind_swiglu_oracle_decompiler.py');
if (!fs.existsSync(toolPath)) throw new Error(`Missing blind SwiGLU oracle decompiler: ${toolPath}`);

const resolvedSweep = resolveSweepCases();
const sweepCases = resolvedSweep.cases;
if (!sweepCases.length) throw new Error('Blind SwiGLU sweep requires at least one case');

const caseResults: Array<{
  case: SweepCase;
  variant: string;
  reportPath: string;
  pass: boolean;
  failures: string[];
  summary?: CaseReport['summary'];
}> = [];

for (const testCase of sweepCases) {
  const variant = caseVariant(testCase);
  const output = resolveOutputForVariant(variant);
  fs.mkdirSync(output.outputDir, { recursive: true });
  for (const staleOutput of [output.jsonOutputPath, output.markdownOutputPath, output.manifestOutputPath]) {
    fs.rmSync(staleOutput, { force: true });
  }
  const args = [
    toolPath,
    '--model', envValue('GCL_BLIND_SWIGLU_MODEL', 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'),
    '--layer', String(testCase.layer),
    '--unit', String(testCase.unit),
    ...(testCase.units.length > 1 ? ['--units', testCase.units.join(',')] : []),
    ...(outputChannel >= 0 ? ['--output-channel', String(outputChannel)] : []),
    '--device', envValue('GCL_BLIND_SWIGLU_DEVICE', 'cuda'),
    '--oracle-quantization-step', envValue('GCL_BLIND_SWIGLU_ORACLE_QUANTIZATION_STEP', '0'),
    '--active-rank', String(testCase.activeRank),
    ...(testCase.atomCount > 1 ? ['--atom-count', String(testCase.atomCount)] : []),
    '--gradient-mode', envValue('GCL_BLIND_SWIGLU_GRADIENT_MODE', 'coordinate'),
    '--basis-moment', envValue('GCL_BLIND_SWIGLU_BASIS_MOMENT', 'auto'),
    '--probe-points', envValue('GCL_BLIND_SWIGLU_PROBE_POINTS', '2'),
    '--direction-count', envValue('GCL_BLIND_SWIGLU_DIRECTION_COUNT', '384'),
    '--subspace-count', envValue('GCL_BLIND_SWIGLU_SUBSPACE_COUNT', '0'),
    '--cubic-direction-count', envValue('GCL_BLIND_SWIGLU_CUBIC_DIRECTION_COUNT', '0'),
    '--cubic-step', envValue('GCL_BLIND_SWIGLU_CUBIC_STEP', '0.35'),
    '--cubic-steps', envValue('GCL_BLIND_SWIGLU_CUBIC_STEPS', ''),
    '--cubic-holdout-count', envValue('GCL_BLIND_SWIGLU_CUBIC_HOLDOUT_COUNT', '0'),
    '--finite-difference-step', envValue('GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEP', '0.08'),
    '--finite-difference-steps', envValue('GCL_BLIND_SWIGLU_FINITE_DIFFERENCE_STEPS', ''),
    '--input-scale', envValue('GCL_BLIND_SWIGLU_INPUT_SCALE', '0.45'),
    '--train-scale-mixture', envValue('GCL_BLIND_SWIGLU_TRAIN_SCALE_MIXTURE', ''),
    '--train-sampling-mode', envValue('GCL_BLIND_SWIGLU_TRAIN_SAMPLING_MODE', 'gaussian'),
    '--validation-scale', envValue('GCL_BLIND_SWIGLU_VALIDATION_SCALE', '0.65'),
    '--train-count', envValue('GCL_BLIND_SWIGLU_TRAIN_COUNT', '1600'),
    '--validation-count', envValue('GCL_BLIND_SWIGLU_VALIDATION_COUNT', '512'),
    '--fit-steps', envValue('GCL_BLIND_SWIGLU_FIT_STEPS', '1200'),
    '--fit-learning-rate', envValue('GCL_BLIND_SWIGLU_FIT_LEARNING_RATE', '0.025'),
    '--fit-restarts', envValue('GCL_BLIND_SWIGLU_FIT_RESTARTS', '0'),
    '--fit-restart-chunk-size', envValue('GCL_BLIND_SWIGLU_FIT_RESTART_CHUNK_SIZE', '0'),
    '--fit-mode', envValue('GCL_BLIND_SWIGLU_FIT_MODE', 'auto'),
    '--recovery-path', envValue('GCL_BLIND_SWIGLU_RECOVERY_PATH', 'nonlinear-fit'),
    '--fit-gradient-count', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_COUNT', '0'),
    '--fit-gradient-mode', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_MODE', 'forward'),
    '--fit-gradient-step', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_STEP', '0'),
    '--fit-gradient-steps', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_STEPS', ''),
    '--fit-gradient-weight', envValue('GCL_BLIND_SWIGLU_FIT_GRADIENT_WEIGHT', '0.2'),
    '--fit-selection-count', envValue('GCL_BLIND_SWIGLU_FIT_SELECTION_COUNT', '0'),
    '--fit-selection-scale', envValue('GCL_BLIND_SWIGLU_FIT_SELECTION_SCALE', '0'),
    '--selection-tail-weight', envValue('GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT', '0'),
    '--selection-tail-quantile', envValue('GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE', '0.995'),
    '--selection-worst-weight', envValue('GCL_BLIND_SWIGLU_SELECTION_WORST_WEIGHT', '0'),
    '--seed', envValue('GCL_BLIND_SWIGLU_SEED', '260620'),
    ...optionalEnvArg('--polish-top-k', 'GCL_BLIND_SWIGLU_POLISH_TOP_K'),
    ...optionalEnvArg('--backfit-cycles', 'GCL_BLIND_SWIGLU_BACKFIT_CYCLES'),
    ...optionalEnvArg('--tail-polish-steps', 'GCL_BLIND_SWIGLU_TAIL_POLISH_STEPS'),
    ...optionalEnvArg('--tail-polish-weight', 'GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT'),
    ...optionalEnvArg('--tail-polish-fraction', 'GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION'),
    ...optionalEnvArg('--tail-polish-lr', 'GCL_BLIND_SWIGLU_TAIL_POLISH_LR'),
    ...optionalEnvArg('--tail-polish-loss', 'GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS'),
    ...optionalEnvArg('--tail-polish-smoothness', 'GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS'),
    ...optionalEnvArg('--tail-polish-calibration', 'GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION'),
    ...optionalEnvArg('--final-lbfgs-steps', 'GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS'),
    ...optionalEnvArg('--final-lbfgs-tail-weight', 'GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT'),
    ...optionalEnvArg('--final-lbfgs-tail-fraction', 'GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION'),
    ...(envValue('GCL_BLIND_SWIGLU_ROBUST_SCALAR_CALIBRATION', '0') === '1' ? ['--robust-scalar-calibration'] : []),
    ...optionalEnvArg('--scalar-calibration-steps', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_STEPS'),
    ...optionalEnvArg('--scalar-calibration-lr', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_LR'),
    ...optionalEnvArg('--scalar-calibration-tail-weight', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_WEIGHT'),
    ...optionalEnvArg('--scalar-calibration-tail-fraction', 'GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_FRACTION'),
    '--counterexample-count', envValue('GCL_BLIND_SWIGLU_COUNTEREXAMPLE_COUNT', '0'),
    '--counterexample-repeat', envValue('GCL_BLIND_SWIGLU_COUNTEREXAMPLE_REPEAT', '4'),
    '--counterexample-jitter-count', envValue('GCL_BLIND_SWIGLU_COUNTEREXAMPLE_JITTER_COUNT', '0'),
    '--counterexample-jitter-scale', envValue('GCL_BLIND_SWIGLU_COUNTEREXAMPLE_JITTER_SCALE', '0.025'),
    '--tail-probe-count', envValue('GCL_BLIND_SWIGLU_TAIL_PROBE_COUNT', '0'),
    '--tail-probe-repeat', envValue('GCL_BLIND_SWIGLU_TAIL_PROBE_REPEAT', '4'),
    '--tail-probe-scale', envValue('GCL_BLIND_SWIGLU_TAIL_PROBE_SCALE', '0'),
    '--rescue-train-target', envValue('GCL_BLIND_SWIGLU_RESCUE_TRAIN_TARGET', '0'),
    '--rescue-gradient-target', envValue('GCL_BLIND_SWIGLU_RESCUE_GRADIENT_TARGET', '0'),
    '--rescue-steps', envValue('GCL_BLIND_SWIGLU_RESCUE_STEPS', '0'),
    '--rescue-restarts', envValue('GCL_BLIND_SWIGLU_RESCUE_RESTARTS', '0'),
    '--min-r2', String(testCase.minR2),
    '--max-mae', String(testCase.maxMae),
    '--max-worst-error', String(testCase.maxWorstError),
    '--min-subspace-cosine', String(testCase.minSubspaceCosine),
    '--max-oracle-calls', envValue('GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS', '20000'),
    ...optionalEnvArg('--fit-snapshot-input', 'GCL_BLIND_SWIGLU_FIT_SNAPSHOT_INPUT'),
    ...optionalEnvArg('--fit-snapshot-output', 'GCL_BLIND_SWIGLU_FIT_SNAPSHOT_OUTPUT'),
    '--local-files-only',
    ...(envValue('GCL_BLIND_SWIGLU_DISABLE_ADAPTIVE_RESCUE', '0') === '1' ? ['--disable-adaptive-rescue'] : []),
    ...(envValue('GCL_BLIND_SWIGLU_PROGRESS', '0') === '1' ? ['--progress'] : []),
    '--output-json', output.jsonOutputPath,
    '--output-md', output.markdownOutputPath
  ];

  const result = spawnSync(resolveGclPython(), args, {
    cwd: repoRoot,
    stdio: 'inherit',
    env: process.env,
    timeout: Number(envValue('GCL_BLIND_SWIGLU_TIMEOUT_MS', '1800000')),
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;

  let report: CaseReport | undefined;
  let failures: string[] = [];
  if (fs.existsSync(output.jsonOutputPath)) {
    report = JSON.parse(fs.readFileSync(output.jsonOutputPath, 'utf8')) as CaseReport;
    failures = assertCaseReport(report, testCase);
    writeBenchmarkManifest(repoRoot, output, {
      lane,
      status: failures.length === 0 ? 'pass' : 'fail',
      summary: report.summary,
      notes: [
        'Per-case report for the blind 4096D SwiGLU oracle-only sweep.',
        'The solver process receives only f(x) batch outputs. Source weights are requested only after recovery for post-hoc witness alignment.'
      ]
    });
  } else {
    failures = ['missing report.json'];
  }
  if (result.status !== 0 && failures.length === 0) failures.push(`process exit ${result.status ?? 'unknown'}`);
  caseResults.push({
    case: testCase,
    variant,
    reportPath: path.relative(repoRoot, output.jsonOutputPath),
    pass: failures.length === 0,
    failures,
    summary: report?.summary
  });
}

const aggregateVariant = envValue('GCL_BLIND_SWIGLU_SWEEP_VARIANT', `${modelLabel}-4unit-layer-sweep-2probe-oracle-only`);
const aggregateOutput = resolveOutputForVariant(aggregateVariant);
fs.mkdirSync(aggregateOutput.outputDir, { recursive: true });
for (const staleOutput of [aggregateOutput.jsonOutputPath, aggregateOutput.markdownOutputPath, aggregateOutput.manifestOutputPath]) {
  fs.rmSync(staleOutput, { force: true });
}

const summaries = caseResults.map(result => result.summary).filter((summary): summary is CaseReport['summary'] => Boolean(summary));
const aggregate = {
  summary: {
    pass: caseResults.every(result => result.pass),
    caseCount: caseResults.length,
    passedCases: caseResults.filter(result => result.pass).length,
    failedCases: caseResults.filter(result => !result.pass).length,
    minR2: Math.min(...summaries.map(summary => summary.metrics.r2)),
    maxMae: Math.max(...summaries.map(summary => summary.metrics.mae)),
    maxWorstError: Math.max(...summaries.map(summary => summary.metrics.worstError)),
    minSubspacePrincipalCosine: Math.min(...summaries.map(summary => summary.sourceAlignment.subspaceMinPrincipalCosine)),
    minPairedAtomJointCosine: summaries.every(summary => summary.sourceAlignment.pairedAtomMinJointCosine !== undefined)
      ? Math.min(...summaries.map(summary => summary.sourceAlignment.pairedAtomMinJointCosine as number))
      : null,
    minPairedAtomCountAboveThreshold: minPairedAtomCosine > 0
      ? Math.min(...summaries.map(summary => (
        minPairedAtomCosine >= 0.99
          ? summary.sourceAlignment.pairedAtomCountAbove099 ?? 0
          : summary.sourceAlignment.pairedAtomCountAbove095 ?? 0
      )))
      : null,
    minSourceCancellationFraction: summaries.every(summary => summary.sourceAlignment.sourceCancellation !== undefined)
      ? Math.min(...summaries.map(summary => summary.sourceAlignment.sourceCancellation?.cancellationFraction ?? 0))
      : null,
    minimumSourcePairCorrelation: summaries.every(summary => summary.sourceAlignment.sourceCancellation !== undefined)
      ? Math.min(...summaries.map(summary => (
        summary.sourceAlignment.sourceCancellation?.minimumPairCorrelation ?? Number.POSITIVE_INFINITY
      )))
      : null,
    pairedAtomThreshold: minPairedAtomCosine > 0 ? minPairedAtomCosine : null,
    maxRawOracleCallsPerCase: Math.max(...summaries.map(summary => summary.rawOracleCalls)),
    totalRawOracleCalls: summaries.reduce((sum, summary) => sum + summary.rawOracleCalls, 0),
    maxUnitCount: Math.max(...summaries.map(summary => summary.unitCount ?? 1)),
    maxAtomCount: Math.max(...summaries.map(summary => summary.atomCount ?? 1)),
    accessInvariant: caseResults.every(result => !result.failures.some(failure => /solverAccess|weightsUsed|hooksUsed|sourceIds|postHoc/.test(failure)))
  },
  cases: caseResults,
  accessAudit: {
    solverAccess: 'oracle-only',
    weightsUsedBySolver: false,
    hooksUsedBySolver: false,
    sourceIdsUsedBySolver: false,
    postHocWitnessOnly: true,
    perCaseReportsEnforceInvariant: true
  },
  selection: resolvedSweep.selection,
  claimBoundary: [
    `This sweep proves selected real local ${familyLabel} ${expectedDimension}D SwiGLU unit or small-unit-block outputs can be recovered as blind oracle-only explicit SwiGLU atoms under the recorded case budgets.`,
    `It does not prove full ${familyLabel} FFN extraction, all-channel ${expectedDimension}D decompilation, exhaustive LLM circuit recovery, or semantic original-neuron identity from output-only access.`
  ]
};

fs.writeFileSync(aggregateOutput.jsonOutputPath, `${JSON.stringify(aggregate, null, 2)}\n`);
fs.writeFileSync(
  aggregateOutput.markdownOutputPath,
  [
    `# Blind ${expectedDimension}D SwiGLU Oracle Decompiler Sweep`,
    '',
    `- pass: \`${String(aggregate.summary.pass)}\``,
    `- cases: \`${aggregate.summary.passedCases}/${aggregate.summary.caseCount}\``,
    `- min R2: \`${aggregate.summary.minR2}\``,
    `- max MAE: \`${aggregate.summary.maxMae}\``,
    `- max worst error: \`${aggregate.summary.maxWorstError}\``,
    `- min subspace principal cosine: \`${aggregate.summary.minSubspacePrincipalCosine}\``,
    `- min post-hoc paired atom joint cosine: \`${aggregate.summary.minPairedAtomJointCosine}\``,
    `- min paired atoms above threshold: \`${aggregate.summary.minPairedAtomCountAboveThreshold}\``,
    `- min source cancellation fraction: \`${aggregate.summary.minSourceCancellationFraction}\``,
    `- minimum source pair correlation: \`${aggregate.summary.minimumSourcePairCorrelation}\``,
    `- max unit count: \`${aggregate.summary.maxUnitCount}\``,
    `- max atom count: \`${aggregate.summary.maxAtomCount}\``,
    `- total raw oracle calls: \`${aggregate.summary.totalRawOracleCalls}\``,
    '',
    'Access audit:',
    '',
    '- solverAccess: `oracle-only`',
    '- weightsUsedBySolver: `false`',
    '- hooksUsedBySolver: `false`',
    '- postHocWitnessOnly: `true`',
    '',
    ...caseResults.map(result => `- ${result.case.id}: \`${result.pass ? 'pass' : `fail ${result.failures.join('; ')}`}\` (${result.reportPath})`),
    '',
    'Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.',
    ''
  ].join('\n'),
  'utf8'
);
writeBenchmarkManifest(repoRoot, aggregateOutput, {
  lane,
  status: aggregate.summary.pass ? 'pass' : 'fail',
  summary: aggregate.summary,
  notes: aggregate.claimBoundary
});

console.log(JSON.stringify(aggregate.summary, null, 2));
if (!aggregate.summary.pass) process.exit(1);
