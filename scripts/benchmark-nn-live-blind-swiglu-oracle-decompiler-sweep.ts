#!/usr/bin/env ts-node

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type SweepCase = {
  id: string;
  layer: number;
  unit: number;
  minR2: number;
  maxMae: number;
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
    activeRank: number;
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

function resolveGclPython(): string {
  if (process.env.GCL_PYTHON) return process.env.GCL_PYTHON;
  if (process.env.GCL_LLM_READER_PYTHON) return process.env.GCL_LLM_READER_PYTHON;
  const gpuPython = path.join(repoRoot, '.gcl-python-gpu', 'bin', 'python');
  if (fs.existsSync(gpuPython)) return gpuPython;
  return 'python3';
}

function caseVariant(testCase: SweepCase): string {
  return `mistral-4096d-layer${testCase.layer}-unit${testCase.unit}-oracle-only`;
}

function parseCases(raw: string): SweepCase[] {
  return raw
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
    .map(item => {
      const [layerRaw, unitRaw, minR2Raw, maxMaeRaw, minSubspaceRaw] = item.split(':');
      const layer = Number(layerRaw);
      const unit = Number(unitRaw);
      if (!Number.isInteger(layer) || !Number.isInteger(unit)) {
        throw new Error(`Invalid blind SwiGLU sweep case ${item}; expected layer:unit[:minR2[:maxMae[:minSubspaceCosine]]]`);
      }
      return {
        id: `layer${layer}-unit${unit}`,
        layer,
        unit,
        minR2: minR2Raw ? Number(minR2Raw) : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_R2', '0.995')),
        maxMae: maxMaeRaw ? Number(maxMaeRaw) : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MAX_MAE', '0.002')),
        minSubspaceCosine: minSubspaceRaw
          ? Number(minSubspaceRaw)
          : Number(envValue('GCL_BLIND_SWIGLU_SWEEP_MIN_SUBSPACE_COSINE', '0.99'))
      };
    });
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
  if (report.summary.dimension !== 4096) failures.push(`dimension=${report.summary.dimension}`);
  if (report.summary.layer !== testCase.layer) failures.push(`layer=${report.summary.layer}`);
  if (report.summary.unit !== testCase.unit) failures.push(`unit=${report.summary.unit}`);
  if (report.summary.metrics.r2 < testCase.minR2) failures.push(`r2=${report.summary.metrics.r2}<${testCase.minR2}`);
  if (report.summary.metrics.mae > testCase.maxMae) failures.push(`mae=${report.summary.metrics.mae}>${testCase.maxMae}`);
  if (report.summary.sourceAlignment.subspaceMinPrincipalCosine < testCase.minSubspaceCosine) {
    failures.push(`subspace=${report.summary.sourceAlignment.subspaceMinPrincipalCosine}<${testCase.minSubspaceCosine}`);
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

const sweepCases = parseCases(envValue(
  'GCL_BLIND_SWIGLU_SWEEP_CASES',
  '8:123:0.995:0.002:0.99,16:2048:0.995:0.002:0.99,20:0:0.995:0.002:0.99,24:17:0.995:0.002:0.99'
));
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
  const args = [
    toolPath,
    '--model', envValue('GCL_BLIND_SWIGLU_MODEL', 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'),
    '--layer', String(testCase.layer),
    '--unit', String(testCase.unit),
    '--device', envValue('GCL_BLIND_SWIGLU_DEVICE', 'cuda'),
    '--gradient-mode', envValue('GCL_BLIND_SWIGLU_GRADIENT_MODE', 'coordinate'),
    '--probe-points', envValue('GCL_BLIND_SWIGLU_PROBE_POINTS', '2'),
    '--direction-count', envValue('GCL_BLIND_SWIGLU_DIRECTION_COUNT', '384'),
    '--train-count', envValue('GCL_BLIND_SWIGLU_TRAIN_COUNT', '1600'),
    '--validation-count', envValue('GCL_BLIND_SWIGLU_VALIDATION_COUNT', '512'),
    '--fit-steps', envValue('GCL_BLIND_SWIGLU_FIT_STEPS', '1200'),
    '--min-r2', String(testCase.minR2),
    '--max-mae', String(testCase.maxMae),
    '--min-subspace-cosine', String(testCase.minSubspaceCosine),
    '--max-oracle-calls', envValue('GCL_BLIND_SWIGLU_MAX_ORACLE_CALLS', '20000'),
    '--local-files-only',
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

const aggregateVariant = envValue('GCL_BLIND_SWIGLU_SWEEP_VARIANT', 'mistral-4096d-4unit-layer-sweep-2probe-oracle-only');
const aggregateOutput = resolveOutputForVariant(aggregateVariant);
fs.mkdirSync(aggregateOutput.outputDir, { recursive: true });

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
    maxRawOracleCallsPerCase: Math.max(...summaries.map(summary => summary.rawOracleCalls)),
    totalRawOracleCalls: summaries.reduce((sum, summary) => sum + summary.rawOracleCalls, 0),
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
  claimBoundary: [
    'This sweep proves selected real local Mistral-family 4096D SwiGLU units can be recovered as blind oracle-only explicit rank-2 SwiGLU atoms under the recorded case budgets.',
    'It does not prove full Mistral FFN extraction, all-channel 4096D decompilation, exhaustive LLM circuit recovery, or semantic original-neuron identity from output-only access.'
  ]
};

fs.writeFileSync(aggregateOutput.jsonOutputPath, `${JSON.stringify(aggregate, null, 2)}\n`);
fs.writeFileSync(
  aggregateOutput.markdownOutputPath,
  [
    '# Blind 4096D SwiGLU Oracle Decompiler Sweep',
    '',
    `- pass: \`${String(aggregate.summary.pass)}\``,
    `- cases: \`${aggregate.summary.passedCases}/${aggregate.summary.caseCount}\``,
    `- min R2: \`${aggregate.summary.minR2}\``,
    `- max MAE: \`${aggregate.summary.maxMae}\``,
    `- max worst error: \`${aggregate.summary.maxWorstError}\``,
    `- min subspace principal cosine: \`${aggregate.summary.minSubspacePrincipalCosine}\``,
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
