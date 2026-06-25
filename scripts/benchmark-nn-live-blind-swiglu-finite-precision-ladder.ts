#!/usr/bin/env ts-node

import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

type LadderInput = {
  id: string;
  label: string;
  caseType: 'selected' | 'randomized';
  quantizationStep: number;
  variant: string;
  note: string;
};

type LadderRow = {
  id: string;
  label: string;
  caseType: 'selected' | 'randomized';
  quantizationStep: number;
  variant: string;
  reportPath: string;
  pass: boolean;
  caseCount: number;
  passedCases: number;
  r2: number;
  mae: number;
  worstError: number;
  subspaceMinPrincipalCosine: number;
  rawOracleCallsPerCase: number;
  totalRawOracleCalls: number;
  unitCount: number;
  atomCount: number;
  accessInvariant: boolean;
  deltaVsCleanSelected?: {
    r2: number;
    mae: number;
    worstError: number;
    subspaceMinPrincipalCosine: number;
  };
  note: string;
};

const repoRoot = path.resolve(__dirname, '..');
const sourceLane = 'nn-live-blind-swiglu-oracle-decompiler';
const lane = 'nn-live-blind-swiglu-finite-precision-ladder';
const defaultVariant = 'mistral-4096d-8unit-selected-randomized-ladder';

const inputs: LadderInput[] = [
  {
    id: 'clean-selected',
    label: 'clean selected',
    caseType: 'selected',
    quantizationStep: 0,
    variant: 'mistral-4096d-layer20-8atom-forward-clean-centered-gradfit-oracle-only',
    note: 'clean selected 8-unit Mistral scalar oracle baseline'
  },
  {
    id: 'q1e-8-selected',
    label: 'q=1e-8 selected',
    caseType: 'selected',
    quantizationStep: 1e-8,
    variant: 'mistral-4096d-layer20-8atom-forward-q00000001-centered-gradfit-oracle-only',
    note: 'selected 8-unit oracle with deterministic scalar response quantization step 1e-8'
  },
  {
    id: 'q1e-8-randomized',
    label: 'q=1e-8 randomized',
    caseType: 'randomized',
    quantizationStep: 1e-8,
    variant: 'mistral-4096d-randomized-2block-8unit-q00000001-centered-gradfit-sweep-oracle-only',
    note: 'two deterministic-random 8-unit Mistral scalar oracles with quantization step 1e-8'
  },
  {
    id: 'q1e-7-selected',
    label: 'q=1e-7 selected',
    caseType: 'selected',
    quantizationStep: 1e-7,
    variant: 'mistral-4096d-layer20-8atom-forward-q0000001-centered-gradfit-step008-probe25-oracle-only',
    note: 'selected 8-unit oracle with 25 probes and quantization step 1e-7'
  },
  {
    id: 'q1e-7-randomized',
    label: 'q=1e-7 randomized',
    caseType: 'randomized',
    quantizationStep: 1e-7,
    variant: 'mistral-4096d-randomized-2block-8unit-q0000001-centered-gradfit-probe25-sweep-oracle-only',
    note: 'two deterministic-random 8-unit Mistral scalar oracles with 25 probes and quantization step 1e-7'
  },
  {
    id: 'q1e-6-selected',
    label: 'q=1e-6 selected',
    caseType: 'selected',
    quantizationStep: 1e-6,
    variant: 'mistral-4096d-layer20-8atom-forward-q000001-centered-gradfit-multistep0812-probe25-oracle-only',
    note: 'selected 8-unit oracle with multi-step finite differences 0.08,0.12 and quantization step 1e-6'
  },
  {
    id: 'q1e-6-randomized',
    label: 'q=1e-6 randomized',
    caseType: 'randomized',
    quantizationStep: 1e-6,
    variant: 'mistral-4096d-randomized-2block-8unit-q000001-centered-gradfit-multistep0812-probe33-grad0075-sweep-oracle-only',
    note: 'two deterministic-random 8-unit Mistral scalar oracles with 33 probes, multi-step finite differences, and quantization step 1e-6'
  }
];

function asRecord(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function asNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function asBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`);
  return value;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function loadReport(input: LadderInput): JsonRecord {
  const reportPath = sourceReportPath(input.variant);
  if (!fs.existsSync(reportPath)) {
    throw new Error(`Missing source report for ${input.id}: ${path.relative(repoRoot, reportPath)}`);
  }
  return asRecord(JSON.parse(fs.readFileSync(reportPath, 'utf8')), `report ${input.id}`);
}

function sourceReportPath(variant: string): string {
  return path.join(repoRoot, 'benchmarks', 'release', sourceLane, variant, 'report.json');
}

function accessInvariant(report: JsonRecord, summary: JsonRecord): boolean {
  const aggregateAccess = summary.accessInvariant;
  if (typeof aggregateAccess === 'boolean') return aggregateAccess;

  const audit = report.accessAudit;
  if (!audit || typeof audit !== 'object' || Array.isArray(audit)) return false;
  const access = audit as JsonRecord;
  return access.solverAccess === 'oracle-only'
    && access.weightsUsedBySolver === false
    && access.hooksUsedBySolver === false
    && access.sourceIdsUsedBySolver === false
    && access.postHocWitnessOnly === true;
}

function selectedRow(input: LadderInput, report: JsonRecord): LadderRow {
  const summary = asRecord(report.summary, `${input.id}.summary`);
  const metrics = asRecord(summary.metrics, `${input.id}.summary.metrics`);
  const sourceAlignment = asRecord(summary.sourceAlignment, `${input.id}.summary.sourceAlignment`);
  return {
    id: input.id,
    label: input.label,
    caseType: input.caseType,
    quantizationStep: input.quantizationStep,
    variant: input.variant,
    reportPath: path.relative(repoRoot, sourceReportPath(input.variant)),
    pass: asBoolean(summary.pass, `${input.id}.summary.pass`),
    caseCount: 1,
    passedCases: asBoolean(summary.pass, `${input.id}.summary.pass`) ? 1 : 0,
    r2: asNumber(metrics.r2, `${input.id}.summary.metrics.r2`),
    mae: asNumber(metrics.mae, `${input.id}.summary.metrics.mae`),
    worstError: asNumber(metrics.worstError, `${input.id}.summary.metrics.worstError`),
    subspaceMinPrincipalCosine: asNumber(
      sourceAlignment.subspaceMinPrincipalCosine,
      `${input.id}.summary.sourceAlignment.subspaceMinPrincipalCosine`
    ),
    rawOracleCallsPerCase: asNumber(summary.rawOracleCalls, `${input.id}.summary.rawOracleCalls`),
    totalRawOracleCalls: asNumber(summary.rawOracleCalls, `${input.id}.summary.rawOracleCalls`),
    unitCount: optionalNumber(summary.unitCount) ?? 1,
    atomCount: optionalNumber(summary.atomCount) ?? 1,
    accessInvariant: accessInvariant(report, summary),
    note: input.note
  };
}

function randomizedRow(input: LadderInput, report: JsonRecord): LadderRow {
  const summary = asRecord(report.summary, `${input.id}.summary`);
  return {
    id: input.id,
    label: input.label,
    caseType: input.caseType,
    quantizationStep: input.quantizationStep,
    variant: input.variant,
    reportPath: path.relative(repoRoot, sourceReportPath(input.variant)),
    pass: asBoolean(summary.pass, `${input.id}.summary.pass`),
    caseCount: asNumber(summary.caseCount, `${input.id}.summary.caseCount`),
    passedCases: asNumber(summary.passedCases, `${input.id}.summary.passedCases`),
    r2: asNumber(summary.minR2, `${input.id}.summary.minR2`),
    mae: asNumber(summary.maxMae, `${input.id}.summary.maxMae`),
    worstError: asNumber(summary.maxWorstError, `${input.id}.summary.maxWorstError`),
    subspaceMinPrincipalCosine: asNumber(
      summary.minSubspacePrincipalCosine,
      `${input.id}.summary.minSubspacePrincipalCosine`
    ),
    rawOracleCallsPerCase: asNumber(summary.maxRawOracleCallsPerCase, `${input.id}.summary.maxRawOracleCallsPerCase`),
    totalRawOracleCalls: asNumber(summary.totalRawOracleCalls, `${input.id}.summary.totalRawOracleCalls`),
    unitCount: asNumber(summary.maxUnitCount, `${input.id}.summary.maxUnitCount`),
    atomCount: asNumber(summary.maxAtomCount, `${input.id}.summary.maxAtomCount`),
    accessInvariant: accessInvariant(report, summary),
    note: input.note
  };
}

function buildRow(input: LadderInput): LadderRow {
  const report = loadReport(input);
  return input.caseType === 'selected'
    ? selectedRow(input, report)
    : randomizedRow(input, report);
}

function withSelectedDeltas(rows: LadderRow[]): LadderRow[] {
  const baseline = rows.find(row => row.id === 'clean-selected');
  if (!baseline) throw new Error('clean-selected baseline row is required');
  return rows.map(row => {
    if (row.caseType !== 'selected' || row.id === baseline.id) return row;
    return {
      ...row,
      deltaVsCleanSelected: {
        r2: row.r2 - baseline.r2,
        mae: row.mae - baseline.mae,
        worstError: row.worstError - baseline.worstError,
        subspaceMinPrincipalCosine: row.subspaceMinPrincipalCosine - baseline.subspaceMinPrincipalCosine
      }
    };
  });
}

function fmt(value: number): string {
  if (value === 0) return '0';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 1e-4) return value.toExponential(6);
  return value.toFixed(12).replace(/0+$/, '').replace(/\.$/, '');
}

function markdownTable(rows: LadderRow[]): string {
  const header = [
    '| case | type | q | pass | R2 | MAE | worst | subspace cosine | raw calls | delta vs clean selected |',
    '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |'
  ];
  const body = rows.map(row => {
    const delta = row.deltaVsCleanSelected
      ? `R2 ${fmt(row.deltaVsCleanSelected.r2)}, MAE ${fmt(row.deltaVsCleanSelected.mae)}, worst ${fmt(row.deltaVsCleanSelected.worstError)}, subspace ${fmt(row.deltaVsCleanSelected.subspaceMinPrincipalCosine)}`
      : row.id === 'clean-selected'
        ? 'baseline'
        : 'not comparable to selected baseline';
    return [
      row.label,
      row.caseType,
      fmt(row.quantizationStep),
      String(row.pass),
      fmt(row.r2),
      fmt(row.mae),
      fmt(row.worstError),
      fmt(row.subspaceMinPrincipalCosine),
      String(row.totalRawOracleCalls),
      delta
    ].join(' | ');
  }).map(line => `| ${line} |`);
  return [...header, ...body].join('\n');
}

const rows = withSelectedDeltas(inputs.map(buildRow));
const failures: string[] = [];
for (const row of rows) {
  if (!row.pass) failures.push(`${row.id}: pass=false`);
  if (row.caseType === 'randomized' && row.passedCases !== row.caseCount) {
    failures.push(`${row.id}: passedCases=${row.passedCases}/${row.caseCount}`);
  }
  if (!row.accessInvariant) failures.push(`${row.id}: accessInvariant=false`);
  if (row.subspaceMinPrincipalCosine < 0.999) {
    failures.push(`${row.id}: subspaceMinPrincipalCosine=${row.subspaceMinPrincipalCosine}<0.999`);
  }
  if (row.r2 < 0.995) failures.push(`${row.id}: r2=${row.r2}<0.995`);
}

const selectedRows = rows.filter(row => row.caseType === 'selected');
const randomizedRows = rows.filter(row => row.caseType === 'randomized');
const summary = {
  pass: failures.length === 0,
  rowCount: rows.length,
  selectedCount: selectedRows.length,
  randomizedCount: randomizedRows.length,
  minR2: Math.min(...rows.map(row => row.r2)),
  maxMae: Math.max(...rows.map(row => row.mae)),
  maxWorstError: Math.max(...rows.map(row => row.worstError)),
  minSubspacePrincipalCosine: Math.min(...rows.map(row => row.subspaceMinPrincipalCosine)),
  maxTotalRawOracleCalls: Math.max(...rows.map(row => row.totalRawOracleCalls)),
  selectedDegradationVsClean: selectedRows
    .filter(row => row.deltaVsCleanSelected)
    .map(row => ({
      id: row.id,
      quantizationStep: row.quantizationStep,
      deltaVsCleanSelected: row.deltaVsCleanSelected
    })),
  randomizedNoiseLadder: randomizedRows.map(row => ({
    id: row.id,
    quantizationStep: row.quantizationStep,
    minR2: row.r2,
    maxMae: row.mae,
    maxWorstError: row.worstError,
    minSubspacePrincipalCosine: row.subspaceMinPrincipalCosine,
    totalRawOracleCalls: row.totalRawOracleCalls
  })),
  accessInvariant: rows.every(row => row.accessInvariant),
  failures
};

const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  lane,
  sourceLane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  rows,
  boundaries: [
    'This is an artifact-backed finite-precision geometry audit, not a new solver tuning result.',
    '4096D is the Mistral hidden/input dimension observed through scalar oracle calls.',
    '8-unit is the recovered selected/randomized SwiGLU block size; this does not prove full-FFN or all-channel extraction.',
    'The solver access invariant remains oracle-only for the source reports; weights/hooks are post-hoc witnesses only.',
    'Randomized rows are deterministic two-case sweeps and are not directly delta-comparable to the selected clean baseline.'
  ]
};

const output = resolveBenchmarkOutput(repoRoot, lane, defaultVariant);
fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# NN Live Blind SwiGLU Finite-Precision Ladder',
    '',
    `Status: **${summary.pass ? 'PASS' : 'FAIL'}**`,
    '',
    'This report reads existing real-model oracle-only Mistral SwiGLU decompiler artifacts and centralizes finite-precision geometry degradation. It does not retune the solver.',
    '',
    markdownTable(rows),
    '',
    '## Summary',
    '',
    `- rows: ${summary.rowCount}`,
    `- min R2: ${fmt(summary.minR2)}`,
    `- max MAE: ${fmt(summary.maxMae)}`,
    `- max worst error: ${fmt(summary.maxWorstError)}`,
    `- min subspace principal cosine: ${fmt(summary.minSubspacePrincipalCosine)}`,
    `- max total raw oracle calls: ${summary.maxTotalRawOracleCalls}`,
    `- access invariant: ${summary.accessInvariant}`,
    '',
    '## Boundaries',
    '',
    ...report.boundaries.map(item => `- ${item}`),
    '',
    failures.length ? `## Failures\n\n${failures.map(item => `- ${item}`).join('\n')}\n` : ''
  ].join('\n')
);
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'fail',
  summary,
  notes: report.boundaries
});

console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
