#!/usr/bin/env ts-node

import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

type ParityInput = {
  id: string;
  modelFamily: 'Mistral-family' | 'LLaMA-family';
  modelId: string;
  dimension: number;
  caseType: 'selected' | 'randomized';
  quantizationStep: number;
  variant: string;
  note: string;
};

type ParityRow = ParityInput & {
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
};

const repoRoot = path.resolve(__dirname, '..');
const sourceLane = 'nn-live-blind-swiglu-oracle-decompiler';
const lane = 'nn-live-blind-swiglu-real-model-parity';
const defaultVariant = 'mistral-4096d-llama-3072d-8unit-oracle-only';

const inputs: ParityInput[] = [
  {
    id: 'mistral-clean-selected-8unit',
    modelFamily: 'Mistral-family',
    modelId: 'local Mistral-family checkpoint',
    dimension: 4096,
    caseType: 'selected',
    quantizationStep: 0,
    variant: 'mistral-4096d-layer20-8atom-forward-clean-centered-gradfit-oracle-only',
    note: 'selected real-Mistral 4096D 8-unit scalar SwiGLU oracle'
  },
  {
    id: 'mistral-clean-randomized-8unit',
    modelFamily: 'Mistral-family',
    modelId: 'local Mistral-family checkpoint',
    dimension: 4096,
    caseType: 'randomized',
    quantizationStep: 0,
    variant: 'mistral-4096d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only',
    note: 'two deterministic-random real-Mistral 4096D 8-unit scalar SwiGLU oracles'
  },
  {
    id: 'mistral-q1e-6-randomized-8unit',
    modelFamily: 'Mistral-family',
    modelId: 'local Mistral-family checkpoint',
    dimension: 4096,
    caseType: 'randomized',
    quantizationStep: 1e-6,
    variant: 'mistral-4096d-randomized-2block-8unit-q000001-centered-gradfit-multistep0812-probe33-grad0075-sweep-oracle-only',
    note: 'two deterministic-random finite-precision real-Mistral 4096D 8-unit scalar SwiGLU oracles'
  },
  {
    id: 'llama-clean-selected-8unit',
    modelFamily: 'LLaMA-family',
    modelId: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    dimension: 3072,
    caseType: 'selected',
    quantizationStep: 0,
    variant: 'llama-3072d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-budgeted-oracle-only',
    note: 'selected real-LLaMA-family 3072D 8-unit scalar SwiGLU oracle'
  },
  {
    id: 'llama-clean-randomized-8unit',
    modelFamily: 'LLaMA-family',
    modelId: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    dimension: 3072,
    caseType: 'randomized',
    quantizationStep: 0,
    variant: 'llama-3072d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only',
    note: 'two deterministic-random real-LLaMA-family 3072D 8-unit scalar SwiGLU oracles'
  },
  {
    id: 'llama-q2p5e-4-selected-8unit',
    modelFamily: 'LLaMA-family',
    modelId: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    dimension: 3072,
    caseType: 'selected',
    quantizationStep: 0.00025,
    variant: 'llama-3072d-layer20-8atom-forward-q00025-nogradfit-oracle-only',
    note: 'selected finite-precision real-LLaMA-family 3072D 8-unit scalar SwiGLU oracle'
  },
  {
    id: 'llama-q2p5e-4-randomized-8unit',
    modelFamily: 'LLaMA-family',
    modelId: 'unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit',
    dimension: 3072,
    caseType: 'randomized',
    quantizationStep: 0.00025,
    variant: 'llama-3072d-randomized-2block-8unit-q00025-forward-sweep-oracle-only',
    note: 'two deterministic-random finite-precision real-LLaMA-family 3072D 8-unit scalar SwiGLU oracles'
  }
];

function asRecord(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function asNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be a finite number`);
  return value;
}

function asBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`);
  return value;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function sourceReportPath(variant: string): string {
  return path.join(repoRoot, 'benchmarks', 'release', sourceLane, variant, 'report.json');
}

function loadReport(input: ParityInput): JsonRecord {
  const reportPath = sourceReportPath(input.variant);
  if (!fs.existsSync(reportPath)) throw new Error(`Missing source report for ${input.id}: ${path.relative(repoRoot, reportPath)}`);
  return asRecord(JSON.parse(fs.readFileSync(reportPath, 'utf8')), `report ${input.id}`);
}

function accessInvariant(report: JsonRecord, summary: JsonRecord): boolean {
  if (typeof summary.accessInvariant === 'boolean') return summary.accessInvariant;
  const audit = report.accessAudit;
  if (!audit || typeof audit !== 'object' || Array.isArray(audit)) return false;
  const access = audit as JsonRecord;
  return access.solverAccess === 'oracle-only'
    && access.weightsUsedBySolver === false
    && access.hooksUsedBySolver === false
    && access.sourceIdsUsedBySolver === false
    && access.postHocWitnessOnly === true;
}

function assertModelWitness(input: ParityInput, summary: JsonRecord): string[] {
  // Every check here used to be skipped when its field was absent, so a source report
  // carrying no dimension, modelType or model satisfied all three. This gate's stated job
  // is to separate real 4096D Mistral and 3072D LLaMA parity from synthetic teacher
  // claims, which makes an identity witness that degrades to a pass on absence the one
  // thing it must not do. A missing witness is now a failure, and a wrong witness stays a
  // failure. Dimension is required unconditionally; the model identity fields are required
  // for the family whose identity this gate asserts.
  const failures: string[] = [];
  const observedDimension = optionalNumber(summary.dimension);
  if (observedDimension === undefined) {
    failures.push(`${input.id}: source report carries no dimension witness`);
  } else if (observedDimension !== input.dimension) {
    failures.push(`${input.id}: dimension=${observedDimension}<expected ${input.dimension}`);
  }
  if (input.modelFamily === 'LLaMA-family') {
    const observedModelType = summary.modelType;
    if (observedModelType === undefined) {
      failures.push(`${input.id}: source report carries no modelType witness`);
    } else if (observedModelType !== 'llama') {
      failures.push(`${input.id}: modelType=${String(observedModelType)}<expected llama`);
    }
    const observedModel = summary.model;
    if (observedModel === undefined) {
      failures.push(`${input.id}: source report carries no model identity witness`);
    } else if (observedModel !== input.modelId) {
      failures.push(`${input.id}: model=${String(observedModel)}<expected ${input.modelId}`);
    }
  }
  return failures;
}

function selectedRow(input: ParityInput, report: JsonRecord): { row: ParityRow; failures: string[] } {
  const summary = asRecord(report.summary, `${input.id}.summary`);
  const metrics = asRecord(summary.metrics, `${input.id}.summary.metrics`);
  const sourceAlignment = asRecord(summary.sourceAlignment, `${input.id}.summary.sourceAlignment`);
  const row: ParityRow = {
    ...input,
    reportPath: path.relative(repoRoot, sourceReportPath(input.variant)),
    pass: asBoolean(summary.pass, `${input.id}.summary.pass`),
    caseCount: 1,
    passedCases: asBoolean(summary.pass, `${input.id}.summary.pass`) ? 1 : 0,
    r2: asNumber(metrics.r2, `${input.id}.summary.metrics.r2`),
    mae: asNumber(metrics.mae, `${input.id}.summary.metrics.mae`),
    worstError: asNumber(metrics.worstError, `${input.id}.summary.metrics.worstError`),
    subspaceMinPrincipalCosine: asNumber(sourceAlignment.subspaceMinPrincipalCosine, `${input.id}.summary.sourceAlignment.subspaceMinPrincipalCosine`),
    rawOracleCallsPerCase: asNumber(summary.rawOracleCalls, `${input.id}.summary.rawOracleCalls`),
    totalRawOracleCalls: asNumber(summary.rawOracleCalls, `${input.id}.summary.rawOracleCalls`),
    unitCount: optionalNumber(summary.unitCount) ?? 1,
    atomCount: optionalNumber(summary.atomCount) ?? 1,
    accessInvariant: accessInvariant(report, summary)
  };
  return { row, failures: assertModelWitness(input, summary) };
}

function randomizedRow(input: ParityInput, report: JsonRecord): { row: ParityRow; failures: string[] } {
  const summary = asRecord(report.summary, `${input.id}.summary`);
  const row: ParityRow = {
    ...input,
    reportPath: path.relative(repoRoot, sourceReportPath(input.variant)),
    pass: asBoolean(summary.pass, `${input.id}.summary.pass`),
    caseCount: asNumber(summary.caseCount, `${input.id}.summary.caseCount`),
    passedCases: asNumber(summary.passedCases, `${input.id}.summary.passedCases`),
    r2: asNumber(summary.minR2, `${input.id}.summary.minR2`),
    mae: asNumber(summary.maxMae, `${input.id}.summary.maxMae`),
    worstError: asNumber(summary.maxWorstError, `${input.id}.summary.maxWorstError`),
    subspaceMinPrincipalCosine: asNumber(summary.minSubspacePrincipalCosine, `${input.id}.summary.minSubspacePrincipalCosine`),
    rawOracleCallsPerCase: asNumber(summary.maxRawOracleCallsPerCase, `${input.id}.summary.maxRawOracleCallsPerCase`),
    totalRawOracleCalls: asNumber(summary.totalRawOracleCalls, `${input.id}.summary.totalRawOracleCalls`),
    unitCount: asNumber(summary.maxUnitCount, `${input.id}.summary.maxUnitCount`),
    atomCount: asNumber(summary.maxAtomCount, `${input.id}.summary.maxAtomCount`),
    accessInvariant: accessInvariant(report, summary)
  };
  // An aggregate sweep does not carry the model identity at the top level; each case does,
  // under cases[].summary. Reading only the top level meant the witness was always absent
  // here, which the old optional checks silently accepted. Every case must now carry the
  // witness and every case must agree, so a sweep that mixed families or dimensions is
  // caught rather than averaged away.
  const cases = Array.isArray(report.cases) ? report.cases : [];
  if (cases.length === 0) {
    return { row, failures: [`${input.id}: aggregate report carries no cases to witness`] };
  }
  const failures = cases.flatMap((entry, index) => {
    const caseRecord = asRecord(entry, `${input.id}.cases[${index}]`);
    const caseSummary = asRecord(caseRecord.summary, `${input.id}.cases[${index}].summary`);
    return assertModelWitness({ ...input, id: `${input.id}.cases[${index}]` }, caseSummary);
  });
  return { row, failures };
}

function buildRow(input: ParityInput): { row: ParityRow; failures: string[] } {
  const report = loadReport(input);
  return input.caseType === 'selected' ? selectedRow(input, report) : randomizedRow(input, report);
}

function fmt(value: number): string {
  if (value === 0) return '0';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 1e-4) return value.toExponential(6);
  return value.toFixed(12).replace(/0+$/, '').replace(/\.$/, '');
}

function markdownTable(rows: ParityRow[]): string {
  const header = [
    '| case | model family | dim | type | q | pass | R2 | MAE | worst | subspace cosine | raw calls |',
    '| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |'
  ];
  const body = rows.map(row => [
    row.id,
    row.modelFamily,
    String(row.dimension),
    row.caseType,
    fmt(row.quantizationStep),
    String(row.pass),
    fmt(row.r2),
    fmt(row.mae),
    fmt(row.worstError),
    fmt(row.subspaceMinPrincipalCosine),
    String(row.totalRawOracleCalls)
  ].join(' | ')).map(line => `| ${line} |`);
  return [...header, ...body].join('\n');
}

const built = inputs.map(buildRow);
const rows = built.map(item => item.row);
const failures = built.flatMap(item => item.failures);

for (const row of rows) {
  if (!row.pass) failures.push(`${row.id}: pass=false`);
  if (row.caseType === 'randomized' && row.passedCases !== row.caseCount) {
    failures.push(`${row.id}: passedCases=${row.passedCases}/${row.caseCount}`);
  }
  if (!row.accessInvariant) failures.push(`${row.id}: accessInvariant=false`);
  if (row.r2 < 0.995) failures.push(`${row.id}: r2=${row.r2}<0.995`);
  if (row.subspaceMinPrincipalCosine < 0.999) {
    failures.push(`${row.id}: subspaceMinPrincipalCosine=${row.subspaceMinPrincipalCosine}<0.999`);
  }
}

const summary = {
  pass: failures.length === 0,
  rowCount: rows.length,
  modelFamilies: [...new Set(rows.map(row => row.modelFamily))].sort(),
  dimensions: [...new Set(rows.map(row => row.dimension))].sort((a, b) => a - b),
  selectedCount: rows.filter(row => row.caseType === 'selected').length,
  randomizedCount: rows.filter(row => row.caseType === 'randomized').length,
  finitePrecisionCount: rows.filter(row => row.quantizationStep > 0).length,
  minR2: Math.min(...rows.map(row => row.r2)),
  maxMae: Math.max(...rows.map(row => row.mae)),
  maxWorstError: Math.max(...rows.map(row => row.worstError)),
  minSubspacePrincipalCosine: Math.min(...rows.map(row => row.subspaceMinPrincipalCosine)),
  maxTotalRawOracleCalls: Math.max(...rows.map(row => row.totalRawOracleCalls)),
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
    'This is a real-model parity evidence ledger, not a new decompiler method.',
    'Mistral-family rows use a local Mistral-family checkpoint artifact; LLaMA-family rows use unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit artifacts.',
    'Rows prove selected/randomized scalar 8-unit SwiGLU block recovery only; they do not prove all-channel or full-FFN extraction.',
    'Solver access remains oracle-only in the source reports; source weights/hooks are post-hoc witnesses only.'
  ]
};

const output = resolveBenchmarkOutput(repoRoot, lane, defaultVariant);
fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# NN Live Blind SwiGLU Real-Model Parity',
    '',
    `Status: **${summary.pass ? 'PASS' : 'FAIL'}**`,
    '',
    'This report reads existing real-model oracle-only Mistral and LLaMA-family SwiGLU decompiler artifacts. It separates real model parity from synthetic teacher claims.',
    '',
    markdownTable(rows),
    '',
    '## Summary',
    '',
    `- model families: ${summary.modelFamilies.join(', ')}`,
    `- dimensions: ${summary.dimensions.join(', ')}`,
    `- rows: ${summary.rowCount}`,
    `- min R2: ${fmt(summary.minR2)}`,
    `- max MAE: ${fmt(summary.maxMae)}`,
    `- max worst error: ${fmt(summary.maxWorstError)}`,
    `- min subspace principal cosine: ${fmt(summary.minSubspacePrincipalCosine)}`,
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
