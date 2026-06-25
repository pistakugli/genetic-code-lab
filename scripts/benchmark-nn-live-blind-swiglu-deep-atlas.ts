#!/usr/bin/env ts-node

import fs from 'fs';
import path from 'path';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type JsonRecord = Record<string, unknown>;

type AtlasRow = {
  id: string;
  variant: string;
  reportPath: string;
  modelFamily: 'Mistral-family' | 'LLaMA-family' | 'unknown';
  modelId: string;
  dimension: number | null;
  layer: number | null;
  units: number[];
  unitCount: number;
  quantizationStep: number;
  r2: number;
  mae: number;
  worstError: number;
  subspaceMinPrincipalCosine: number;
  rawOracleCalls: number;
  sourceKind: 'selected' | 'aggregate';
  accessInvariant: boolean;
};

type LayerAtlas = {
  modelFamily: AtlasRow['modelFamily'];
  layer: number;
  rowCount: number;
  bestR2: number;
  worstR2: number;
  maxUnitCount: number;
  minSubspacePrincipalCosine: number;
  maxWorstError: number;
  finitePrecisionRows: number;
  variants: string[];
};

const repoRoot = path.resolve(__dirname, '..');
const sourceLane = 'nn-live-blind-swiglu-oracle-decompiler';
const lane = 'nn-live-blind-swiglu-deep-atlas';
const defaultVariant = 'mistral-llama-multilayer-swiglu-atlas';
const sourceDir = path.join(repoRoot, 'benchmarks', 'release', sourceLane);

function asRecord(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as JsonRecord;
}

function optionalNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function numeric(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionalString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function listReports(): string[] {
  if (!fs.existsSync(sourceDir)) throw new Error(`Missing source lane: ${path.relative(repoRoot, sourceDir)}`);
  return fs.readdirSync(sourceDir)
    .map(variant => path.join(sourceDir, variant, 'report.json'))
    .filter(reportPath => fs.existsSync(reportPath))
    .sort();
}

function inferFamily(variant: string, summary: JsonRecord): AtlasRow['modelFamily'] {
  const modelType = optionalString(summary.modelType);
  const model = optionalString(summary.model);
  if (modelType === 'llama' || model.toLowerCase().includes('llama') || variant.startsWith('llama-')) return 'LLaMA-family';
  if (modelType === 'mistral' || model.toLowerCase().includes('mistral') || variant.startsWith('mistral-')) return 'Mistral-family';
  return 'unknown';
}

function readUnits(summary: JsonRecord): number[] {
  if (Array.isArray(summary.units)) {
    return summary.units.filter((value): value is number => Number.isInteger(value));
  }
  const unit = optionalNumber(summary.unit);
  return unit === null ? [] : [unit];
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

function rowFromSelected(reportPath: string, report: JsonRecord): AtlasRow | null {
  const variant = path.basename(path.dirname(reportPath));
  const summary = asRecord(report.summary, `${variant}.summary`);
  const metrics = summary.metrics;
  const alignment = summary.sourceAlignment;
  if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) return null;
  if (!alignment || typeof alignment !== 'object' || Array.isArray(alignment)) return null;
  const metricRecord = metrics as JsonRecord;
  const alignmentRecord = alignment as JsonRecord;
  const r2 = optionalNumber(metricRecord.r2);
  const mae = optionalNumber(metricRecord.mae);
  const worstError = optionalNumber(metricRecord.worstError);
  const subspace = optionalNumber(alignmentRecord.subspaceMinPrincipalCosine);
  if (r2 === null || mae === null || worstError === null || subspace === null) return null;

  const units = readUnits(summary);
  const quantizationStep = numeric(asRecord(report.oracleResponse ?? {}, `${variant}.oracleResponse`).quantizationStep);
  return {
    id: variant,
    variant,
    reportPath: path.relative(repoRoot, reportPath),
    modelFamily: inferFamily(variant, summary),
    modelId: optionalString(summary.model, inferFamily(variant, summary)),
    dimension: optionalNumber(summary.dimension),
    layer: optionalNumber(summary.layer),
    units,
    unitCount: numeric(summary.unitCount, units.length || 1),
    quantizationStep,
    r2,
    mae,
    worstError,
    subspaceMinPrincipalCosine: subspace,
    rawOracleCalls: numeric(summary.rawOracleCalls),
    sourceKind: 'selected',
    accessInvariant: accessInvariant(report, summary)
  };
}

function rowFromAggregate(reportPath: string, report: JsonRecord): AtlasRow | null {
  const variant = path.basename(path.dirname(reportPath));
  const summary = asRecord(report.summary, `${variant}.summary`);
  const minR2 = optionalNumber(summary.minR2);
  const maxMae = optionalNumber(summary.maxMae);
  const maxWorstError = optionalNumber(summary.maxWorstError);
  const minSubspace = optionalNumber(summary.minSubspacePrincipalCosine);
  if (minR2 === null || maxMae === null || maxWorstError === null || minSubspace === null) return null;
  return {
    id: variant,
    variant,
    reportPath: path.relative(repoRoot, reportPath),
    modelFamily: inferFamily(variant, summary),
    modelId: optionalString(summary.model, inferFamily(variant, summary)),
    dimension: optionalNumber(summary.dimension),
    layer: null,
    units: [],
    unitCount: numeric(summary.maxUnitCount, 1),
    quantizationStep: inferQuantizationFromVariant(variant),
    r2: minR2,
    mae: maxMae,
    worstError: maxWorstError,
    subspaceMinPrincipalCosine: minSubspace,
    rawOracleCalls: numeric(summary.totalRawOracleCalls),
    sourceKind: 'aggregate',
    accessInvariant: accessInvariant(report, summary)
  };
}

function inferQuantizationFromVariant(variant: string): number {
  if (variant.includes('q000001') || variant.includes('q1e-6')) return 1e-6;
  if (variant.includes('q0000001') || variant.includes('q1e-7')) return 1e-7;
  if (variant.includes('q00000001') || variant.includes('q1e-8')) return 1e-8;
  if (variant.includes('q00025')) return 0.00025;
  return 0;
}

function buildRows(): AtlasRow[] {
  const rows: AtlasRow[] = [];
  for (const reportPath of listReports()) {
    const report = asRecord(JSON.parse(fs.readFileSync(reportPath, 'utf8')), reportPath);
    const selected = rowFromSelected(reportPath, report);
    if (selected) rows.push(selected);
    else {
      const aggregate = rowFromAggregate(reportPath, report);
      if (aggregate) rows.push(aggregate);
    }
  }
  return rows.filter(row => row.modelFamily !== 'unknown' && row.accessInvariant);
}

function contiguousRuns(layers: number[]): number[][] {
  const sorted = [...new Set(layers)].sort((a, b) => a - b);
  const runs: number[][] = [];
  for (const layer of sorted) {
    const last = runs[runs.length - 1];
    if (!last || layer !== last[last.length - 1] + 1) runs.push([layer]);
    else last.push(layer);
  }
  return runs;
}

function bucketCoverage(layers: number[], layerCount: number): Record<string, number[]> {
  const buckets = {
    early: [] as number[],
    middle: [] as number[],
    late: [] as number[]
  };
  const firstCut = Math.floor(layerCount / 3);
  const secondCut = Math.floor((2 * layerCount) / 3);
  for (const layer of [...new Set(layers)].sort((a, b) => a - b)) {
    if (layer < firstCut) buckets.early.push(layer);
    else if (layer < secondCut) buckets.middle.push(layer);
    else buckets.late.push(layer);
  }
  return buckets;
}

function layerCountForFamily(family: AtlasRow['modelFamily']): number {
  if (family === 'Mistral-family') return 32;
  if (family === 'LLaMA-family') return 28;
  return 0;
}

function layerAtlas(rows: AtlasRow[]): LayerAtlas[] {
  const selected = rows.filter(row => row.layer !== null && row.sourceKind === 'selected');
  const groups = new Map<string, AtlasRow[]>();
  for (const row of selected) {
    const key = `${row.modelFamily}:${row.layer}`;
    const group = groups.get(key) ?? [];
    group.push(row);
    groups.set(key, group);
  }
  return [...groups.entries()].map(([key, group]) => {
    const [family, layerRaw] = key.split(':');
    return {
      modelFamily: family as AtlasRow['modelFamily'],
      layer: Number(layerRaw),
      rowCount: group.length,
      bestR2: Math.max(...group.map(row => row.r2)),
      worstR2: Math.min(...group.map(row => row.r2)),
      maxUnitCount: Math.max(...group.map(row => row.unitCount)),
      minSubspacePrincipalCosine: Math.min(...group.map(row => row.subspaceMinPrincipalCosine)),
      maxWorstError: Math.max(...group.map(row => row.worstError)),
      finitePrecisionRows: group.filter(row => row.quantizationStep > 0).length,
      variants: group.map(row => row.variant).sort()
    };
  }).sort((a, b) => a.modelFamily.localeCompare(b.modelFamily) || a.layer - b.layer);
}

function summarizeFamily(family: AtlasRow['modelFamily'], rows: AtlasRow[], layers: LayerAtlas[]) {
  const familyRows = rows.filter(row => row.modelFamily === family);
  const layerRows = layers.filter(layer => layer.modelFamily === family);
  const uniqueLayers = layerRows.map(row => row.layer).sort((a, b) => a - b);
  const runs = contiguousRuns(uniqueLayers);
  const layerCount = layerCountForFamily(family);
  const buckets = bucketCoverage(uniqueLayers, layerCount);
  const selectedRows = familyRows.filter(row => row.sourceKind === 'selected');
  const multiUnitRows = selectedRows.filter(row => row.unitCount >= 2);
  const eightUnitRows = selectedRows.filter(row => row.unitCount >= 8);
  return {
    modelFamily: family,
    layerCount,
    artifactRows: familyRows.length,
    selectedRows: selectedRows.length,
    aggregateRows: familyRows.filter(row => row.sourceKind === 'aggregate').length,
    multiUnitSelectedRows: multiUnitRows.length,
    eightUnitSelectedRows: eightUnitRows.length,
    finitePrecisionRows: familyRows.filter(row => row.quantizationStep > 0).length,
    uniqueLayerCount: uniqueLayers.length,
    uniqueLayers,
    coverageRatio: layerCount ? uniqueLayers.length / layerCount : 0,
    bucketCoverage: buckets,
    longestContiguousRun: runs.reduce((max, run) => Math.max(max, run.length), 0),
    contiguousRuns: runs,
    minR2: Math.min(...familyRows.map(row => row.r2)),
    maxMae: Math.max(...familyRows.map(row => row.mae)),
    maxWorstError: Math.max(...familyRows.map(row => row.worstError)),
    minSubspacePrincipalCosine: Math.min(...familyRows.map(row => row.subspaceMinPrincipalCosine)),
    maxRawOracleCalls: Math.max(...familyRows.map(row => row.rawOracleCalls))
  };
}

function fmt(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 1e-4) return value.toExponential(6);
  return value.toFixed(12).replace(/0+$/, '').replace(/\.$/, '');
}

function markdownLayerTable(layers: LayerAtlas[]): string {
  const lines = [
    '| family | layer | rows | max units | worst R2 | max worst error | min subspace | finite rows |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'
  ];
  for (const layer of layers) {
    lines.push(`| ${layer.modelFamily} | ${layer.layer} | ${layer.rowCount} | ${layer.maxUnitCount} | ${fmt(layer.worstR2)} | ${fmt(layer.maxWorstError)} | ${fmt(layer.minSubspacePrincipalCosine)} | ${layer.finitePrecisionRows} |`);
  }
  return lines.join('\n');
}

const rows = buildRows();
const layers = layerAtlas(rows);
const families: Array<AtlasRow['modelFamily']> = ['LLaMA-family', 'Mistral-family'];
const familySummaries = families.map(family => summarizeFamily(family, rows, layers));

const failures: string[] = [];
for (const family of familySummaries) {
  if (family.uniqueLayerCount < (family.modelFamily === 'Mistral-family' ? 12 : 10)) {
    failures.push(`${family.modelFamily}: uniqueLayerCount=${family.uniqueLayerCount}`);
  }
  if (family.multiUnitSelectedRows < 6) failures.push(`${family.modelFamily}: multiUnitSelectedRows=${family.multiUnitSelectedRows}`);
  if (!family.bucketCoverage.early.length) failures.push(`${family.modelFamily}: missing early-layer coverage`);
  if (!family.bucketCoverage.middle.length) failures.push(`${family.modelFamily}: missing middle-layer coverage`);
  if (!family.bucketCoverage.late.length) failures.push(`${family.modelFamily}: missing late-layer coverage`);
  if (family.minR2 < 0.995) failures.push(`${family.modelFamily}: minR2=${family.minR2}<0.995`);
  if (family.minSubspacePrincipalCosine < 0.999) {
    failures.push(`${family.modelFamily}: minSubspacePrincipalCosine=${family.minSubspacePrincipalCosine}<0.999`);
  }
}
if (rows.length < 80) failures.push(`totalRows=${rows.length}<80`);

const summary = {
  pass: failures.length === 0,
  sourceReportCount: listReports().length,
  atlasRowCount: rows.length,
  selectedRowCount: rows.filter(row => row.sourceKind === 'selected').length,
  aggregateRowCount: rows.filter(row => row.sourceKind === 'aggregate').length,
  layerAtlasRowCount: layers.length,
  familySummaries,
  global: {
    minR2: Math.min(...rows.map(row => row.r2)),
    maxMae: Math.max(...rows.map(row => row.mae)),
    maxWorstError: Math.max(...rows.map(row => row.worstError)),
    minSubspacePrincipalCosine: Math.min(...rows.map(row => row.subspaceMinPrincipalCosine)),
    maxRawOracleCalls: Math.max(...rows.map(row => row.rawOracleCalls))
  },
  failures
};

const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  lane,
  sourceLane,
  status: summary.pass ? 'pass' : 'needs-backfill',
  summary,
  layerAtlas: layers,
  rows,
  backfillPolicy: {
    ifNeedsBackfill: 'Run or add more oracle-only live SwiGLU decompiler cases for the missing model-family/layer bucket; do not relabel gaps as success.',
    currentBackfillNeeded: failures
  },
  boundaries: [
    'This is a deep multi-layer atlas over existing real-model oracle-only SwiGLU decompiler artifacts.',
    'It is not a claim that the entire LLM brain, every channel, or semantic hidden-neuron identity has been recovered.',
    'The atlas proves multi-layer selected-block recovery coverage for the covered real Mistral/LLaMA-family scalar SwiGLU oracle rows.',
    'If coverage gates fail, the remedy is additional oracle-only recovery/backfill runs, not lowering thresholds or hiding the gap.'
  ]
};

const output = resolveBenchmarkOutput(repoRoot, lane, defaultVariant);
fs.mkdirSync(output.outputDir, { recursive: true });
fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
fs.writeFileSync(
  output.markdownOutputPath,
  [
    '# NN Live Blind SwiGLU Deep Atlas',
    '',
    `Status: **${summary.pass ? 'PASS' : 'NEEDS BACKFILL'}**`,
    '',
    'This atlas reads all current real-model oracle-only Mistral/LLaMA-family SwiGLU decompiler artifacts and builds a layer-by-layer recovery map.',
    '',
    '## Family Summary',
    '',
    ...familySummaries.map(family => [
      `- ${family.modelFamily}: ${family.uniqueLayerCount}/${family.layerCount} layers, rows ${family.artifactRows}, selected ${family.selectedRows}, multi-unit selected ${family.multiUnitSelectedRows}, finite-precision rows ${family.finitePrecisionRows}, min R2 ${fmt(family.minR2)}, min subspace ${fmt(family.minSubspacePrincipalCosine)}, max worst ${fmt(family.maxWorstError)}.`,
      `  Layers: ${family.uniqueLayers.join(', ')}`,
      `  Buckets: early ${family.bucketCoverage.early.join(', ') || '-'}; middle ${family.bucketCoverage.middle.join(', ') || '-'}; late ${family.bucketCoverage.late.join(', ') || '-'}.`
    ].join('\n')),
    '',
    '## Layer Atlas',
    '',
    markdownLayerTable(layers),
    '',
    '## Boundaries',
    '',
    ...report.boundaries.map(item => `- ${item}`),
    '',
    failures.length ? `## Backfill Needed\n\n${failures.map(item => `- ${item}`).join('\n')}\n` : ''
  ].join('\n')
);
writeBenchmarkManifest(repoRoot, output, {
  lane,
  status: summary.pass ? 'pass' : 'generated',
  summary,
  notes: report.boundaries
});

console.log(JSON.stringify(summary, null, 2));
if (!summary.pass) process.exit(1);
