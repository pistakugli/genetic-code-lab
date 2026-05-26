import * as fs from 'fs';
import * as path from 'path';
import { BlockEvaluator } from '../src/core/block-evaluator';
import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';
import { SemanticArchive } from '../src/core/semantic-archive';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Expr = {
  label: string;
  opcodes: number[];
  constants: number[];
};

type ArchiveCase = {
  id: string;
  name: string;
  inputSize: number;
  trainInputs: Float64Array;
  trainExpected: Float64Array;
  testInputs: Float64Array;
  testExpected: Float64Array;
  elites: Expr[];
  waves: number[][];
  capacity: number;
  successMae: number;
};

type MetricSet = {
  mae: number;
  rmse: number;
  r2: number;
  maxAbsError: number;
};

type CaseResult = {
  caseId: string;
  caseName: string;
  bestSingleElite: string;
  bestSingleMetrics: MetricSet;
  archiveDisabledInjected: boolean;
  archiveInjected: boolean;
  archiveMetrics: MetricSet;
  liftRatio: number;
  success: boolean;
  retainedKeys: string[];
};

const repoRoot = path.resolve(__dirname, '..');
const benchmarkOutput = resolveBenchmarkOutput(repoRoot, 'semantic-archive-composition', 'positive-use-case');
const benchmarkDir = benchmarkOutput.outputDir;
const jsonOutputPath = benchmarkOutput.jsonOutputPath;
const markdownOutputPath = benchmarkOutput.markdownOutputPath;
const maxGenomeSize = parsePositiveInt(process.env.GCL_ARCHIVE_BENCH_MAX_GENOME_SIZE, 128);

function main(): void {
  fs.mkdirSync(benchmarkDir, { recursive: true });
  const cases = buildCases();
  const started = process.hrtime.bigint();
  const results = cases.map(runCase);
  const runtimeMs = Number(process.hrtime.bigint() - started) / 1_000_000;
  const report = {
    generatedAt: new Date().toISOString(),
    benchmark: 'semantic-archive-composition',
    budget: {
      cases: cases.length,
      maxGenomeSize,
      note: 'Direct mechanism benchmark: fixed elites are recorded into the semantic archive, then the archive must synthesize a train-only composite candidate.'
    },
    results,
    runtimeMs
  };

  const failures = results.filter(result => !result.success);
  fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(markdownOutputPath, renderMarkdown(report));
  writeBenchmarkManifest(repoRoot, benchmarkOutput, {
    lane: 'semantic-archive-composition',
    status: failures.length > 0 ? 'fail' : 'pass',
    summary: { caseCount: results.length, failures: failures.map(result => result.caseId) }
  });

  console.log('Semantic archive composition benchmark');
  for (const result of results) {
    console.log([
      result.caseId,
      `singleMAE=${formatNumber(result.bestSingleMetrics.mae)}`,
      `archiveMAE=${formatNumber(result.archiveMetrics.mae)}`,
      `lift=${formatNumber(result.liftRatio)}x`,
      `injected=${result.archiveInjected}`,
      `success=${result.success}`
    ].join(' | '));
  }
  console.log(`\nWrote ${path.relative(repoRoot, jsonOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, markdownOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.manifestOutputPath)}`);

  if (failures.length > 0) {
    process.exitCode = 1;
  }
}

function runCase(spec: ArchiveCase): CaseResult {
  const buffer = new PopulationBuffer(spec.elites.length + 2, maxGenomeSize, 10, spec.trainExpected.length);
  for (let i = 0; i < spec.elites.length; i++) {
    const elite = spec.elites[i];
    buffer.writeGenome(i, elite.opcodes, [], new Float64Array(elite.constants));
  }
  BlockEvaluator.evaluateBlock(buffer, 0, spec.elites.length, spec.trainInputs, spec.trainExpected, spec.inputSize);

  const incumbentIdx = selectBestElite(buffer, spec);
  const bestSingleMetrics = evaluateMetrics(buffer, incumbentIdx, spec.testInputs, spec.testExpected, spec.inputSize);

  const disabledInjected = withArchiveEnv('0', () => {
    const disabledArchive = new SemanticArchive(spec.capacity);
    for (const wave of spec.waves) {
      disabledArchive.recordElites(buffer, wave, spec.trainInputs, spec.trainExpected, spec.inputSize);
    }
    return disabledArchive.injectCandidate(
      buffer,
      incumbentIdx,
      spec.elites.length,
      spec.trainInputs,
      spec.trainExpected,
      spec.inputSize,
      maxGenomeSize
    );
  });

  const archive = new SemanticArchive(spec.capacity);
  const targetIdx = spec.elites.length + 1;
  const archiveInjected = withArchiveEnv(undefined, () => {
    for (const wave of spec.waves) {
      archive.recordElites(buffer, wave, spec.trainInputs, spec.trainExpected, spec.inputSize);
    }
    return archive.injectCandidate(
      buffer,
      incumbentIdx,
      targetIdx,
      spec.trainInputs,
      spec.trainExpected,
      spec.inputSize,
      maxGenomeSize
    );
  });

  const archiveMetrics = archiveInjected
    ? evaluateMetrics(buffer, targetIdx, spec.testInputs, spec.testExpected, spec.inputSize)
    : bestSingleMetrics;
  const liftRatio = bestSingleMetrics.mae / Math.max(archiveMetrics.mae, 1e-12);

  return {
    caseId: spec.id,
    caseName: spec.name,
    bestSingleElite: spec.elites[incumbentIdx].label,
    bestSingleMetrics,
    archiveDisabledInjected: disabledInjected,
    archiveInjected,
    archiveMetrics,
    liftRatio,
    success: !disabledInjected && archiveInjected && archiveMetrics.mae <= spec.successMae && liftRatio >= 10,
    retainedKeys: archive.diagnostics().map(entry => entry.key)
  };
}

function selectBestElite(buffer: PopulationBuffer, spec: ArchiveCase): number {
  let bestIdx = 0;
  let bestMae = Number.POSITIVE_INFINITY;
  for (let i = 0; i < spec.elites.length; i++) {
    const metrics = evaluateMetrics(buffer, i, spec.trainInputs, spec.trainExpected, spec.inputSize);
    if (metrics.mae < bestMae) {
      bestMae = metrics.mae;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function evaluateMetrics(
  buffer: PopulationBuffer,
  idx: number,
  inputs: Float64Array,
  expected: Float64Array,
  inputSize: number
): MetricSet {
  return computeMetrics(expected, BlockEvaluator.getPredictions(buffer, idx, inputs, inputSize));
}

function buildCases(): ArchiveCase[] {
  const bivariateTrain = gridInputs([-2, 2], 11, [-1.75, 1.75], 10);
  const bivariateTest = gridInputs([-2.25, 2.25], 13, [-2, 2], 12);
  const duplicateTrain = gridInputs([-3, 3], 13, [-2, 2], 11);
  const duplicateTest = gridInputs([-3.25, 3.25], 15, [-2.25, 2.25], 13);
  const rationalTrain = gridInputs([-2.5, 2.5], 13, [-2.5, 2.5], 13);
  const rationalTest = gridInputs([-2.75, 2.75], 15, [-2.75, 2.75], 15);

  return [
    {
      id: 'delayed_bivariate_fragments',
      name: 'Delayed bivariate fragments',
      inputSize: 2,
      trainInputs: bivariateTrain,
      trainExpected: mapExpected(bivariateTrain, 2, ([x0, x1]) => 0.7 * Math.sin(x0) + 1.3 * x1 * x1 - 0.4 * x0 * x1),
      testInputs: bivariateTest,
      testExpected: mapExpected(bivariateTest, 2, ([x0, x1]) => 0.7 * Math.sin(x0) + 1.3 * x1 * x1 - 0.4 * x0 * x1),
      elites: [
        sin(x(0)),
        mul(x(1), x(1), 'mul(x1,x1)'),
        mul(x(0), x(1), 'mul(x0,x1)'),
        cos(x(0)),
        scale(2, sin(x(0)), 'mul(2,sin(x0))')
      ],
      waves: [[0], [1], [2, 3, 4]],
      capacity: 8,
      successMae: 1e-6
    },
    {
      id: 'duplicate_pressure_market',
      name: 'Duplicate pressure market',
      inputSize: 2,
      trainInputs: duplicateTrain,
      trainExpected: mapExpected(duplicateTrain, 2, ([x0, x1]) => x0 + 0.35 * Math.sin(x1) - 0.25 * x0 * x1),
      testInputs: duplicateTest,
      testExpected: mapExpected(duplicateTest, 2, ([x0, x1]) => x0 + 0.35 * Math.sin(x1) - 0.25 * x0 * x1),
      elites: [
        x(0),
        scale(2, x(0), 'mul(2,x0)'),
        sin(x(1)),
        mul(x(0), x(1), 'mul(x0,x1)'),
        cos(x(1))
      ],
      waves: [[0, 1, 2, 3, 4]],
      capacity: 4,
      successMae: 1e-6
    },
    {
      id: 'rational_trig_composition',
      name: 'Rational + trig composition',
      inputSize: 2,
      trainInputs: rationalTrain,
      trainExpected: mapExpected(rationalTrain, 2, ([x0, x1]) => Math.sin(x0) / (1 + x1 * x1) + 0.5 * Math.cos(x1)),
      testInputs: rationalTest,
      testExpected: mapExpected(rationalTest, 2, ([x0, x1]) => Math.sin(x0) / (1 + x1 * x1) + 0.5 * Math.cos(x1)),
      elites: [
        div(sin(x(0)), add(c(1), mul(x(1), x(1), 'mul(x1,x1)'), 'add(1,mul(x1,x1))'), 'div(sin(x0),add(1,mul(x1,x1)))'),
        cos(x(1)),
        scale(2, cos(x(1)), 'mul(2,cos(x1))'),
        sin(x(0)),
        mul(x(1), x(1), 'mul(x1,x1)')
      ],
      waves: [[0], [1, 2, 3, 4]],
      capacity: 4,
      successMae: 1e-4
    }
  ];
}

function gridInputs(xRange: [number, number], xCount: number, yRange: [number, number], yCount: number): Float64Array {
  const inputs = new Float64Array(xCount * yCount * 2);
  let offset = 0;
  for (let i = 0; i < xCount; i++) {
    const x0 = xCount === 1 ? xRange[0] : xRange[0] + ((xRange[1] - xRange[0]) * i) / (xCount - 1);
    for (let j = 0; j < yCount; j++) {
      const x1 = yCount === 1 ? yRange[0] : yRange[0] + ((yRange[1] - yRange[0]) * j) / (yCount - 1);
      inputs[offset++] = x0;
      inputs[offset++] = x1;
    }
  }
  return inputs;
}

function mapExpected(inputs: Float64Array, inputSize: number, fn: (row: number[]) => number): Float64Array {
  const rows = inputs.length / inputSize;
  const expected = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    const values: number[] = [];
    for (let col = 0; col < inputSize; col++) values.push(inputs[row * inputSize + col]);
    expected[row] = fn(values);
  }
  return expected;
}

function x(index: number): Expr {
  return {
    label: `x${index}`,
    opcodes: [PopulationBuffer.pack(OpCode.XN, index)],
    constants: []
  };
}

function c(value: number): Expr {
  return {
    label: formatNumber(value),
    opcodes: [PopulationBuffer.pack(OpCode.CONST, 0)],
    constants: [value]
  };
}

function sin(term: Expr, label = `sin(${term.label})`): Expr {
  return unary(OpCode.SIN, term, label);
}

function cos(term: Expr, label = `cos(${term.label})`): Expr {
  return unary(OpCode.COS, term, label);
}

function add(left: Expr, right: Expr, label = `add(${left.label},${right.label})`): Expr {
  return binary(OpCode.ADD, left, right, label);
}

function mul(left: Expr, right: Expr, label = `mul(${left.label},${right.label})`): Expr {
  return binary(OpCode.MUL, left, right, label);
}

function div(left: Expr, right: Expr, label = `div(${left.label},${right.label})`): Expr {
  return binary(OpCode.DIV, left, right, label);
}

function scale(value: number, term: Expr, label = `mul(${formatNumber(value)},${term.label})`): Expr {
  return mul(c(value), term, label);
}

function unary(op: OpCode, term: Expr, label: string): Expr {
  return {
    label,
    opcodes: [...term.opcodes, PopulationBuffer.pack(op, 0)],
    constants: [...term.constants]
  };
}

function binary(op: OpCode, left: Expr, right: Expr, label: string): Expr {
  return {
    label,
    opcodes: [
      ...left.opcodes,
      ...shiftConstants(right.opcodes, left.constants.length),
      PopulationBuffer.pack(op, 0)
    ],
    constants: [...left.constants, ...right.constants]
  };
}

function shiftConstants(opcodes: number[], shift: number): number[] {
  return opcodes.map(instruction => {
    const op = (instruction >>> 24) as OpCode;
    const data = instruction & 0xFFFFFF;
    return op === OpCode.CONST ? PopulationBuffer.pack(OpCode.CONST, data + shift) : instruction;
  });
}

function computeMetrics(actual: Float64Array, predicted: Float64Array): MetricSet {
  let absoluteError = 0;
  let squaredError = 0;
  let maxAbsError = 0;
  let mean = 0;
  for (let i = 0; i < actual.length; i++) mean += actual[i];
  mean /= Math.max(1, actual.length);

  let totalSumSquares = 0;
  for (let i = 0; i < actual.length; i++) {
    const prediction = Number.isFinite(predicted[i]) ? predicted[i] : 0;
    const error = prediction - actual[i];
    const absError = Math.abs(error);
    absoluteError += absError;
    squaredError += error * error;
    maxAbsError = Math.max(maxAbsError, absError);
    const centered = actual[i] - mean;
    totalSumSquares += centered * centered;
  }

  return {
    mae: absoluteError / actual.length,
    rmse: Math.sqrt(squaredError / actual.length),
    r2: totalSumSquares === 0 ? 0 : 1 - squaredError / totalSumSquares,
    maxAbsError
  };
}

function withArchiveEnv<T>(value: string | undefined, action: () => T): T {
  const previousArchive = process.env.GCL_SEMANTIC_ARCHIVE;
  const previousRefiner = process.env.GCL_SEMANTIC_REFINER;
  try {
    if (value === undefined) delete process.env.GCL_SEMANTIC_ARCHIVE;
    else process.env.GCL_SEMANTIC_ARCHIVE = value;
    delete process.env.GCL_SEMANTIC_REFINER;
    return action();
  } finally {
    if (previousArchive === undefined) delete process.env.GCL_SEMANTIC_ARCHIVE;
    else process.env.GCL_SEMANTIC_ARCHIVE = previousArchive;
    if (previousRefiner === undefined) delete process.env.GCL_SEMANTIC_REFINER;
    else process.env.GCL_SEMANTIC_REFINER = previousRefiner;
  }
}

function renderMarkdown(report: { generatedAt: string; budget: Record<string, unknown>; results: CaseResult[]; runtimeMs: number }): string {
  const lines: string[] = [];
  lines.push('# Semantic archive composition benchmark');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push('- Measures the semantic archive as a mechanism, not the whole evolutionary loop.');
  lines.push('- Each case gives the archive multiple elite fragments, including duplicates and distractors.');
  lines.push('- The pass condition requires `GCL_SEMANTIC_ARCHIVE=0` to inject nothing, while the enabled archive synthesizes a lower-error composite from train data only.');
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Case | Single-best MAE | Archive MAE | Lift | Injected | Success | Retained keys |');
  lines.push('|---|---:|---:|---:|---:|---:|---|');
  for (const result of report.results) {
    lines.push(`| ${result.caseId} | ${formatNumber(result.bestSingleMetrics.mae)} | ${formatNumber(result.archiveMetrics.mae)} | ${formatNumber(result.liftRatio)}x | ${result.archiveInjected ? 'yes' : 'no'} | ${result.success ? 'yes' : 'no'} | ${truncate(result.retainedKeys.join(', '), 180)} |`);
  }
  lines.push('');
  lines.push('## Details');
  lines.push('');
  for (const result of report.results) {
    lines.push(`- ${result.caseId}: best single elite \`${result.bestSingleElite}\`, archive disabled injected=${result.archiveDisabledInjected}, max error ${formatNumber(result.archiveMetrics.maxAbsError)}.`);
  }
  lines.push('');
  lines.push(`Runtime: ${formatNumber(report.runtimeMs)} ms`);
  lines.push('');
  return `${lines.join('\n')}\n`;
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) return fallback;
  return parsed;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs < 0.0001 || abs >= 100000) return value.toExponential(3);
  return value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}

function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 3)}...`;
}

main();
