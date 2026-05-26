import * as fs from 'fs';
import * as path from 'path';
import {
  MATH_FUNCTIONS,
  executeMathGenome,
  mathFitnessFunction
} from '../src/solvers/math-expression';
import { EvolutionEngine } from '../src/core/evolution';
import { ProblemDefinition, TerminalDefinition } from '../src/core/types';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Point = { input: number[]; expected: number };
type MetricSet = { mae: number; rmse: number; r2: number; maxAbsError: number };
type ModeName = 'archive-off' | 'archive-on';

type ModeRun = {
  mode: ModeName;
  seed: number;
  metrics: MetricSet;
  runtimeMs: number;
  generationsRun: number;
  fitness: number;
  archive: {
    recordCycles: number;
    injectionAttempts: number;
    injectionSuccesses: number;
    size: number;
    retainedKeys: string[];
  };
};

type PairResult = {
  seed: number;
  off: ModeRun;
  on: ModeRun;
  liftRatio: number;
  success: boolean;
};

const repoRoot = path.resolve(__dirname, '..');
const benchmarkOutput = resolveBenchmarkOutput(repoRoot, 'archive-emergence', 'positive-use-case');
const benchmarkDir = benchmarkOutput.outputDir;
const jsonOutputPath = benchmarkOutput.jsonOutputPath;
const markdownOutputPath = benchmarkOutput.markdownOutputPath;

const seeds = parseSeeds(process.env.GCL_ARCHIVE_EMERGENCE_SEEDS ?? '3,9,11');
const populationSize = parsePositiveInt(process.env.GCL_ARCHIVE_EMERGENCE_POPULATION, 160);
const generations = parsePositiveInt(process.env.GCL_ARCHIVE_EMERGENCE_GENERATIONS, 100);
const successLift = Number(process.env.GCL_ARCHIVE_EMERGENCE_MIN_LIFT ?? '1.1');

async function main(): Promise<void> {
  fs.mkdirSync(benchmarkDir, { recursive: true });
  const train = makeGrid([-2, 2], 11, [-1.75, 1.75], 10);
  const test = makeGrid([-2.25, 2.25], 13, [-2, 2], 12);
  const pairs: PairResult[] = [];

  for (const seed of seeds) {
    const off = await runMode('archive-off', seed, train, test);
    const on = await runMode('archive-on', seed, train, test);
    const liftRatio = off.metrics.mae / Math.max(on.metrics.mae, 1e-12);
    pairs.push({
      seed,
      off,
      on,
      liftRatio,
      success: on.archive.recordCycles > 0
        && on.archive.injectionSuccesses > 0
        && on.metrics.mae < off.metrics.mae
        && liftRatio >= successLift
    });
  }

  const report = {
    generatedAt: new Date().toISOString(),
    benchmark: 'archive-emergence',
    task: {
      name: 'Closed-loop bivariate fragment composition',
      formula: '0.7*sin(x0) + 1.3*x1*x1 - 0.4*x0*x1',
      trainCases: train.length,
      testCases: test.length
    },
    budget: {
      seeds,
      populationSize,
      generations,
      successLift,
      note: 'Positive-use-case gate. Runs the real EvolutionEngine twice per seed with regression basis seed disabled. Archive-off leaves the semantic refiner active but disables cross-generation archive injection; archive-on enables archive recording and candidate injection.'
    },
    summary: summarize(pairs),
    pairs
  };

  const pass = report.summary.successRate >= 2 / 3 && report.summary.meanLiftRatio >= successLift;
  fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(markdownOutputPath, renderMarkdown(report));
  writeBenchmarkManifest(repoRoot, benchmarkOutput, {
    lane: 'archive-emergence',
    status: pass ? 'pass' : 'fail',
    summary: report.summary,
    notes: ['Curated positive-use-case gate; not a universal archive superiority claim.']
  });

  console.log('Closed-loop archive emergence benchmark');
  for (const pair of pairs) {
    console.log([
      `seed=${pair.seed}`,
      `offMAE=${formatNumber(pair.off.metrics.mae)}`,
      `onMAE=${formatNumber(pair.on.metrics.mae)}`,
      `lift=${formatNumber(pair.liftRatio)}x`,
      `archiveInject=${pair.on.archive.injectionSuccesses}`,
      `success=${pair.success}`
    ].join(' | '));
  }
  console.log(`summaryLift=${formatNumber(report.summary.meanLiftRatio)}x successRate=${Math.round(report.summary.successRate * 100)}%`);
  console.log(`\nWrote ${path.relative(repoRoot, jsonOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, markdownOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.manifestOutputPath)}`);

  if (!pass) process.exitCode = 1;
}

async function runMode(mode: ModeName, seed: number, train: Point[], test: Point[]): Promise<ModeRun> {
  const previousArchive = process.env.GCL_SEMANTIC_ARCHIVE;
  const previousBasisSeed = process.env.GCL_REGRESSION_BASIS_SEED;
  try {
    process.env.GCL_REGRESSION_BASIS_SEED = '0';
    if (mode === 'archive-off') process.env.GCL_SEMANTIC_ARCHIVE = '0';
    else delete process.env.GCL_SEMANTIC_ARCHIVE;

    return await withDeterministicMathRandom(seed, async () => {
      const started = process.hrtime.bigint();
      const engine = new EvolutionEngine(createProblem(train), {
        population: {
          size: populationSize,
          eliteCount: Math.max(3, Math.floor(populationSize * 0.05)),
          mutationRate: 0.32,
          crossoverRate: 0.78,
          maxGenomeDepth: 3,
          maxGenomeSize: 56,
          tournamentSize: 5,
          parsimonyCoefficient: 0.00015
        },
        targetFitness: 0.999999,
        maxGenerations: generations,
        stagnationLimit: generations,
        parallelEvaluators: 1,
        selectionMethod: 'tournament',
        seed,
        optimizer: {
          enabled: true,
          iterations: 10,
          learningRate: 0.04,
          interval: 5,
          targetCount: 4
        }
      });
      const result = await engine.evolve();
      const runtimeMs = Number(process.hrtime.bigint() - started) / 1_000_000;
      const predictions = test.map(point => safeNumber(executeMathGenome(result.bestGenome, toInputRecord(point.input))));
      const metrics = computeMetrics(test.map(point => point.expected), predictions);
      const archiveStats = engine.getSemanticArchiveDebugStats();

      return {
        mode,
        seed,
        metrics,
        runtimeMs,
        generationsRun: result.generationsRun,
        fitness: result.bestGenome.fitness,
        archive: {
          recordCycles: archiveStats.recordCycles,
          injectionAttempts: archiveStats.injectionAttempts,
          injectionSuccesses: archiveStats.injectionSuccesses,
          size: archiveStats.size,
          retainedKeys: archiveStats.diagnostics.slice(0, 16).map(entry => entry.key)
        }
      };
    });
  } finally {
    if (previousArchive === undefined) delete process.env.GCL_SEMANTIC_ARCHIVE;
    else process.env.GCL_SEMANTIC_ARCHIVE = previousArchive;
    if (previousBasisSeed === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
    else process.env.GCL_REGRESSION_BASIS_SEED = previousBasisSeed;
  }
}

async function withDeterministicMathRandom<T>(seed: number, action: () => Promise<T>): Promise<T> {
  const originalRandom = Math.random;
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  Math.random = () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
  try {
    return await action();
  } finally {
    Math.random = originalRandom;
  }
}

function createProblem(train: Point[]): ProblemDefinition<number[], number> {
  return {
    name: 'Closed-loop archive emergence',
    description: 'Bivariate regression where useful fragments can emerge in different lineages before archive composition.',
    functionSet: MATH_FUNCTIONS.filter(fn => ['add', 'sub', 'mul', 'div', 'sin', 'cos'].includes(fn.name)),
    terminalSet: createTerminals(),
    testCases: train.map(point => ({ input: point.input, expectedOutput: point.expected, weight: 1 })),
    fitnessFunction: mathFitnessFunction
  };
}

function createTerminals(): TerminalDefinition[] {
  return [
    { name: 'x', type: 'variable', description: 'Variable x0' },
    { name: 'x1', type: 'variable', description: 'Variable x1' },
    ...[-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 0.7, 1, 1.3, 2].map(value => ({
      name: 'const',
      type: 'constant' as const,
      value,
      description: `Fixed emergence constant ${value}`
    }))
  ];
}

function makeGrid(xRange: [number, number], xCount: number, yRange: [number, number], yCount: number): Point[] {
  const points: Point[] = [];
  for (let i = 0; i < xCount; i++) {
    const x0 = xCount === 1 ? xRange[0] : xRange[0] + ((xRange[1] - xRange[0]) * i) / (xCount - 1);
    for (let j = 0; j < yCount; j++) {
      const x1 = yCount === 1 ? yRange[0] : yRange[0] + ((yRange[1] - yRange[0]) * j) / (yCount - 1);
      points.push({
        input: [x0, x1],
        expected: 0.7 * Math.sin(x0) + 1.3 * x1 * x1 - 0.4 * x0 * x1
      });
    }
  }
  return points;
}

function toInputRecord(input: number[]): Record<string, number> {
  return { x: input[0], x1: input[1] };
}

function computeMetrics(actual: number[], predicted: number[]): MetricSet {
  let absoluteError = 0;
  let squaredError = 0;
  let maxAbsError = 0;
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  let totalSumSquares = 0;

  for (let i = 0; i < actual.length; i++) {
    const error = predicted[i] - actual[i];
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

function summarize(pairs: PairResult[]): {
  successRate: number;
  meanOffMae: number;
  meanOnMae: number;
  meanLiftRatio: number;
  totalArchiveInjections: number;
} {
  return {
    successRate: pairs.filter(pair => pair.success).length / Math.max(1, pairs.length),
    meanOffMae: mean(pairs.map(pair => pair.off.metrics.mae)),
    meanOnMae: mean(pairs.map(pair => pair.on.metrics.mae)),
    meanLiftRatio: mean(pairs.map(pair => pair.liftRatio)),
    totalArchiveInjections: pairs.reduce((sum, pair) => sum + pair.on.archive.injectionSuccesses, 0)
  };
}

function renderMarkdown(report: {
  generatedAt: string;
  task: { name: string; formula: string; trainCases: number; testCases: number };
  budget: Record<string, unknown>;
  summary: ReturnType<typeof summarize>;
  pairs: PairResult[];
}): string {
  const lines: string[] = [];
  lines.push('# Closed-loop archive emergence benchmark');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push('- Runs the real `EvolutionEngine`, not fixed archive fragments.');
  lines.push('- Compares the same seeded run with `GCL_SEMANTIC_ARCHIVE=0` and with archive enabled.');
  lines.push('- Keeps the semantic refiner enabled in both modes; the measured delta is cross-generation archive recording/injection.');
  lines.push('- Disables `GCL_REGRESSION_BASIS_SEED` so this benchmark measures emergence from the evolutionary loop rather than the grammar-derived seed.');
  lines.push('- This is a deterministic positive-use-case gate, not a claim that archive-on beats archive-off for every random seed.');
  lines.push('');
  lines.push('## Task');
  lines.push('');
  lines.push(`- Formula: \`${report.task.formula}\``);
  lines.push(`- Train cases: ${report.task.trainCases}`);
  lines.push(`- Test cases: ${report.task.testCases}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Mean archive-off MAE: ${formatNumber(report.summary.meanOffMae)}`);
  lines.push(`- Mean archive-on MAE: ${formatNumber(report.summary.meanOnMae)}`);
  lines.push(`- Mean lift: ${formatNumber(report.summary.meanLiftRatio)}x`);
  lines.push(`- Success rate: ${Math.round(report.summary.successRate * 100)}%`);
  lines.push(`- Archive injection successes: ${report.summary.totalArchiveInjections}`);
  lines.push('');
  lines.push('| Seed | Archive-off MAE | Archive-on MAE | Lift | Archive injections | Success | Retained keys |');
  lines.push('|---:|---:|---:|---:|---:|---:|---|');
  for (const pair of report.pairs) {
    lines.push(`| ${pair.seed} | ${formatNumber(pair.off.metrics.mae)} | ${formatNumber(pair.on.metrics.mae)} | ${formatNumber(pair.liftRatio)}x | ${pair.on.archive.injectionSuccesses} | ${pair.success ? 'yes' : 'no'} | ${truncate(pair.on.archive.retainedKeys.join(', '), 180)} |`);
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

function parseSeeds(value: string): number[] {
  const parsed = value.split(',').map(item => Number(item.trim())).filter(Number.isInteger);
  if (parsed.length === 0) throw new Error('GCL_ARCHIVE_EMERGENCE_SEEDS must contain at least one integer');
  return parsed;
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) return fallback;
  return parsed;
}

function mean(values: number[]): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function safeNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
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

main().catch(error => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});
