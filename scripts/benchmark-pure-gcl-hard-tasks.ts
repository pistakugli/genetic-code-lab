import * as fs from 'fs';
import * as path from 'path';
import { EvolutionEngine } from '../src/core/evolution';
import { Genome, ProblemDefinition, TerminalDefinition } from '../src/core/types';
import { MATH_FUNCTIONS, executeMathGenome, genomeToString, mathFitnessFunction } from '../src/solvers/math-expression';
import { setSeed } from '../src/utils/helpers';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Point = { x: number; y: number };
type MetricSet = { mae: number; rmse: number; r2: number; maxAbsError: number };
type HardTask = {
  id: string;
  name: string;
  formula: string;
  fn: (x: number) => number;
  trainRange: [number, number];
  validationRange: [number, number];
  guardRange: [number, number];
  testRange: [number, number];
  functionSet: string[];
  successMae: number;
};
type CandidateResult = {
  label: string;
  role: string;
  validation: MetricSet;
  guard: MetricSet;
  test: MetricSet;
  expression: string;
  runtimeMs: number;
  generationsRun: number;
  fitness: number;
};
type CompletedSeedResult = {
  seed: number;
  taskId: string;
  completed: true;
  single: CandidateResult;
  islands: CandidateResult[];
  selected: CandidateResult;
  oracle: CandidateResult;
  selectedLiftVsSingle: number;
  oracleLiftVsSingle: number;
  success: boolean;
  selectedBy: 'validation' | 'single-guard-fallback';
  outcome: 'completed';
};
type FailedSeedResult = {
  seed: number;
  taskId: string;
  completed: false;
  success: false;
  outcome: 'runtime-timeout' | 'runtime-error';
  error: string;
};
type SeedResult = CompletedSeedResult | FailedSeedResult;

const repoRoot = path.resolve(__dirname, '..');
const seeds = parseSeeds(process.env.GCL_HARD_TASK_SEEDS ?? '1337,2026,4242');
const generations = parsePositiveInt(process.env.GCL_HARD_TASK_GENERATIONS, 60);
const singlePopulation = parsePositiveInt(process.env.GCL_HARD_TASK_SINGLE_POPULATION, 220);
const islandPopulation = parsePositiveInt(process.env.GCL_HARD_TASK_ISLAND_POPULATION, 120);
const seedTimeoutMs = parsePositiveInt(process.env.GCL_HARD_TASK_SEED_TIMEOUT_MS, 120000);
const guardFallbackRatio = Number(process.env.GCL_HARD_TASK_GUARD_FALLBACK_RATIO ?? '1.15');
const benchmarkOutput = resolveBenchmarkOutput(repoRoot, 'pure-gcl-hard-tasks', `multi-island-${seeds.length}seed`);

const tasks: HardTask[] = [
  {
    id: 'inverse_quadratic',
    name: 'Inverse quadratic',
    formula: '1 / (1 + x^2)',
    fn: x => 1 / (1 + x * x),
    trainRange: [-3, 3],
    validationRange: [-3.1, 3.1],
    guardRange: [-3.2, 3.2],
    testRange: [-3.25, 3.25],
    functionSet: ['add', 'sub', 'mul', 'div'],
    successMae: 1e-3
  },
  {
    id: 'trig_rational_combo',
    name: 'Trigonometric + rational',
    formula: 'sin(x) + x / (1 + x^2)',
    fn: x => Math.sin(x) + x / (1 + x * x),
    trainRange: [-3, 3],
    validationRange: [-3.1, 3.1],
    guardRange: [-3.2, 3.2],
    testRange: [-3.25, 3.25],
    functionSet: ['add', 'sub', 'mul', 'div', 'sin', 'cos'],
    successMae: 1e-3
  }
];

const roles = [
  { name: 'full', functions: ['add', 'sub', 'mul', 'div', 'sin', 'cos'] },
  { name: 'arithmetic', functions: ['add', 'sub', 'mul', 'div'] },
  { name: 'trig', functions: ['add', 'sub', 'mul', 'sin', 'cos'] },
  { name: 'rational-trig', functions: ['add', 'sub', 'mul', 'div', 'sin', 'cos'] }
];

async function main(): Promise<void> {
  fs.mkdirSync(benchmarkOutput.outputDir, { recursive: true });
  const previousBasisSeed = process.env.GCL_REGRESSION_BASIS_SEED;
  const previousRefiner = process.env.GCL_SEMANTIC_REFINER;

  try {
    process.env.GCL_REGRESSION_BASIS_SEED = '0';
    process.env.GCL_SEMANTIC_REFINER = '0';

    const results: SeedResult[] = [];
    for (const task of tasks) {
      const data = {
        train: makeData(task.fn, task.trainRange, 71),
        validation: makeData(task.fn, task.validationRange, 81),
        guard: makeData(task.fn, task.guardRange, 91),
        test: makeData(task.fn, task.testRange, 121)
      };
      for (const seed of seeds) {
        try {
          results.push(await withTimeout(runSeed(task, data, seed), seedTimeoutMs));
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          results.push(makeFailedSeedResult(task, seed, message.includes('timed out') ? 'runtime-timeout' : 'runtime-error', message));
        }
      }
    }

    const summary = summarize(results);
    const report = {
      generatedAt: new Date().toISOString(),
      benchmark: 'pure-gcl-hard-tasks',
      scope: 'Diagnostic pure-GCL hard-task benchmark. It disables regression-basis seeding and semantic refiner. It does not use gcl-hybrid.',
      budget: {
        seeds,
        generations,
        singlePopulation,
        islandPopulation,
        seedTimeoutMs,
        guardFallbackRatio,
        disabled: {
          GCL_REGRESSION_BASIS_SEED: '0',
          GCL_SEMANTIC_REFINER: '0'
        }
      },
      tasks: tasks.map(task => ({
        id: task.id,
        name: task.name,
        formula: task.formula,
        successMae: task.successMae,
        functionSet: task.functionSet
      })),
      summary,
      results
    };
    const pass = summary.allTasksSolved && summary.nonRegressionRate >= 1 && summary.runtimeErrorCount === 0;
    fs.writeFileSync(benchmarkOutput.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
    fs.writeFileSync(benchmarkOutput.markdownOutputPath, renderMarkdown(report));
    writeBenchmarkManifest(repoRoot, benchmarkOutput, {
      lane: 'pure-gcl-hard-tasks',
      status: pass ? 'pass' : 'fail',
      summary,
      notes: ['Diagnostic lane. Failing this lane blocks pure-GCL hard-task claims, not gcl-hybrid production status.']
    });

    for (const result of results) {
      if (isCompletedResult(result)) {
        console.log([
          `task=${result.taskId}`,
          `seed=${result.seed}`,
          `single=${formatNumber(result.single.test.mae)}`,
          `selected=${formatNumber(result.selected.test.mae)}`,
          `oracle=${formatNumber(result.oracle.test.mae)}`,
          `lift=${formatNumber(result.selectedLiftVsSingle)}x`,
          `success=${result.success}`,
          `selectedBy=${result.selectedBy}`
        ].join(' | '));
      } else {
        console.log([
          `task=${result.taskId}`,
          `seed=${result.seed}`,
          `outcome=${result.outcome}`,
          `success=false`,
          `error=${result.error}`
        ].join(' | '));
      }
    }
    console.log(JSON.stringify(summary, null, 2));
    console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.jsonOutputPath)}`);
    console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.markdownOutputPath)}`);
    console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.manifestOutputPath)}`);

    if (!pass) process.exitCode = 1;
  } finally {
    if (previousBasisSeed === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
    else process.env.GCL_REGRESSION_BASIS_SEED = previousBasisSeed;
    if (previousRefiner === undefined) delete process.env.GCL_SEMANTIC_REFINER;
    else process.env.GCL_SEMANTIC_REFINER = previousRefiner;
  }
}

async function runSeed(
  task: HardTask,
  data: { train: Point[]; validation: Point[]; guard: Point[]; test: Point[] },
  seed: number
): Promise<SeedResult> {
  return withDeterministicMathRandom(seed, async () => {
    const single = await runCandidate(task, data, seed, 'single', 'single', singlePopulation);
    const islands: CandidateResult[] = [];
    for (let i = 0; i < roles.length; i++) {
      const role = roles[i];
      const filteredFunctions = role.functions.filter(name => task.functionSet.includes(name));
      islands.push(await runCandidate(task, data, seed + i * 7919, `island-${role.name}`, role.name, islandPopulation, filteredFunctions));
    }

    const validationBest = [...islands].sort((a, b) => a.validation.mae - b.validation.mae)[0];
    const selected = validationBest.guard.mae <= single.guard.mae * guardFallbackRatio ? validationBest : single;
    const selectedBy = selected === single ? 'single-guard-fallback' : 'validation';
    const oracle = [single, ...islands].sort((a, b) => a.test.mae - b.test.mae)[0];
    const selectedLiftVsSingle = ratio(single.test.mae, selected.test.mae);
    const oracleLiftVsSingle = ratio(single.test.mae, oracle.test.mae);
    return {
      seed,
      taskId: task.id,
      single,
      islands,
      selected,
      oracle,
      selectedLiftVsSingle,
      oracleLiftVsSingle,
      success: selected.test.mae <= task.successMae,
      selectedBy,
      completed: true,
      outcome: 'completed'
    };
  });
}

async function runCandidate(
  task: HardTask,
  data: { train: Point[]; validation: Point[]; guard: Point[]; test: Point[] },
  seed: number,
  label: string,
  role: string,
  populationSize: number,
  functionNames: string[] = task.functionSet
): Promise<CandidateResult> {
  setSeed(seed);
  const started = process.hrtime.bigint();
  const engine = new EvolutionEngine(createProblem(task, data.train, functionNames, `${task.id}-${label}`), {
    population: {
      size: populationSize,
      eliteCount: Math.max(3, Math.floor(populationSize * 0.05)),
      mutationRate: role === 'arithmetic' ? 0.28 : 0.34,
      crossoverRate: 0.78,
      maxGenomeDepth: task.functionSet.includes('sin') ? 7 : 6,
      maxGenomeSize: 96,
      tournamentSize: 5,
      parsimonyCoefficient: 0.00012
    },
    targetFitness: 0.999999,
    maxGenerations: generations,
    stagnationLimit: generations,
    parallelEvaluators: 1,
    selectionMethod: 'tournament',
    seed,
    optimizer: {
      enabled: true,
      iterations: 16,
      learningRate: 0.04,
      interval: 5,
      targetCount: 5
    }
  });
  const result = await engine.evolve();
  const runtimeMs = Number(process.hrtime.bigint() - started) / 1_000_000;
  return {
    label,
    role,
    validation: evaluateGenome(result.bestGenome, data.validation),
    guard: evaluateGenome(result.bestGenome, data.guard),
    test: evaluateGenome(result.bestGenome, data.test),
    expression: genomeToString(result.bestGenome),
    runtimeMs,
    generationsRun: result.generationsRun,
    fitness: result.bestGenome.fitness
  };
}

function createProblem(task: HardTask, train: Point[], functionNames: string[], name: string): ProblemDefinition<number[], number> {
  const selectedFunctions = MATH_FUNCTIONS.filter(fn => functionNames.includes(fn.name));
  const selected = new Set(selectedFunctions.map(fn => fn.name));
  const missing = functionNames.filter(fn => !selected.has(fn));
  if (missing.length > 0) throw new Error(`${task.id}: unsupported functions ${missing.join(', ')}`);
  return {
    name,
    description: `Pure GCL hard task diagnostic: ${task.formula}`,
    functionSet: selectedFunctions,
    terminalSet: fixedTerminals(),
    testCases: train.map(point => ({ input: [point.x], expectedOutput: point.y, weight: 1 })),
    fitnessFunction: mathFitnessFunction
  };
}

function fixedTerminals(): TerminalDefinition[] {
  return [
    { name: 'x', type: 'variable', description: 'Variable x' },
    ...[-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, Math.PI, Math.E].map(value => ({
      name: 'const',
      type: 'constant' as const,
      value,
      description: `Fixed benchmark constant ${value}`
    }))
  ];
}

function makeData(fn: (x: number) => number, range: [number, number], count: number): Point[] {
  const [lo, hi] = range;
  return Array.from({ length: count }, (_, index) => {
    const x = count === 1 ? lo : lo + ((hi - lo) * index) / (count - 1);
    return { x, y: fn(x) };
  });
}

function evaluateGenome(genome: Genome, points: Point[]): MetricSet {
  return computeMetrics(
    points.map(point => point.y),
    points.map(point => safeNumber(executeMathGenome(genome, { x: point.x })))
  );
}

function makeFailedSeedResult(task: HardTask, seed: number, outcome: FailedSeedResult['outcome'], error: string): FailedSeedResult {
  return {
    seed,
    taskId: task.id,
    completed: false,
    success: false,
    outcome,
    error
  };
}

function isCompletedResult(result: SeedResult): result is CompletedSeedResult {
  return result.completed;
}

function computeMetrics(actual: number[], predicted: number[]): MetricSet {
  let absoluteError = 0;
  let squaredError = 0;
  let maxAbsError = 0;
  const actualMean = mean(actual);
  let totalSumSquares = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = predicted[i] - actual[i];
    absoluteError += Math.abs(error);
    squaredError += error * error;
    maxAbsError = Math.max(maxAbsError, Math.abs(error));
    const centered = actual[i] - actualMean;
    totalSumSquares += centered * centered;
  }
  return {
    mae: absoluteError / actual.length,
    rmse: Math.sqrt(squaredError / actual.length),
    r2: totalSumSquares === 0 ? 0 : 1 - squaredError / totalSumSquares,
    maxAbsError
  };
}

function summarize(results: SeedResult[]) {
  const byTask = Object.fromEntries(tasks.map(task => {
    const taskResults = results.filter(result => result.taskId === task.id);
    const completedTaskResults = taskResults.filter(isCompletedResult);
    return [task.id, {
      successRate: taskResults.filter(result => result.success).length / Math.max(1, taskResults.length),
      completedCount: completedTaskResults.length,
      timedOutCount: taskResults.filter(result => result.outcome === 'runtime-timeout').length,
      runtimeErrorCount: taskResults.filter(result => result.outcome === 'runtime-error').length,
      meanSingleMae: mean(completedTaskResults.map(result => result.single.test.mae)),
      meanSelectedMae: mean(completedTaskResults.map(result => result.selected.test.mae)),
      meanOracleMae: mean(completedTaskResults.map(result => result.oracle.test.mae)),
      medianSelectedLift: median(completedTaskResults.map(result => result.selectedLiftVsSingle)),
      medianOracleLift: median(completedTaskResults.map(result => result.oracleLiftVsSingle))
    }];
  }));
  const completedResults = results.filter(isCompletedResult);
  const nonRegressionRate = completedResults.filter(result => result.selected.test.mae <= result.single.test.mae + 1e-12).length / Math.max(1, completedResults.length);
  return {
    allTasksSolved: results.every(result => result.success),
    successRate: results.filter(result => result.success).length / Math.max(1, results.length),
    nonRegressionRate,
    completedCount: completedResults.length,
    timedOutCount: results.filter(result => result.outcome === 'runtime-timeout').length,
    runtimeErrorCount: results.filter(result => result.outcome === 'runtime-error').length,
    meanSingleMae: mean(completedResults.map(result => result.single.test.mae)),
    meanSelectedMae: mean(completedResults.map(result => result.selected.test.mae)),
    meanOracleMae: mean(completedResults.map(result => result.oracle.test.mae)),
    medianSelectedLift: median(completedResults.map(result => result.selectedLiftVsSingle)),
    medianOracleLift: median(completedResults.map(result => result.oracleLiftVsSingle)),
    byTask
  };
}

function renderMarkdown(report: {
  generatedAt: string;
  scope: string;
  budget: Record<string, unknown>;
  summary: ReturnType<typeof summarize>;
  results: SeedResult[];
}): string {
  const lines: string[] = [];
  lines.push('# Pure GCL hard-task diagnostic');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push(`- ${report.scope}`);
  lines.push('- Compares a single pure GCL run with role-diverse pure GCL island runs.');
  lines.push('- Selection uses validation MAE with a guard fallback to the single baseline.');
  lines.push('- Oracle rows are diagnostic only and are selected using final test MAE.');
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Success rate: ${formatPercent(report.summary.successRate)}`);
  lines.push(`- Non-regression rate vs single: ${formatPercent(report.summary.nonRegressionRate)}`);
  lines.push(`- Completed task/seed runs: ${report.summary.completedCount}/${report.results.length}`);
  lines.push(`- Runtime timeouts: ${report.summary.timedOutCount}`);
  lines.push(`- Runtime errors: ${report.summary.runtimeErrorCount}`);
  lines.push(`- Mean single MAE: ${formatNumber(report.summary.meanSingleMae)}`);
  lines.push(`- Mean selected MAE: ${formatNumber(report.summary.meanSelectedMae)}`);
  lines.push(`- Mean oracle MAE: ${formatNumber(report.summary.meanOracleMae)}`);
  lines.push(`- Median selected lift: ${formatNumber(report.summary.medianSelectedLift)}x`);
  lines.push(`- Median oracle lift: ${formatNumber(report.summary.medianOracleLift)}x`);
  lines.push('');
  lines.push('## Results');
  lines.push('');
  lines.push('| Task | Seed | Single MAE | Selected MAE | Oracle MAE | Selected lift | Oracle lift | Selected by | Success |');
  lines.push('|---|---:|---:|---:|---:|---:|---:|---|---|');
  for (const result of report.results) {
    if (isCompletedResult(result)) {
      lines.push(`| ${result.taskId} | ${result.seed} | ${formatNumber(result.single.test.mae)} | ${formatNumber(result.selected.test.mae)} | ${formatNumber(result.oracle.test.mae)} | ${formatNumber(result.selectedLiftVsSingle)}x | ${formatNumber(result.oracleLiftVsSingle)}x | ${result.selectedBy} | ${result.success ? 'yes' : 'no'} |`);
    } else {
      lines.push(`| ${result.taskId} | ${result.seed} | - | - | - | - | - | ${result.outcome}: ${escapeMarkdownCell(result.error)} | no |`);
    }
  }
  lines.push('');
  lines.push('## Claim Boundary');
  lines.push('');
  lines.push('- Passing this lane would support a pure-GCL hard-task improvement claim for these tasks only.');
  lines.push('- Failing this lane preserves the current boundary: hard inverse/rational tasks remain pure-GCL blockers while gcl-hybrid remains separate.');
  return `${lines.join('\n')}\n`;
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

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  const started = Date.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const result = await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`seed timed out after ${timeoutMs}ms`)), timeoutMs);
      })
    ]);
    const elapsedMs = Date.now() - started;
    if (elapsedMs > timeoutMs) throw new Error(`seed timed out after ${timeoutMs}ms; completed after ${elapsedMs}ms`);
    return result;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function parseSeeds(value: string): number[] {
  const parsed = value.split(',').map(item => Number(item.trim())).filter(Number.isInteger);
  if (parsed.length === 0) throw new Error('GCL_HARD_TASK_SEEDS must contain at least one integer seed');
  return parsed;
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function safeNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function ratio(before: number, after: number): number {
  const floor = 1e-12;
  if (before <= floor && after <= floor) return 1;
  return before / Math.max(after, floor);
}

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) return value.toExponential(3);
  return value.toFixed(6).replace(/\.?0+$/, '');
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function escapeMarkdownCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
