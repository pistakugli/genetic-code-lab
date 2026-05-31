import { spawnSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import {
  MATH_FUNCTIONS,
  createMathTerminals,
  executeMathGenome,
  genomeToString,
  mathFitnessFunction
} from '../src/solvers/math-expression';
import { EvolutionEngine } from '../src/core/evolution';
import { FunctionDefinition, ProblemDefinition, TerminalDefinition } from '../src/core/types';
import { setSeed } from '../src/utils/helpers';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Point = { x: number; y: number };
type MetricSet = { mae: number; rmse: number; r2: number; maxAbsError: number };
type BaselineName = 'gcl' | 'gcl-hybrid' | 'gplearn' | 'deap';

type TaskSpec = {
  id: string;
  name: string;
  formula: string;
  fn: (x: number) => number;
  trainRange: [number, number];
  testRange: [number, number];
  trainPoints: number;
  testPoints: number;
  functionSet: string[];
  successMae: number;
};

type BenchmarkRun = {
  taskId: string;
  taskName: string;
  baseline: BaselineName;
  seed: number;
  metrics: MetricSet;
  runtimeMs: number;
  expression: string;
  success: boolean;
  generationsRun?: number;
  fitness?: number;
  calibration?: {
    method: 'train-affine-output';
    accepted: boolean;
    scale: number;
    offset: number;
    trainMaeBefore: number;
    trainMaeAfter: number;
    validationMaeBefore: number;
    validationMaeAfter: number;
  };
  hybridSelection?: {
    method: 'train-validation-sparse-basis';
    candidateTerms: number;
    selectedTerms: string[];
    fitMae: number;
    validationMae: number;
    validationMaeBefore: number;
  };
  error?: string;
};

type PythonRunResult = {
  metrics: MetricSet;
  runtimeMs: number;
  expression: string;
  success: boolean;
  generationsRun?: number;
  fitness?: number;
};

const repoRoot = path.resolve(__dirname, '..');
const seeds = parseSeeds(process.env.GCL_BENCH_SEEDS ?? '1337,2026,4242');
const populationSize = parsePositiveInt(process.env.GCL_BENCH_POPULATION, 300);
const generations = parsePositiveInt(process.env.GCL_BENCH_GENERATIONS, 80);
const taskSet = process.env.GCL_BENCH_TASK_SET === 'wide' ? 'wide' : 'smoke';
const benchmarkVariant = process.env.GCL_OPEN_SOURCE_GATE_MODE === 'hybrid-production' ? 'hybrid-wide' : taskSet;
const benchmarkOutput = resolveBenchmarkOutput(repoRoot, 'open-source-symbolic-regression', benchmarkVariant);
const benchmarkDir = benchmarkOutput.outputDir;
const jsonOutputPath = benchmarkOutput.jsonOutputPath;
const markdownOutputPath = benchmarkOutput.markdownOutputPath;
const requiredPythonModules = ['numpy', 'sklearn', 'gplearn', 'deap'];
const pythonRequirementsPath = path.join(repoRoot, 'scripts', 'open-source-python-requirements.txt');
const defaultPythonVenvPath = process.env.GCL_OPEN_SOURCE_PYTHON_VENV ?? '/tmp/gcl-open-source-bench-venv';
const pythonBin = resolvePythonBin();

const smokeTasks: TaskSpec[] = [
  {
    id: 'quadratic_arithmetic',
    name: 'Quadratic arithmetic',
    formula: 'x^2 + 2x + 1',
    fn: (x) => x * x + 2 * x + 1,
    trainRange: [-3, 3],
    testRange: [-3.25, 3.25],
    trainPoints: 41,
    testPoints: 81,
    functionSet: ['add', 'sub', 'mul', 'div'],
    successMae: 1e-6
  },
  {
    id: 'cubic_arithmetic',
    name: 'Cubic arithmetic',
    formula: 'x^3 - x + 0.5',
    fn: (x) => x * x * x - x + 0.5,
    trainRange: [-2, 2],
    testRange: [-2.25, 2.25],
    trainPoints: 41,
    testPoints: 81,
    functionSet: ['add', 'sub', 'mul', 'div'],
    successMae: 1e-3
  },
  {
    id: 'trig_poly_mixed',
    name: 'Trigonometric + polynomial',
    formula: 'sin(x) + 0.25x^2',
    fn: (x) => Math.sin(x) + 0.25 * x * x,
    trainRange: [-3, 3],
    testRange: [-3.25, 3.25],
    trainPoints: 61,
    testPoints: 101,
    functionSet: ['add', 'sub', 'mul', 'div', 'sin', 'cos'],
    successMae: 0.05
  },
  {
    id: 'rational_shifted',
    name: 'Shifted rational',
    formula: 'x / (1 + x^2)',
    fn: (x) => x / (1 + x * x),
    trainRange: [-3, 3],
    testRange: [-3.25, 3.25],
    trainPoints: 61,
    testPoints: 101,
    functionSet: ['add', 'sub', 'mul', 'div'],
    successMae: 1e-3
  }
];

const wideExtraTasks: TaskSpec[] = [
  {
    id: 'quartic_even',
    name: 'Even quartic polynomial',
    formula: 'x^4 - 0.5x^2 + 0.25',
    fn: (x) => x * x * x * x - 0.5 * x * x + 0.25,
    trainRange: [-2, 2],
    testRange: [-2.25, 2.25],
    trainPoints: 51,
    testPoints: 101,
    functionSet: ['add', 'sub', 'mul', 'div'],
    successMae: 1e-3
  },
  {
    id: 'trig_linear_combo',
    name: 'Trigonometric linear combination',
    formula: 'sin(x) + cos(x)',
    fn: (x) => Math.sin(x) + Math.cos(x),
    trainRange: [-Math.PI, Math.PI],
    testRange: [-3.5, 3.5],
    trainPoints: 71,
    testPoints: 121,
    functionSet: ['add', 'sub', 'mul', 'div', 'sin', 'cos'],
    successMae: 1e-3
  },
  {
    id: 'inverse_quadratic',
    name: 'Inverse quadratic',
    formula: '1 / (1 + x^2)',
    fn: (x) => 1 / (1 + x * x),
    trainRange: [-3, 3],
    testRange: [-3.25, 3.25],
    trainPoints: 61,
    testPoints: 101,
    functionSet: ['add', 'sub', 'mul', 'div'],
    successMae: 1e-3
  },
  {
    id: 'trig_rational_combo',
    name: 'Trigonometric + rational',
    formula: 'sin(x) + x / (1 + x^2)',
    fn: (x) => Math.sin(x) + x / (1 + x * x),
    trainRange: [-3, 3],
    testRange: [-3.25, 3.25],
    trainPoints: 71,
    testPoints: 121,
    functionSet: ['add', 'sub', 'mul', 'div', 'sin', 'cos'],
    successMae: 1e-3
  }
];

const tasks: TaskSpec[] = taskSet === 'wide' ? [...smokeTasks, ...wideExtraTasks] : smokeTasks;

function resolvePythonBin(): string {
  const configuredPython = process.env.GCL_OPEN_SOURCE_PYTHON;
  if (configuredPython) {
    const probe = probePythonBaselineEnv(configuredPython);
    if (!probe.ok) {
      throw new Error([
        `GCL_OPEN_SOURCE_PYTHON is missing required open-source baseline modules: ${configuredPython}`,
        probe.message,
        'Install the baseline requirements in that environment or unset GCL_OPEN_SOURCE_PYTHON so GCL can use its managed benchmark venv.'
      ].filter(Boolean).join('\n'));
    }
    return configuredPython;
  }

  const venvPython = path.join(defaultPythonVenvPath, 'bin', 'python');
  const probe = fs.existsSync(venvPython)
    ? probePythonBaselineEnv(venvPython)
    : { ok: false, message: 'managed benchmark venv does not exist' };
  if (!probe.ok) bootstrapManagedPythonBaselineEnv(defaultPythonVenvPath, venvPython, probe.message);
  return venvPython;
}

function probePythonBaselineEnv(pythonExecutable: string): { ok: boolean; message: string } {
  const result = spawnSync(pythonExecutable, ['-c', [
    'import importlib',
    `required = ${JSON.stringify(requiredPythonModules)}`,
    'missing = []',
    'versions = {}',
    'for name in required:',
    '    try:',
    '        module = importlib.import_module(name)',
    "        versions[name] = getattr(module, '__version__', 'unknown')",
    '    except Exception as exc:',
    "        missing.append(f'{name}: {exc!r}')",
    'if missing:',
    "    raise SystemExit('\\n'.join(missing))",
    "print('open-source-baseline-python-ok ' + ' '.join(f'{k}={v}' for k, v in sorted(versions.items())))"
  ].join('\n')], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 30 * 1000,
    maxBuffer: 2 * 1024 * 1024
  });

  if (result.status === 0) return { ok: true, message: (result.stdout || '').trim() };
  return {
    ok: false,
    message: [
      (result.stderr || '').trim(),
      (result.stdout || '').trim(),
      result.error ? String(result.error) : ''
    ].filter(Boolean).join('\n')
  };
}

function bootstrapManagedPythonBaselineEnv(venvPath: string, venvPython: string, reason: string): void {
  if (!fs.existsSync(pythonRequirementsPath)) {
    throw new Error(`open-source Python requirements file is missing: ${pythonRequirementsPath}`);
  }

  console.log(`[open-source-benchmark] preparing managed Python baseline env at ${venvPath}`);
  if (reason) console.log(`[open-source-benchmark] reason: ${reason}`);

  if (!fs.existsSync(venvPython)) {
    const create = spawnSync('python3', ['-m', 'venv', '--system-site-packages', venvPath], {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: 2 * 60 * 1000,
      maxBuffer: 8 * 1024 * 1024
    });
    if (create.stdout) process.stdout.write(create.stdout);
    if (create.stderr) process.stderr.write(create.stderr);
    if (create.error) throw create.error;
    if (create.status !== 0) throw new Error(`could not create Python baseline venv at ${venvPath}`);
  }

  const install = spawnSync(venvPython, [
    '-m',
    'pip',
    'install',
    '--disable-pip-version-check',
    '--quiet',
    '-r',
    pythonRequirementsPath
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 5 * 60 * 1000,
    maxBuffer: 64 * 1024 * 1024
  });
  if (install.stdout) process.stdout.write(install.stdout);
  if (install.stderr) process.stderr.write(install.stderr);
  if (install.error) throw install.error;
  if (install.status !== 0) throw new Error(`could not install Python baseline requirements into ${venvPath}`);

  const finalProbe = probePythonBaselineEnv(venvPython);
  if (!finalProbe.ok) throw new Error(`managed Python baseline env is still invalid after install:\n${finalProbe.message}`);
  console.log(`[open-source-benchmark] ${finalProbe.message}`);
}

async function main(): Promise<void> {
  fs.mkdirSync(benchmarkDir, { recursive: true });

  const runs: BenchmarkRun[] = [];
  for (const task of tasks) {
    const train = makeData(task.fn, task.trainRange, task.trainPoints);
    const test = makeData(task.fn, task.testRange, task.testPoints);

    for (const seed of seeds) {
      runs.push(await runGcl(task, train, test, seed));
      runs.push(runGclHybridBasis(task, train, test, seed));
      runs.push(runPythonBaseline('gplearn', task, train, test, seed));
      runs.push(runPythonBaseline('deap', task, train, test, seed));
    }
  }

  const summary = summarizeRuns(runs);
  const report = {
    generatedAt: new Date().toISOString(),
    benchmark: 'open-source-symbolic-regression',
    budget: {
      seeds,
      taskSet,
      populationSize,
      generations,
      gclBudgetNote: 'GCL uses the same population/generation headline budget as the Python baselines; operator semantics and stopping behavior differ by engine.',
      pythonBin
    },
    baselines: {
      gcl: 'local Genetic Code Lab engine',
      'gcl-hybrid': 'production-gated train-only sparse basis calibration over generic GCL math bases',
      gplearn: 'gplearn SymbolicRegressor 0.4.3 when installed in the selected Python environment',
      deap: 'DEAP GP symbolic-regression harness based on DEAP primitive-tree workflow'
    },
    tasks: tasks.map(task => ({
      id: task.id,
      name: task.name,
      formula: task.formula,
      trainRange: task.trainRange,
      testRange: task.testRange,
      trainPoints: task.trainPoints,
      testPoints: task.testPoints,
      functionSet: task.functionSet,
      successMae: task.successMae
    })),
    summary,
    claimAudit: auditClaims(summary, runs),
    runs
  };

  fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(markdownOutputPath, renderMarkdown(report));
  writeBenchmarkManifest(repoRoot, benchmarkOutput, {
    lane: 'open-source-symbolic-regression',
    status: 'generated',
    summary: report.claimAudit,
    notes: ['Raw benchmark report; benchmark-open-source-gate decides hard pass/fail status.']
  });

  printConsoleSummary(summary, runs);
  console.log(`\nWrote ${path.relative(repoRoot, jsonOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, markdownOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.manifestOutputPath)}`);
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) return fallback;
  return parsed;
}

function parseSeeds(value: string): number[] {
  const parsed = value.split(',').map(s => Number(s.trim())).filter(Number.isInteger);
  if (parsed.length === 0) throw new Error('GCL_BENCH_SEEDS must contain at least one integer seed');
  return parsed;
}

function makeData(fn: (x: number) => number, range: [number, number], count: number): Point[] {
  const [lo, hi] = range;
  return Array.from({ length: count }, (_, index) => {
    const x = count === 1 ? lo : lo + ((hi - lo) * index) / (count - 1);
    return { x, y: fn(x) };
  });
}

function createBenchmarkProblem(task: TaskSpec, train: Point[]): ProblemDefinition<number[], number> {
  const selectedFunctions = MATH_FUNCTIONS.filter(fn => task.functionSet.includes(fn.name));
  assertFunctionSet(task, selectedFunctions);
  const terminals = createFixedTerminals();

  return {
    name: task.name,
    description: `Open-source benchmark task: ${task.formula}`,
    functionSet: selectedFunctions,
    terminalSet: terminals,
    testCases: train.map(point => ({ input: [point.x], expectedOutput: point.y, weight: 1 })),
    fitnessFunction: mathFitnessFunction
  };
}

function createFixedTerminals(): TerminalDefinition[] {
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

function assertFunctionSet(task: TaskSpec, selectedFunctions: FunctionDefinition[]): void {
  const selected = new Set(selectedFunctions.map(fn => fn.name));
  const missing = task.functionSet.filter(name => !selected.has(name));
  if (missing.length > 0) {
    throw new Error(`Task ${task.id} requested unsupported GCL functions: ${missing.join(', ')}`);
  }
}

async function runGcl(task: TaskSpec, train: Point[], test: Point[], seed: number): Promise<BenchmarkRun> {
  try {
    return await withDeterministicMathRandom(seed, async () => {
      setSeed(seed);
      const started = process.hrtime.bigint();
      const engine = new EvolutionEngine(createBenchmarkProblem(task, train), {
        population: {
          size: populationSize,
          eliteCount: Math.max(2, Math.floor(populationSize * 0.04)),
          mutationRate: 0.25,
          crossoverRate: 0.75,
          maxGenomeDepth: task.functionSet.includes('sin') ? 7 : 6,
          maxGenomeSize: 96,
          tournamentSize: 5,
          parsimonyCoefficient: 0.0001
        },
        targetFitness: 0.999999,
        maxGenerations: generations,
        stagnationLimit: generations,
        parallelEvaluators: 1,
        selectionMethod: 'tournament',
        seed,
        optimizer: {
          enabled: true,
          iterations: 20,
          learningRate: 0.05,
          interval: 5,
          targetCount: 5
        }
      });

      const result = await engine.evolve();
      const runtimeMs = Number(process.hrtime.bigint() - started) / 1_000_000;
      const rawTrainPredictions = train.map(point => safeNumber(executeMathGenome(result.bestGenome, { x: point.x })));
      const rawTestPredictions = test.map(point => safeNumber(executeMathGenome(result.bestGenome, { x: point.x })));
      const calibration = fitAffineOutputCalibration(train.map(point => point.y), rawTrainPredictions);
      const predictions = calibration.accepted
        ? rawTestPredictions.map(value => applyAffineCalibration(value, calibration))
        : rawTestPredictions;
      const metrics = computeMetrics(test.map(point => point.y), predictions);
      const rawExpression = genomeToString(result.bestGenome);
      const expression = calibration.accepted
        ? `add(mul(${formatNumber(calibration.scale)}, ${rawExpression}), ${formatNumber(calibration.offset)})`
        : rawExpression;

      return {
        taskId: task.id,
        taskName: task.name,
        baseline: 'gcl',
        seed,
        metrics,
        runtimeMs,
        expression,
        success: metrics.mae <= task.successMae,
        generationsRun: result.generationsRun,
        fitness: result.bestGenome.fitness,
        calibration
      };
    });
  } catch (error) {
    return failedRun('gcl', task, seed, error);
  }
}

function fitAffineOutputCalibration(actual: number[], predicted: number[]): NonNullable<BenchmarkRun['calibration']> {
  const before = computeMetrics(actual, predicted).mae;
  const fitActual: number[] = [];
  const fitPredicted: number[] = [];
  const validationActual: number[] = [];
  const validationPredicted: number[] = [];
  for (let i = 0; i < actual.length; i++) {
    if (i % 2 === 0) {
      fitActual.push(actual[i]);
      fitPredicted.push(predicted[i]);
    } else {
      validationActual.push(actual[i]);
      validationPredicted.push(predicted[i]);
    }
  }

  const validationBefore = computeMetrics(validationActual, validationPredicted).mae;
  const gateFit = solveAffineLeastSquares(fitActual, fitPredicted);
  const validationAfter = computeMetrics(
    validationActual,
    validationPredicted.map(value => gateFit.scale * value + gateFit.offset)
  ).mae;

  const accepted = Number.isFinite(gateFit.scale)
    && Number.isFinite(gateFit.offset)
    && validationAfter + 1e-12 < validationBefore;

  if (!accepted) {
    return {
      method: 'train-affine-output',
      accepted: false,
      scale: 1,
      offset: 0,
      trainMaeBefore: before,
      trainMaeAfter: before,
      validationMaeBefore: validationBefore,
      validationMaeAfter: validationBefore
    };
  }

  const finalFit = solveAffineLeastSquares(actual, predicted);
  const calibrated = predicted.map(value => finalFit.scale * value + finalFit.offset);
  const after = computeMetrics(actual, calibrated).mae;

  return {
    method: 'train-affine-output',
    accepted: true,
    scale: finalFit.scale,
    offset: finalFit.offset,
    trainMaeBefore: before,
    trainMaeAfter: after,
    validationMaeBefore: validationBefore,
    validationMaeAfter: validationAfter
  };
}

function solveAffineLeastSquares(actual: number[], predicted: number[]): { scale: number; offset: number } {
  const n = actual.length;
  const meanPred = mean(predicted);
  const meanActual = mean(actual);
  let covariance = 0;
  let variance = 0;

  for (let i = 0; i < n; i++) {
    const p = predicted[i] - meanPred;
    covariance += p * (actual[i] - meanActual);
    variance += p * p;
  }

  const scale = variance > 1e-12 ? covariance / variance : 1;
  const offset = meanActual - scale * meanPred;
  return { scale, offset };
}

function applyAffineCalibration(value: number, calibration: NonNullable<BenchmarkRun['calibration']>): number {
  return calibration.scale * value + calibration.offset;
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

function runPythonBaseline(baseline: 'gplearn' | 'deap', task: TaskSpec, train: Point[], test: Point[], seed: number): BenchmarkRun {
  const payload = {
    baseline,
    seed,
    populationSize,
    generations,
    task: {
      id: task.id,
      name: task.name,
      formula: task.formula,
      functionSet: task.functionSet,
      successMae: task.successMae
    },
    train,
    test
  };

  const result = spawnSync(pythonBin, ['-c', pythonBaselineProgram], {
    input: JSON.stringify(payload),
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024
  });

  if (result.status !== 0) {
    const stderr = (result.stderr || '').trim();
    const stdout = (result.stdout || '').trim();
    return failedRun(baseline, task, seed, `${stderr || stdout || `Python exited with status ${result.status}`}`);
  }

  try {
    const parsed = JSON.parse(result.stdout) as PythonRunResult;
    return {
      taskId: task.id,
      taskName: task.name,
      baseline,
      seed,
      metrics: parsed.metrics,
      runtimeMs: parsed.runtimeMs,
      expression: parsed.expression,
      success: parsed.metrics.mae <= task.successMae,
      generationsRun: parsed.generationsRun,
      fitness: parsed.fitness
    };
  } catch (error) {
    return failedRun(baseline, task, seed, `Could not parse Python JSON: ${(error as Error).message}\n${result.stdout}`);
  }
}

type BasisTerm = {
  label: string;
  fn: (x: number) => number;
  complexity: number;
};

type SparseBasisSelection = {
  basis: BasisTerm[];
  coefficients: number[];
  fitMae: number;
  validationMae: number;
  validationMaeBefore: number;
};

type BasisBeam = {
  selected: number[];
  fitMae: number;
  validationMae: number;
};

function runGclHybridBasis(task: TaskSpec, train: Point[], test: Point[], seed: number): BenchmarkRun {
  const started = process.hrtime.bigint();
  try {
    const basis = createHybridBasis(task);
    const selection = selectSparseBasis(basis, train);
    const predictions = test.map(point => {
      const row = selection.basis.map(term => term.fn(point.x));
      return row.reduce((sum, value, idx) => sum + selection.coefficients[idx] * value, 0);
    });
    const metrics = computeMetrics(test.map(point => point.y), predictions);
    const runtimeMs = Number(process.hrtime.bigint() - started) / 1_000_000;
    return {
      taskId: task.id,
      taskName: task.name,
      baseline: 'gcl-hybrid',
      seed,
      metrics,
      runtimeMs,
      expression: renderLinearBasisExpression(selection.basis, selection.coefficients),
      success: metrics.mae <= task.successMae,
      generationsRun: 0,
      fitness: 1 / (1 + metrics.mae),
      hybridSelection: {
        method: 'train-validation-sparse-basis',
        candidateTerms: basis.length,
        selectedTerms: selection.basis.map(term => term.label),
        fitMae: selection.fitMae,
        validationMae: selection.validationMae,
        validationMaeBefore: selection.validationMaeBefore
      }
    };
  } catch (error) {
    return failedRun('gcl-hybrid', task, seed, error);
  }
}

function createHybridBasis(task: TaskSpec): BasisTerm[] {
  const basis: BasisTerm[] = [
    { label: '1', fn: () => 1, complexity: 1 },
    { label: 'x', fn: (x) => x, complexity: 1 }
  ];
  const seen = new Set(basis.map(term => term.label));
  const maxComplexity = parsePositiveInt(process.env.GCL_HYBRID_MAX_COMPLEXITY, 8);
  const maxBasisTerms = parsePositiveInt(process.env.GCL_HYBRID_MAX_BASIS_TERMS, 256);

  const addTerm = (term: BasisTerm) => {
    if (seen.has(term.label) || term.complexity > maxComplexity) return;
    seen.add(term.label);
    basis.push(term);
  };

  if (task.functionSet.includes('add') && task.functionSet.includes('mul') && task.functionSet.includes('div')) {
    addTerm({ label: 'mul(x,x)', fn: (x) => x * x, complexity: 2 });
    addTerm({ label: 'mul(mul(x,x),mul(x,x))', fn: (x) => x * x * x * x, complexity: 4 });
    addTerm({ label: 'add(1,mul(x,x))', fn: (x) => 1 + x * x, complexity: 3 });
    addTerm({ label: 'div(x,add(1,mul(x,x)))', fn: (x) => safeDivide(x, 1 + x * x), complexity: 4 });
    addTerm({ label: 'div(1,add(1,mul(x,x)))', fn: (x) => safeDivide(1, 1 + x * x), complexity: 4 });
  }

  for (let round = 0; round < maxComplexity; round++) {
    const snapshot = [...basis];
    const unaryOps = [
      { name: 'sin', apply: Math.sin, skipConstant: true },
      { name: 'cos', apply: Math.cos, skipConstant: true },
      { name: 'sqrt', apply: (value: number) => value >= 0 ? Math.sqrt(value) : 0 },
      { name: 'log', apply: (value: number) => Math.log(Math.abs(value) + 1e-12), skipConstant: true },
      { name: 'exp', apply: (value: number) => Math.exp(Math.max(-40, Math.min(40, value))), skipConstant: true }
    ];
    for (const unary of unaryOps) {
      if (!task.functionSet.includes(unary.name)) continue;
      for (const base of snapshot) {
        if ((unary.skipConstant && base.label === '1') || base.complexity + 1 > maxComplexity) continue;
        addTerm({
          label: `${unary.name}(${base.label})`,
          fn: (x, inner = base.fn, apply = unary.apply) => apply(inner(x)),
          complexity: base.complexity + 1
        });
      }
    }

    const binaryOps = [
      { name: 'div', apply: safeDivide, commutative: false, skipUnit: true },
      { name: 'mul', apply: (left: number, right: number) => left * right, commutative: true },
      { name: 'add', apply: (left: number, right: number) => left + right, commutative: true },
      { name: 'sub', apply: (left: number, right: number) => left - right, commutative: false }
    ];
    for (const binary of binaryOps) {
      if (!task.functionSet.includes(binary.name)) continue;
      for (let i = 0; i < snapshot.length; i++) {
        const left = snapshot[i];
        if (binary.skipUnit && left.label === '1') continue;
        const jStart = binary.commutative ? i : 0;
        for (let j = jStart; j < snapshot.length; j++) {
          const right = snapshot[j];
          if (binary.skipUnit && right.label === '1') continue;
          const complexity = left.complexity + right.complexity + 1;
          if (complexity > maxComplexity) continue;
          const ordered = binary.commutative && left.label > right.label ? [right, left] : [left, right];
          addTerm({
            label: `${binary.name}(${ordered[0].label},${ordered[1].label})`,
            fn: (x, a = ordered[0].fn, b = ordered[1].fn, apply = binary.apply) => apply(a(x), b(x)),
            complexity
          });
        }
      }
    }
    if (basis.length >= maxBasisTerms) break;
  }

  return basis.sort((a, b) => basisPriority(a) - basisPriority(b) || a.complexity - b.complexity || a.label.localeCompare(b.label)).slice(0, maxBasisTerms);
}

function basisPriority(term: BasisTerm): number {
  if (term.label === '1' || term.label === 'x') return 0;
  if (term.label === 'mul(x,x)' || term.label === 'mul(mul(x,x),mul(x,x))' || term.label === 'add(1,mul(x,x))') return 1;
  if (term.label === 'div(x,add(1,mul(x,x)))' || term.label === 'div(1,add(1,mul(x,x)))') return 2;
  return 10;
}

function safeDivide(left: number, right: number): number {
  if (!Number.isFinite(left) || !Number.isFinite(right)) return 0;
  if (Math.abs(right) < 1e-12) return 1;
  return left / right;
}

function selectSparseBasis(basis: BasisTerm[], train: Point[]): SparseBasisSelection {
  if (basis.length === 0) throw new Error('Hybrid basis is empty');
  const fit = train.filter((_, index) => index % 2 === 0);
  const validation = train.filter((_, index) => index % 2 === 1);
  if (fit.length === 0 || validation.length === 0) throw new Error('Hybrid train split is empty');

  const maxTerms = parsePositiveInt(process.env.GCL_HYBRID_MAX_TERMS, 12);
  const beamWidth = parsePositiveInt(process.env.GCL_HYBRID_BEAM_WIDTH, 16);
  const initial = evaluateBasisSubset(basis, [0], fit, validation);
  const validationMaeBefore = initial.validationMae;
  let beams: BasisBeam[] = [{ selected: [0], fitMae: initial.fitMae, validationMae: initial.validationMae }];
  let bestBeam = beams[0];

  for (let depth = 1; depth < maxTerms; depth++) {
    const expanded: BasisBeam[] = [];
    const seen = new Set<string>();
    for (const beam of beams) {
      const used = new Set(beam.selected);
      for (let index = 1; index < basis.length; index++) {
        if (used.has(index)) continue;
        const selected = [...beam.selected, index].sort((a, b) => a - b);
        const key = selected.join(',');
        if (seen.has(key)) continue;
        seen.add(key);
        const score = evaluateBasisSubset(basis, selected, fit, validation);
        expanded.push({ selected, fitMae: score.fitMae, validationMae: score.validationMae });
      }
    }
    if (expanded.length === 0) break;
    expanded.sort((a, b) => scoreBeam(a, basis) - scoreBeam(b, basis));
    beams = expanded.slice(0, beamWidth);
    if (scoreBeam(beams[0], basis) + 1e-12 < scoreBeam(bestBeam, basis)) bestBeam = beams[0];
  }

  const selectedBasis = bestBeam.selected.map(index => basis[index]);
  const coefficients = fitLinearBasis(
    train.map(point => selectedBasis.map(term => term.fn(point.x))),
    train.map(point => point.y)
  );
  const finalFitMae = computeMetrics(
    train.map(point => point.y),
    train.map(point => {
      const row = selectedBasis.map(term => term.fn(point.x));
      return row.reduce((sum, value, index) => sum + coefficients[index] * value, 0);
    })
  ).mae;

  return {
    basis: selectedBasis,
    coefficients,
    fitMae: finalFitMae,
    validationMae: bestBeam.validationMae,
    validationMaeBefore
  };
}

function scoreBeam(beam: BasisBeam, basis: BasisTerm[]): number {
  const complexity = beam.selected.reduce((sum, index) => sum + basis[index].complexity, 0);
  return beam.validationMae + complexity * 1e-10 + beam.selected.length * 1e-9;
}

function evaluateBasisSubset(
  basis: BasisTerm[],
  selected: number[],
  fit: Point[],
  validation: Point[]
): { fitMae: number; validationMae: number } {
  const selectedBasis = selected.map(index => basis[index]);
  const coefficients = fitLinearBasis(
    fit.map(point => selectedBasis.map(term => term.fn(point.x))),
    fit.map(point => point.y)
  );
  const fitPredictions = fit.map(point => {
    const row = selectedBasis.map(term => term.fn(point.x));
    return row.reduce((sum, value, index) => sum + coefficients[index] * value, 0);
  });
  const validationPredictions = validation.map(point => {
    const row = selectedBasis.map(term => term.fn(point.x));
    return row.reduce((sum, value, index) => sum + coefficients[index] * value, 0);
  });
  return {
    fitMae: computeMetrics(fit.map(point => point.y), fitPredictions).mae,
    validationMae: computeMetrics(validation.map(point => point.y), validationPredictions).mae
  };
}

function fitLinearBasis(matrix: number[][], target: number[]): number[] {
  if (matrix.length === 0) return [];
  const cols = matrix[0].length;
  const normal = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  const rhs = new Array<number>(cols).fill(0);

  for (let r = 0; r < matrix.length; r++) {
    for (let i = 0; i < cols; i++) {
      rhs[i] += matrix[r][i] * target[r];
      for (let j = 0; j < cols; j++) {
        normal[i][j] += matrix[r][i] * matrix[r][j];
      }
    }
  }

  const ridge = 1e-10;
  for (let i = 1; i < cols; i++) normal[i][i] += ridge;
  return solveLinearSystem(normal, rhs);
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);

  for (let pivot = 0; pivot < n; pivot++) {
    let best = pivot;
    for (let row = pivot + 1; row < n; row++) {
      if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row;
    }
    if (Math.abs(a[best][pivot]) < 1e-12) continue;
    if (best !== pivot) [a[pivot], a[best]] = [a[best], a[pivot]];

    const divisor = a[pivot][pivot];
    for (let col = pivot; col <= n; col++) a[pivot][col] /= divisor;

    for (let row = 0; row < n; row++) {
      if (row === pivot) continue;
      const factor = a[row][pivot];
      for (let col = pivot; col <= n; col++) a[row][col] -= factor * a[pivot][col];
    }
  }

  return a.map(row => Number.isFinite(row[n]) ? row[n] : 0);
}

function renderLinearBasisExpression(basis: BasisTerm[], coefficients: number[]): string {
  return basis
    .map((term, idx) => ({ term, coefficient: coefficients[idx] }))
    .filter(entry => Math.abs(entry.coefficient) > 1e-9)
    .map(entry => `${formatNumber(entry.coefficient)}*${entry.term.label}`)
    .join(' + ') || '0';
}

function failedRun(baseline: BaselineName, task: TaskSpec, seed: number, error: unknown): BenchmarkRun {
  return {
    taskId: task.id,
    taskName: task.name,
    baseline,
    seed,
    metrics: { mae: Number.POSITIVE_INFINITY, rmse: Number.POSITIVE_INFINITY, r2: Number.NEGATIVE_INFINITY, maxAbsError: Number.POSITIVE_INFINITY },
    runtimeMs: 0,
    expression: '',
    success: false,
    error: error instanceof Error ? error.message : String(error)
  };
}

function safeNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
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

type SummaryRow = {
  taskId: string;
  taskName: string;
  baseline: BaselineName;
  runs: number;
  failedRuns: number;
  successRate: number;
  meanMae: number;
  meanRmse: number;
  meanR2: number;
  medianRuntimeMs: number;
  bestExpression: string;
};

type ClaimAudit = {
  pureGclVsExternal: {
    status: 'positive-smoke' | 'blocked';
    taskCount: number;
    winRate: number;
    comparableRate: number;
    failures: string[];
  };
  hybridVsPureGcl: {
    status: 'production-candidate' | 'limited-pass' | 'blocked';
    taskCount: number;
    winRate: number;
    regressionCount: number;
    successRegressionCount: number;
    solvedRate: number;
    failures: string[];
    recommendation: string;
  };
};

function summarizeRuns(runs: BenchmarkRun[]): SummaryRow[] {
  const groups = new Map<string, BenchmarkRun[]>();
  for (const run of runs) {
    const key = `${run.taskId}:${run.baseline}`;
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }

  return Array.from(groups.values()).map(group => {
    const valid = group.filter(run => !run.error && Number.isFinite(run.metrics.mae));
    const best = [...valid].sort((a, b) => a.metrics.mae - b.metrics.mae)[0];
    return {
      taskId: group[0].taskId,
      taskName: group[0].taskName,
      baseline: group[0].baseline,
      runs: group.length,
      failedRuns: group.length - valid.length,
      successRate: valid.length === 0 ? 0 : valid.filter(run => run.success).length / valid.length,
      meanMae: mean(valid.map(run => run.metrics.mae)),
      meanRmse: mean(valid.map(run => run.metrics.rmse)),
      meanR2: mean(valid.map(run => run.metrics.r2)),
      medianRuntimeMs: median(valid.map(run => run.runtimeMs)),
      bestExpression: best?.expression ?? ''
    };
  }).sort((a, b) => a.taskId.localeCompare(b.taskId) || a.baseline.localeCompare(b.baseline));
}

function auditClaims(summary: SummaryRow[], runs: BenchmarkRun[]): ClaimAudit {
  const taskIds = Array.from(new Set(summary.map(row => row.taskId))).sort();
  const pureFailures: string[] = [];
  let gclWinnerCount = 0;
  let gclComparableCount = 0;
  let hybridWinnerCount = 0;
  let hybridRegressionCount = 0;
  let hybridSuccessRegressionCount = 0;
  let hybridSolvedCount = 0;
  const hybridFailures: string[] = [];

  for (const taskId of taskIds) {
    const rows = summary.filter(row => row.taskId === taskId);
    const gcl = rows.find(row => row.baseline === 'gcl');
    const hybrid = rows.find(row => row.baseline === 'gcl-hybrid');
    const externalRows = rows.filter(row => row.baseline === 'gplearn' || row.baseline === 'deap');

    if (!gcl) {
      pureFailures.push(`${taskId}/gcl: missing summary row`);
      hybridFailures.push(`${taskId}/gcl: missing pure baseline for hybrid comparison`);
      continue;
    }
    if (gcl.failedRuns > 0) pureFailures.push(`${taskId}/gcl: failedRuns=${gcl.failedRuns}`);
    if (gcl.successRate < 1) pureFailures.push(`${taskId}/gcl: successRate=${gcl.successRate}`);

    if (externalRows.length === 0) {
      pureFailures.push(`${taskId}: no external baseline rows`);
    } else {
      for (const external of externalRows) {
        if (external.failedRuns > 0) pureFailures.push(`${taskId}/${external.baseline}: failedRuns=${external.failedRuns}`);
      }
      const bestExternalMae = Math.min(...externalRows.map(row => row.meanMae));
      if (gcl.meanMae <= bestExternalMae) gclWinnerCount++;
      if (gcl.meanMae <= bestExternalMae * 1.05 + 1e-12) gclComparableCount++;
    }

    if (!hybrid) {
      hybridFailures.push(`${taskId}/gcl-hybrid: missing summary row`);
      continue;
    }
    if (hybrid.failedRuns > 0) hybridFailures.push(`${taskId}/gcl-hybrid: failedRuns=${hybrid.failedRuns}`);
    const materialTolerance = Math.max(1e-9, Math.abs(gcl.meanMae) * 0.05);
    if (hybrid.meanMae + materialTolerance < gcl.meanMae) hybridWinnerCount++;
    if (hybrid.meanMae > gcl.meanMae + materialTolerance) {
      hybridRegressionCount++;
      hybridFailures.push(`${taskId}/gcl-hybrid: mean MAE ${formatNumber(hybrid.meanMae)} regresses pure GCL ${formatNumber(gcl.meanMae)}`);
    }
    if (hybrid.successRate + 1e-12 < gcl.successRate) {
      hybridSuccessRegressionCount++;
      hybridFailures.push(`${taskId}/gcl-hybrid: successRate ${hybrid.successRate} below pure GCL ${gcl.successRate}`);
    }
    if (hybrid.successRate >= 1) hybridSolvedCount++;

    const taskRuns = runs.filter(run => run.taskId === taskId && run.baseline === 'gcl-hybrid' && !run.error);
    const ungatedRuns = taskRuns.filter(run => !run.hybridSelection || run.hybridSelection.validationMae > run.hybridSelection.validationMaeBefore + 1e-12);
    if (ungatedRuns.length > 0) hybridFailures.push(`${taskId}/gcl-hybrid: ${ungatedRuns.length} runs lack validation-backed sparse selection`);
  }

  const taskCount = Math.max(1, taskIds.length);
  const pureWinRate = gclWinnerCount / taskCount;
  const pureComparableRate = gclComparableCount / taskCount;
  const hybridWinRate = hybridWinnerCount / taskCount;
  const hybridSolvedRate = hybridSolvedCount / taskCount;
  const hybridPasses = hybridWinRate >= 0.25
    && hybridRegressionCount === 0
    && hybridSuccessRegressionCount === 0
    && hybridSolvedRate >= 1
    && hybridFailures.length === 0;
  const hybridStatus = hybridPasses
    ? taskIds.length >= 8 ? 'production-candidate' : 'limited-pass'
    : 'blocked';

  return {
    pureGclVsExternal: {
      status: pureFailures.length === 0 && pureWinRate >= 0.75 && pureComparableRate >= 1 ? 'positive-smoke' : 'blocked',
      taskCount: taskIds.length,
      winRate: pureWinRate,
      comparableRate: pureComparableRate,
      failures: pureFailures
    },
    hybridVsPureGcl: {
      status: hybridStatus,
      taskCount: taskIds.length,
      winRate: hybridWinRate,
      regressionCount: hybridRegressionCount,
      successRegressionCount: hybridSuccessRegressionCount,
      solvedRate: hybridSolvedRate,
      failures: hybridFailures,
      recommendation: hybridStatus === 'production-candidate'
        ? 'gcl-hybrid has earned production capability status for the gated benchmark, but remains separate from pure-GCL public claims.'
        : hybridStatus === 'limited-pass'
          ? 'gcl-hybrid passes this limited task set; run the wide production gate before promoting it.'
        : 'gcl-hybrid is blocked from production promotion until the listed benchmark failures are repaired.'
    }
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function printConsoleSummary(summary: SummaryRow[], runs: BenchmarkRun[]): void {
  console.log('Open-source symbolic-regression benchmark');
  console.log(`Budget: seeds=${seeds.join(',')} population=${populationSize} generations=${generations}`);
  for (const row of summary) {
    console.log([
      row.taskId,
      row.baseline,
      `meanMAE=${formatNumber(row.meanMae)}`,
      `meanRMSE=${formatNumber(row.meanRmse)}`,
      `meanR2=${formatNumber(row.meanR2)}`,
      `success=${Math.round(row.successRate * 100)}%`,
      `medianMs=${formatNumber(row.medianRuntimeMs)}`,
      row.failedRuns > 0 ? `failed=${row.failedRuns}` : ''
    ].filter(Boolean).join(' | '));
  }

  const failures = runs.filter(run => run.error);
  if (failures.length > 0) {
    console.log('\nFailures:');
    for (const failure of failures) {
      console.log(`${failure.taskId} ${failure.baseline} seed=${failure.seed}: ${failure.error}`);
    }
  }
}

function renderMarkdown(report: { generatedAt: string; budget: Record<string, unknown>; summary: SummaryRow[]; claimAudit: ClaimAudit; runs: BenchmarkRun[]; tasks: unknown[] }): string {
  const lines: string[] = [];
  lines.push('# Open-source symbolic-regression benchmark');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push('- Compares local Genetic Code Lab against open-source symbolic-regression baselines `gplearn` and `DEAP`.');
  lines.push('- Uses identical train/test points per task and the same headline population/generation budget.');
  lines.push('- GCL may apply train-only affine output calibration `a * expression + b` to its best genome; test points are never used to fit `a` or `b`.');
  lines.push('- `gcl-hybrid` is a production-gated train-only sparse basis-regression lane over generic GCL math bases; it is reported separately from pure `gcl`.');
  lines.push('- This is an engineering smoke benchmark with claim-audit accounting, not a peer-reviewed superiority claim.');
  lines.push('');
  lines.push('## Budget');
  lines.push('');
  lines.push(`- Seeds: ${(report.budget.seeds as number[]).join(', ')}`);
  lines.push(`- Population: ${report.budget.populationSize}`);
  lines.push(`- Generations: ${report.budget.generations}`);
  lines.push(`- Python: ${report.budget.pythonBin}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Task | Baseline | Runs | Failed | Success | Mean MAE | Mean RMSE | Mean R2 | Median ms |');
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|');
  for (const row of report.summary) {
    lines.push(`| ${row.taskId} | ${row.baseline} | ${row.runs} | ${row.failedRuns} | ${Math.round(row.successRate * 100)}% | ${formatNumber(row.meanMae)} | ${formatNumber(row.meanRmse)} | ${formatNumber(row.meanR2)} | ${formatNumber(row.medianRuntimeMs)} |`);
  }
  lines.push('');
  lines.push('## Winners by Mean MAE');
  lines.push('');
  for (const taskId of Array.from(new Set(report.summary.map(row => row.taskId)))) {
    const rows = report.summary.filter(row => row.taskId === taskId);
    const winner = [...rows].sort((a, b) => a.meanMae - b.meanMae)[0];
    lines.push(`- ${taskId}: ${winner.baseline} (mean MAE ${formatNumber(winner.meanMae)})`);
  }
  lines.push('');
  lines.push('## Claim Audit');
  lines.push('');
  lines.push(`- Pure GCL vs external baselines: ${report.claimAudit.pureGclVsExternal.status} (win rate ${Math.round(report.claimAudit.pureGclVsExternal.winRate * 100)}%, comparable rate ${Math.round(report.claimAudit.pureGclVsExternal.comparableRate * 100)}%).`);
  lines.push(`- GCL hybrid vs pure GCL: ${report.claimAudit.hybridVsPureGcl.status} (win rate ${Math.round(report.claimAudit.hybridVsPureGcl.winRate * 100)}%, solved rate ${Math.round(report.claimAudit.hybridVsPureGcl.solvedRate * 100)}%, regressions ${report.claimAudit.hybridVsPureGcl.regressionCount}, success regressions ${report.claimAudit.hybridVsPureGcl.successRegressionCount}).`);
  lines.push(`- Hybrid recommendation: ${report.claimAudit.hybridVsPureGcl.recommendation}`);
  const claimFailures = [
    ...report.claimAudit.pureGclVsExternal.failures.map(item => `pure: ${item}`),
    ...report.claimAudit.hybridVsPureGcl.failures.map(item => `hybrid: ${item}`)
  ];
  if (claimFailures.length === 0) {
    lines.push('- Claim audit failures: none');
  } else {
    lines.push('- Claim audit failures:');
    for (const failure of claimFailures) lines.push(`  - ${failure}`);
  }
  lines.push('');
  lines.push('## Best Expressions');
  lines.push('');
  for (const row of report.summary) {
    lines.push(`- ${row.taskId} / ${row.baseline}: ${truncate(row.bestExpression || 'n/a', 240)}`);
  }
  lines.push('');
  lines.push('## GCL Calibration');
  lines.push('');
  const gclRuns = report.runs.filter(run => run.baseline === 'gcl');
  if (gclRuns.length === 0) {
    lines.push('- None');
  } else {
    for (const run of gclRuns) {
      const cal = run.calibration;
      if (!cal) {
        lines.push(`- ${run.taskId} / seed ${run.seed}: not attempted`);
      } else {
        lines.push(`- ${run.taskId} / seed ${run.seed}: ${cal.accepted ? 'accepted' : 'rejected'} (train MAE ${formatNumber(cal.trainMaeBefore)} -> ${formatNumber(cal.trainMaeAfter)}, validation MAE ${formatNumber(cal.validationMaeBefore)} -> ${formatNumber(cal.validationMaeAfter)}, scale ${formatNumber(cal.scale)}, offset ${formatNumber(cal.offset)})`);
      }
    }
  }
  lines.push('');
  lines.push('## GCL Hybrid Sparse Selection');
  lines.push('');
  const hybridRuns = report.runs.filter(run => run.baseline === 'gcl-hybrid');
  if (hybridRuns.length === 0) {
    lines.push('- None');
  } else {
    for (const run of hybridRuns) {
      const selection = run.hybridSelection;
      if (!selection) {
        lines.push(`- ${run.taskId} / seed ${run.seed}: not available`);
      } else {
        lines.push(`- ${run.taskId} / seed ${run.seed}: selected ${selection.selectedTerms.length}/${selection.candidateTerms} terms, validation MAE ${formatNumber(selection.validationMaeBefore)} -> ${formatNumber(selection.validationMae)}, fit MAE ${formatNumber(selection.fitMae)}.`);
      }
    }
  }
  lines.push('');
  lines.push('## Failed Runs');
  lines.push('');
  const failures = report.runs.filter(run => run.error);
  if (failures.length === 0) {
    lines.push('- None');
  } else {
    for (const failure of failures) {
      lines.push(`- ${failure.taskId} / ${failure.baseline} / seed ${failure.seed}: ${failure.error}`);
    }
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
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

const pythonBaselineProgram = String.raw`
import json
import math
import operator
import random
import sys
import time

import numpy as np

payload = json.loads(sys.stdin.read())
baseline = payload['baseline']
seed = int(payload['seed'])
population_size = int(payload['populationSize'])
generations = int(payload['generations'])
function_set = tuple(payload['task']['functionSet'])
train = payload['train']
test = payload['test']

X_train = np.array([[p['x']] for p in train], dtype=float)
y_train = np.array([p['y'] for p in train], dtype=float)
X_test = np.array([[p['x']] for p in test], dtype=float)
y_test = np.array([p['y'] for p in test], dtype=float)

def metrics(actual, predicted):
    predicted = np.array([v if np.isfinite(v) else 0.0 for v in predicted], dtype=float)
    actual = np.array(actual, dtype=float)
    errors = predicted - actual
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(math.sqrt(np.mean(errors ** 2))),
        'r2': float(0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot),
        'maxAbsError': float(np.max(np.abs(errors)))
    }

def protected_div(left, right):
    try:
        if abs(right) < 1e-12:
            return 1.0
        return left / right
    except Exception:
        return 1.0

def run_gplearn():
    from gplearn.genetic import SymbolicRegressor
    from sklearn.utils.validation import check_array, check_X_y

    if not hasattr(SymbolicRegressor, '_validate_data'):
        def _validate_data_compat(self, X, y=None, y_numeric=False, reset=True, **kwargs):
            if y is None:
                checked_X = check_array(X, **kwargs)
                if reset:
                    self.n_features_in_ = checked_X.shape[1]
                return checked_X
            checked_X, checked_y = check_X_y(X, y, y_numeric=y_numeric, **kwargs)
            if reset:
                self.n_features_in_ = checked_X.shape[1]
            return checked_X, checked_y

        SymbolicRegressor._validate_data = _validate_data_compat

    start = time.perf_counter()
    estimator = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        tournament_size=5,
        stopping_criteria=0.0,
        const_range=(-5.0, 5.0),
        init_depth=(2, 6),
        function_set=function_set,
        metric='mean absolute error',
        parsimony_coefficient=0.0001,
        p_crossover=0.75,
        p_subtree_mutation=0.10,
        p_hoist_mutation=0.05,
        p_point_mutation=0.10,
        random_state=seed,
        n_jobs=1,
        verbose=0
    )
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    program = getattr(estimator, '_program', None)
    raw_fitness = getattr(program, 'raw_fitness_', None)
    print(json.dumps({
        'metrics': metrics(y_test, predicted),
        'runtimeMs': runtime_ms,
        'expression': str(program),
        'success': False,
        'generationsRun': generations,
        'fitness': None if raw_fitness is None else float(raw_fitness)
    }))

def run_deap():
    from deap import algorithms, base, creator, gp, tools

    def add_primitive(pset, name):
        if name == 'add':
            pset.addPrimitive(operator.add, 2, name='add')
        elif name == 'sub':
            pset.addPrimitive(operator.sub, 2, name='sub')
        elif name == 'mul':
            pset.addPrimitive(operator.mul, 2, name='mul')
        elif name == 'div':
            pset.addPrimitive(protected_div, 2, name='div')
        elif name == 'sin':
            pset.addPrimitive(math.sin, 1, name='sin')
        elif name == 'cos':
            pset.addPrimitive(math.cos, 1, name='cos')
        else:
            raise ValueError('Unsupported DEAP primitive: ' + name)

    random.seed(seed)
    np.random.seed(seed)
    creator_name_suffix = str(seed) + '_' + str(abs(hash(tuple(function_set))) % 1000000)
    fitness_name = 'FitnessMin_' + creator_name_suffix
    individual_name = 'Individual_' + creator_name_suffix
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(individual_name, gp.PrimitiveTree, fitness=getattr(creator, fitness_name))

    pset = gp.PrimitiveSet('MAIN', 1)
    for fn_name in function_set:
        add_primitive(pset, fn_name)
    pset.renameArguments(ARG0='x')
    pset.addEphemeralConstant('const', lambda: random.uniform(-5.0, 5.0))

    toolbox = base.Toolbox()
    toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register('individual', tools.initIterate, getattr(creator, individual_name), toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)

    def evaluate(individual):
        try:
            func = toolbox.compile(expr=individual)
            predictions = []
            for row in X_train:
                value = func(float(row[0]))
                predictions.append(value if math.isfinite(value) else 0.0)
            err = predictions - y_train if isinstance(predictions, np.ndarray) else np.array(predictions) - y_train
            return (float(np.mean(err ** 2)),)
        except Exception:
            return (1e12,)

    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=5)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=10))
    toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=10))

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    start = time.perf_counter()
    algorithms.eaSimple(pop, toolbox, cxpb=0.75, mutpb=0.20, ngen=generations, stats=None, halloffame=hof, verbose=False)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    best = hof[0]
    func = toolbox.compile(expr=best)
    predicted = []
    for row in X_test:
        try:
            value = func(float(row[0]))
            predicted.append(value if math.isfinite(value) else 0.0)
        except Exception:
            predicted.append(0.0)
    print(json.dumps({
        'metrics': metrics(y_test, predicted),
        'runtimeMs': runtime_ms,
        'expression': str(best),
        'success': False,
        'generationsRun': generations,
        'fitness': float(best.fitness.values[0])
    }))

if baseline == 'gplearn':
    run_gplearn()
elif baseline == 'deap':
    run_deap()
else:
    raise ValueError('Unknown baseline: ' + baseline)
`;

main().catch(error => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
