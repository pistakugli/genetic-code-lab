import * as fs from 'fs';
import * as path from 'path';
import { EvolutionEngine } from '../src/core/evolution';
import { MigrationHub } from '../src/core/migration-hub';
import { MLP } from '../src/core/neural-network';
import { ProblemDefinition, TerminalDefinition, Genome } from '../src/core/types';
import { MATH_FUNCTIONS, executeMathGenome, mathFitnessFunction } from '../src/solvers/math-expression';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Point = { input: number[]; expected: number };
type MetricSet = { mae: number; rmse: number; r2: number; maxAbsError: number };

type CandidateRecord = {
  islandId: number;
  generation: number;
  trainMae: number;
  validationMae: number;
  guardMae: number;
  complexity: number;
  fitness: number;
  archiveInjections: number;
  archiveSize: number;
  genome: Genome;
};

type IslandResult = {
  islandId: number;
  role: string;
  metrics: MetricSet;
  validationMae: number;
  guardMae: number;
  archive: {
    recordCycles: number;
    injectionAttempts: number;
    injectionSuccesses: number;
    size: number;
  };
};

type SeedResult = {
  seed: number;
  single: {
    validationMetrics: MetricSet;
    guardMetrics: MetricSet;
    metrics: MetricSet;
    archiveInjections: number;
  };
  islands: IslandResult[];
  validationSelected: {
    islandId: number;
    metrics: MetricSet;
    validationMae: number;
    guardMae: number;
  };
  nnSelected: {
    islandId: number;
    probability: number;
    metrics: MetricSet;
    validationMae: number;
    guardMae: number;
    fallbackUsed: boolean;
    fallbackReason: string;
  };
  nn: {
    samples: number;
    positives: number;
    trainingAccuracy: number;
  };
  migration: {
    drops: number;
    pullAttempts: number;
    pullSuccesses: number;
    archiveInjectionAttempts: number;
    archiveInjections: number;
  };
  liftVsSingle: number;
  singleSolved: boolean;
  outcome: 'single-baseline-solved' | 'single-baseline-fallback' | 'island-nn-lift' | 'island-nn-miss' | 'runtime-timeout' | 'runtime-error';
  success: boolean;
  completed: boolean;
  error?: string;
};

const repoRoot = path.resolve(__dirname, '..');
const seeds = parseSeeds(process.env.GCL_ISLAND_NN_SEEDS ?? '4,7,1337');
const benchmarkOutput = resolveBenchmarkOutput(repoRoot, 'island-nn-archive', seeds.length >= 24 ? 'wide' : 'smoke');
const benchmarkDir = benchmarkOutput.outputDir;
const jsonOutputPath = benchmarkOutput.jsonOutputPath;
const markdownOutputPath = benchmarkOutput.markdownOutputPath;
const islandCount = parsePositiveInt(process.env.GCL_ISLAND_NN_COUNT, 4);
const islandPopulation = parsePositiveInt(process.env.GCL_ISLAND_NN_POPULATION, 90);
const singlePopulation = parsePositiveInt(process.env.GCL_ISLAND_NN_SINGLE_POPULATION, 180);
const generations = parsePositiveInt(process.env.GCL_ISLAND_NN_GENERATIONS, 90);
const successLift = Number(process.env.GCL_ISLAND_NN_MIN_LIFT ?? '1.05');
const nnAcceptThreshold = Number(process.env.GCL_ISLAND_NN_ACCEPT_P ?? '0.58');
const validationVetoRatio = Number(process.env.GCL_ISLAND_NN_VALIDATION_VETO_RATIO ?? '1.05');
const singleSolvedMae = Number(process.env.GCL_ISLAND_NN_SINGLE_SOLVED_MAE ?? '0.001');
const singleBaselineVetoRatio = Number(process.env.GCL_ISLAND_NN_SINGLE_VETO_RATIO ?? '1.05');
const singleBaselineGuardVetoRatio = Number(process.env.GCL_ISLAND_NN_SINGLE_GUARD_VETO_RATIO ?? '1.20');
const nonRegressionTolerance = Number(process.env.GCL_ISLAND_NN_NON_REGRESSION_TOLERANCE ?? '1.00');
const seedTimeoutMs = parsePositiveInt(process.env.GCL_ISLAND_NN_SEED_TIMEOUT_MS, 120000);

const roles = [
  { name: 'full', functions: ['add', 'sub', 'mul', 'div', 'sin', 'cos'] },
  { name: 'arithmetic', functions: ['add', 'sub', 'mul', 'div'] },
  { name: 'trig', functions: ['add', 'sub', 'mul', 'sin', 'cos'] },
  { name: 'rational-trig', functions: ['add', 'sub', 'mul', 'div', 'sin', 'cos'] }
];

async function main(): Promise<void> {
  fs.mkdirSync(benchmarkDir, { recursive: true });
  const train = makeGrid([-2, 2], 9, [-1.75, 1.75], 8);
  const validation = makeGrid([-2.1, 2.1], 8, [-1.9, 1.9], 7);
  const guard = makeGrid([-2.2, 2.2], 9, [-1.95, 1.95], 9);
  const test = makeGrid([-2.25, 2.25], 13, [-2, 2], 12);

  const results: SeedResult[] = [];
  for (const seed of seeds) {
    try {
      results.push(await withTimeout(runSeed(seed, train, validation, guard, test), seedTimeoutMs));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      results.push(makeFailedSeedResult(seed, message.includes('timed out') ? 'runtime-timeout' : 'runtime-error', message));
    }
  }
  const completedResults = results.filter(result => result.completed);
  const eligibleResults = completedResults.filter(result => !result.singleSolved);

  const summary = {
    successRate: results.filter(result => result.success).length / Math.max(1, results.length),
    nonDegenerateSuccessRate: eligibleResults.filter(result => result.success).length / Math.max(1, eligibleResults.length),
    nonRegressionRate: eligibleResults.filter(result => result.nnSelected.metrics.mae <= result.single.metrics.mae * nonRegressionTolerance).length / Math.max(1, eligibleResults.length),
    completedSeedCount: completedResults.length,
    timedOutCount: results.filter(result => result.outcome === 'runtime-timeout').length,
    runtimeErrorCount: results.filter(result => result.outcome === 'runtime-error').length,
    singleSolvedCount: completedResults.length - eligibleResults.length,
    singleFallbackCount: completedResults.filter(result => result.outcome === 'single-baseline-fallback').length,
    meanSingleMae: mean(completedResults.map(result => result.single.metrics.mae)),
    meanNnMae: mean(completedResults.map(result => result.nnSelected.metrics.mae)),
    meanValidationSelectedMae: mean(completedResults.map(result => result.validationSelected.metrics.mae)),
    meanLiftVsSingle: mean(completedResults.map(result => result.liftVsSingle)),
    meanEligibleLiftVsSingle: mean(eligibleResults.map(result => result.liftVsSingle)),
    medianLiftVsSingle: median(completedResults.map(result => result.liftVsSingle)),
    medianEligibleLiftVsSingle: median(eligibleResults.map(result => result.liftVsSingle)),
    totalArchiveInjections: completedResults.reduce((sum, result) => sum + result.migration.archiveInjections, 0),
    totalHubPullSuccesses: completedResults.reduce((sum, result) => sum + result.migration.pullSuccesses, 0),
    fallbackCount: completedResults.filter(result => result.nnSelected.fallbackUsed).length,
    meanNnTrainingAccuracy: mean(completedResults.map(result => result.nn.trainingAccuracy))
  };

  const report = {
    generatedAt: new Date().toISOString(),
    benchmark: 'island-nn-archive',
    task: {
      formula: '0.7*sin(x0) + 1.3*x1*x1 - 0.4*x0*x1',
      trainCases: train.length,
      validationCases: validation.length,
      guardCases: guard.length,
      testCases: test.length
    },
    budget: {
      seeds,
      islandCount,
      islandPopulation,
      singlePopulation,
      generations,
      successLift,
      nnAcceptThreshold,
      validationVetoRatio,
      singleSolvedMae,
      singleBaselineVetoRatio,
      singleBaselineGuardVetoRatio,
      nonRegressionTolerance,
      seedTimeoutMs,
      roles: roles.slice(0, islandCount).map(role => role.name),
      note: 'Release-candidate benchmark. Uses real MigrationHub islands, semantic archive telemetry, the existing MLP as a candidate gate, robust validation/guard holdout selection, and a single-baseline fallback for guard regressions.'
    },
    summary,
    results
  };

  const pass = !(
    summary.completedSeedCount !== results.length ||
    summary.timedOutCount > 0 ||
    summary.runtimeErrorCount > 0 ||
    summary.nonDegenerateSuccessRate < 2 / 3 ||
    summary.nonRegressionRate < 0.95 ||
    summary.medianEligibleLiftVsSingle < successLift ||
    summary.totalArchiveInjections === 0
  );
  fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(markdownOutputPath, renderMarkdown(report));
  writeBenchmarkManifest(repoRoot, benchmarkOutput, {
    lane: 'island-nn-archive',
    status: pass ? 'pass' : 'fail',
    summary
  });

  console.log('Multi-island + NN archive benchmark');
  for (const result of results) {
    console.log([
      `seed=${result.seed}`,
      `singleMAE=${formatNumber(result.single.metrics.mae)}`,
      `singleValidationMAE=${formatNumber(result.single.validationMetrics.mae)}`,
      `nnMAE=${formatNumber(result.nnSelected.metrics.mae)}`,
      `lift=${formatNumber(result.liftVsSingle)}x`,
      `nnIsland=${result.nnSelected.islandId}`,
      `nnP=${formatNumber(result.nnSelected.probability)}`,
      `archiveInject=${result.migration.archiveInjections}`,
      `nnAcc=${formatNumber(result.nn.trainingAccuracy)}`,
      `fallback=${result.nnSelected.fallbackUsed}`,
      `outcome=${result.outcome}`,
      `success=${result.success}`
    ].join(' | '));
  }
  console.log([
    `meanLift=${formatNumber(summary.meanLiftVsSingle)}x`,
    `medianLift=${formatNumber(summary.medianLiftVsSingle)}x`,
    `medianEligibleLift=${formatNumber(summary.medianEligibleLiftVsSingle)}x`,
    `successRate=${Math.round(summary.successRate * 100)}%`,
    `nonDegenerateSuccessRate=${Math.round(summary.nonDegenerateSuccessRate * 100)}%`,
    `nonRegressionRate=${Math.round(summary.nonRegressionRate * 100)}%`,
    `completed=${summary.completedSeedCount}/${results.length}`,
    `timeouts=${summary.timedOutCount}`,
    `errors=${summary.runtimeErrorCount}`,
    `singleSolved=${summary.singleSolvedCount}`,
    `singleFallbacks=${summary.singleFallbackCount}`
  ].join(' '));
  console.log(`\nWrote ${path.relative(repoRoot, jsonOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, markdownOutputPath)}`);
  console.log(`Wrote ${path.relative(repoRoot, benchmarkOutput.manifestOutputPath)}`);

  if (!pass) process.exitCode = 1;
}

async function runSeed(seed: number, train: Point[], validation: Point[], guard: Point[], test: Point[]): Promise<SeedResult> {
  return await withDeterministicMathRandom(seed, async () => {
    const single = await runSingle(seed, train, validation, guard, test);
    const previousBasisSeed = process.env.GCL_REGRESSION_BASIS_SEED;
    try {
      process.env.GCL_REGRESSION_BASIS_SEED = '0';

      const hub = new MigrationHub();
      const engines = Array.from({ length: islandCount }, (_, islandId) => {
        const role = roles[islandId % roles.length];
        const engine = new EvolutionEngine(createProblem(train, role.functions, `island-${role.name}`), {
          population: {
            size: islandPopulation,
            eliteCount: Math.max(3, Math.floor(islandPopulation * 0.06)),
            mutationRate: role.name === 'trig' ? 0.36 : 0.30,
            crossoverRate: 0.78,
            maxGenomeDepth: role.name === 'arithmetic' ? 3 : 4,
            maxGenomeSize: 64,
            tournamentSize: 5,
            parsimonyCoefficient: role.name === 'full' ? 0.00012 : 0.00018
          },
          targetFitness: 0.999999,
          maxGenerations: generations,
          stagnationLimit: generations,
          parallelEvaluators: 1,
          selectionMethod: 'tournament',
          seed: seed + islandId * 101,
          optimizer: {
            enabled: true,
            iterations: 10,
            learningRate: 0.04,
            interval: 5,
            targetCount: 4
          }
        });
        engine.preferredSource = (islandId + islandCount - 1) % islandCount;
        return { engine, role };
      });

      const history: CandidateRecord[] = [];
      for (let gen = 0; gen < generations; gen++) {
        for (let islandId = 0; islandId < engines.length; islandId++) {
          await engines[islandId].engine.step(gen, islandId, hub);
        }

        if (gen % 5 === 0 || gen === generations - 1) {
          for (let islandId = 0; islandId < engines.length; islandId++) {
            const genome = engines[islandId].engine.getBestGenome();
            if (!genome) continue;
            const archive = engines[islandId].engine.getSemanticArchiveDebugStats();
            history.push({
              islandId,
              generation: gen,
              trainMae: evaluateGenome(genome, train).mae,
              validationMae: evaluateGenome(genome, validation).mae,
              guardMae: evaluateGenome(genome, guard).mae,
              complexity: genome.genes.length,
              fitness: genome.fitness,
              archiveInjections: archive.injectionSuccesses,
              archiveSize: archive.size,
              genome
            });
          }
        }
      }

      const finalRecords = engines.map(({ engine }, islandId) => {
        const genome = engine.getBestGenome();
        if (!genome) throw new Error(`Island ${islandId} produced no genome`);
        const archive = engine.getSemanticArchiveDebugStats();
        return {
          islandId,
          generation: generations,
          trainMae: evaluateGenome(genome, train).mae,
          validationMae: evaluateGenome(genome, validation).mae,
          guardMae: evaluateGenome(genome, guard).mae,
          complexity: genome.genes.length,
          fitness: genome.fitness,
          archiveInjections: archive.injectionSuccesses,
          archiveSize: archive.size,
          genome
        };
      });

      const nn = trainGate(history, generations);
      const scored = finalRecords.map(record => ({
        record,
        probability: nn.model.predict([features(record, generations)])[0][0]
      })).sort((a, b) => b.probability - a.probability || robustHoldoutMae(a.record) - robustHoldoutMae(b.record));

      const validationBest = [...finalRecords].sort((a, b) => robustHoldoutMae(a) - robustHoldoutMae(b) || a.validationMae - b.validationMae)[0];
      const nnTop = scored[0];
      const policy = applyPolicyGuards(nnTop.record, nnTop.probability, validationBest, single, test);
      const validationMetrics = evaluateGenome(validationBest.genome, test);
      const islandResults: IslandResult[] = engines.map(({ engine, role }, islandId) => {
        const genome = finalRecords[islandId].genome;
        const archive = engine.getSemanticArchiveDebugStats();
        return {
          islandId,
          role: role.name,
          metrics: evaluateGenome(genome, test),
          validationMae: finalRecords[islandId].validationMae,
          guardMae: finalRecords[islandId].guardMae,
          archive: {
            recordCycles: archive.recordCycles,
            injectionAttempts: archive.injectionAttempts,
            injectionSuccesses: archive.injectionSuccesses,
            size: archive.size
          }
        };
      });
      const migrationStats = hub.getStats();
      const archiveInjections = islandResults.reduce((sum, island) => sum + island.archive.injectionSuccesses, 0);
      const archiveInjectionAttempts = islandResults.reduce((sum, island) => sum + island.archive.injectionAttempts, 0);
      const liftVsSingle = single.metrics.mae / Math.max(policy.metrics.mae, 1e-12);
      const singleSolved = single.metrics.mae <= singleSolvedMae;
      const islandLift = archiveInjections > 0
        && nn.trainingAccuracy >= 0.6
        && policy.metrics.mae < single.metrics.mae
        && liftVsSingle >= successLift
        && policy.islandId >= 0;

      return {
        seed,
        single,
        islands: islandResults,
        validationSelected: {
          islandId: validationBest.islandId,
          metrics: validationMetrics,
          validationMae: validationBest.validationMae,
          guardMae: validationBest.guardMae
        },
        nnSelected: policy,
        nn: {
          samples: nn.samples,
          positives: nn.positives,
          trainingAccuracy: nn.trainingAccuracy
        },
        migration: {
          drops: migrationStats.drops,
          pullAttempts: migrationStats.pullAttempts,
          pullSuccesses: migrationStats.pullSuccesses,
          archiveInjectionAttempts,
          archiveInjections
        },
        liftVsSingle,
        singleSolved,
        outcome: singleSolved
          ? 'single-baseline-solved'
          : policy.fallbackReason.startsWith('single-baseline-')
            ? 'single-baseline-fallback'
            : islandLift
              ? 'island-nn-lift'
              : 'island-nn-miss',
        success: singleSolved || islandLift,
        completed: true
      };
    } finally {
      if (previousBasisSeed === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
      else process.env.GCL_REGRESSION_BASIS_SEED = previousBasisSeed;
    }
  });
}

async function runSingle(seed: number, train: Point[], validation: Point[], guard: Point[], test: Point[]): Promise<{ validationMetrics: MetricSet; guardMetrics: MetricSet; metrics: MetricSet; archiveInjections: number }> {
  const previousBasisSeed = process.env.GCL_REGRESSION_BASIS_SEED;
  try {
    process.env.GCL_REGRESSION_BASIS_SEED = '0';
    const engine = new EvolutionEngine(createProblem(train, roles[0].functions, 'single'), {
      population: {
        size: singlePopulation,
        eliteCount: Math.max(3, Math.floor(singlePopulation * 0.05)),
        mutationRate: 0.32,
        crossoverRate: 0.78,
        maxGenomeDepth: 4,
        maxGenomeSize: 64,
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
    const archive = engine.getSemanticArchiveDebugStats();
    return {
      validationMetrics: evaluateGenome(result.bestGenome, validation),
      guardMetrics: evaluateGenome(result.bestGenome, guard),
      metrics: evaluateGenome(result.bestGenome, test),
      archiveInjections: archive.injectionSuccesses
    };
  } finally {
    if (previousBasisSeed === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
    else process.env.GCL_REGRESSION_BASIS_SEED = previousBasisSeed;
  }
}

function applyPolicyGuards(
  nnBest: CandidateRecord,
  probability: number,
  validationBest: CandidateRecord,
  single: SeedResult['single'],
  test: Point[]
): SeedResult['nnSelected'] {
  let selected = nnBest;
  let selectedProbability = probability;
  let fallbackUsed = false;
  let fallbackReason = 'accepted-nn';

  if (probability < nnAcceptThreshold) {
    selected = validationBest;
    selectedProbability = probability;
    fallbackUsed = true;
    fallbackReason = 'low-nn-confidence';
  } else if (robustHoldoutMae(nnBest) > robustHoldoutMae(validationBest) * validationVetoRatio) {
    selected = validationBest;
    selectedProbability = probability;
    fallbackUsed = true;
    fallbackReason = 'validation-veto';
  }

  const selectedMetrics = evaluateGenome(selected.genome, test);
  const singleWinsPrimaryHoldout = single.validationMetrics.mae <= selected.validationMae * singleBaselineVetoRatio;
  const singleWinsGuardHoldout = single.guardMetrics.mae <= selected.guardMae * singleBaselineVetoRatio;
  const selectedFailsGuardAgainstSingle = selected.guardMae > single.guardMetrics.mae * singleBaselineGuardVetoRatio;
  if ((singleWinsPrimaryHoldout && singleWinsGuardHoldout) || selectedFailsGuardAgainstSingle) {
    return {
      islandId: -2,
      probability: selectedProbability,
      metrics: single.metrics,
      validationMae: single.validationMetrics.mae,
      guardMae: single.guardMetrics.mae,
      fallbackUsed: true,
      fallbackReason: selectedFailsGuardAgainstSingle ? 'single-baseline-guard-veto' : 'single-baseline-veto'
    };
  }

  return {
    islandId: selected.islandId,
    probability: selectedProbability,
    metrics: selectedMetrics,
    validationMae: selected.validationMae,
    guardMae: selected.guardMae,
    fallbackUsed,
    fallbackReason
  };
}

function robustHoldoutMae(record: CandidateRecord): number {
  return Math.max(record.validationMae, record.guardMae);
}

function makeFailedSeedResult(
  seed: number,
  outcome: 'runtime-timeout' | 'runtime-error',
  error: string
): SeedResult {
  const metrics = zeroMetrics();
  return {
    seed,
    single: {
      validationMetrics: metrics,
      guardMetrics: metrics,
      metrics,
      archiveInjections: 0
    },
    islands: [],
    validationSelected: {
      islandId: -1,
      metrics,
      validationMae: 0,
      guardMae: 0
    },
    nnSelected: {
      islandId: -1,
      probability: 0,
      metrics,
      validationMae: 0,
      guardMae: 0,
      fallbackUsed: true,
      fallbackReason: outcome
    },
    nn: {
      samples: 0,
      positives: 0,
      trainingAccuracy: 0
    },
    migration: {
      drops: 0,
      pullAttempts: 0,
      pullSuccesses: 0,
      archiveInjectionAttempts: 0,
      archiveInjections: 0
    },
    liftVsSingle: 0,
    singleSolved: false,
    outcome,
    success: false,
    completed: false,
    error
  };
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`seed timed out after ${timeoutMs}ms`)), timeoutMs);
      })
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function trainGate(records: CandidateRecord[], maxGeneration: number): {
  model: MLP;
  samples: number;
  positives: number;
  trainingAccuracy: number;
} {
  if (records.length < 8) throw new Error('Not enough candidate records to train NN gate');
  const validationScores = records.map(record => record.validationMae).sort((a, b) => a - b);
  const cutoff = validationScores[Math.max(0, Math.floor(validationScores.length * 0.35) - 1)];
  const X = records.map(record => features(record, maxGeneration));
  const Y: number[] = records.map(record => record.validationMae <= cutoff ? 1 : 0);
  const positives = Y.reduce((sum, value) => sum + value, 0);
  const model = new MLP([X[0].length, 8, 1], ['tanh', 'sigmoid']);
  model.train(X, Y, 240, 0.04, Math.min(16, X.length), false);
  return {
    model,
    samples: records.length,
    positives,
    trainingAccuracy: model.accuracy(X, Y)
  };
}

function features(record: CandidateRecord, maxGeneration: number): number[] {
  return [
    1 / (1 + record.trainMae),
    Math.min(1, record.complexity / 80),
    Math.max(0, Math.min(1, record.fitness)),
    record.islandId / Math.max(1, islandCount - 1),
    record.generation / Math.max(1, maxGeneration),
    Math.min(1, record.archiveInjections / 8),
    Math.min(1, record.archiveSize / 256)
  ];
}

function createProblem(points: Point[], functionNames: string[], name: string): ProblemDefinition<number[], number> {
  return {
    name,
    description: 'Island NN archive benchmark task',
    functionSet: MATH_FUNCTIONS.filter(fn => functionNames.includes(fn.name)),
    terminalSet: createTerminals(),
    testCases: points.map(point => ({ input: point.input, expectedOutput: point.expected, weight: 1 })),
    fitnessFunction: mathFitnessFunction
  };
}

function createTerminals(): TerminalDefinition[] {
  return [
    { name: 'x', type: 'variable', description: 'Variable x0' },
    { name: 'x1', type: 'variable', description: 'Variable x1' },
    ...[-2, -1, -0.5, -0.4, -0.25, 0, 0.25, 0.4, 0.5, 0.7, 1, 1.3, 2].map(value => ({
      name: 'const',
      type: 'constant' as const,
      value,
      description: `Fixed island benchmark constant ${value}`
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

function evaluateGenome(genome: Genome, points: Point[]): MetricSet {
  return computeMetrics(
    points.map(point => point.expected),
    points.map(point => safeNumber(executeMathGenome(genome, { x: point.input[0], x1: point.input[1] })))
  );
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

function zeroMetrics(): MetricSet {
  return {
    mae: 0,
    rmse: 0,
    r2: 0,
    maxAbsError: 0
  };
}

function renderMarkdown(report: {
  generatedAt: string;
  task: { formula: string; trainCases: number; validationCases: number; guardCases: number; testCases: number };
  budget: Record<string, unknown>;
  summary: Record<string, number>;
  results: SeedResult[];
}): string {
  const lines: string[] = [];
  lines.push('# Multi-island + NN archive benchmark');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push('- Runs four real `EvolutionEngine` islands against a shared `MigrationHub`.');
  lines.push('- Uses different island grammar roles: full, arithmetic, trig, and rational-trig.');
  lines.push('- Uses the existing `MLP` to learn from validation labels over island champion history; validation error itself is not an input feature.');
  lines.push('- Applies robust validation/guard-holdout selection and falls back to single baseline on guard regressions.');
  lines.push('- The wide seed sweep is eligible for `release:gate` only when timeout/error, non-regression, and non-degenerate lift gates pass.');
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Mean single-island MAE: ${formatNumber(report.summary.meanSingleMae)}`);
  lines.push(`- Mean NN-selected island MAE: ${formatNumber(report.summary.meanNnMae)}`);
  lines.push(`- Mean robust holdout-only MAE: ${formatNumber(report.summary.meanValidationSelectedMae)}`);
  lines.push(`- Mean lift vs single: ${formatNumber(report.summary.meanLiftVsSingle)}x`);
  lines.push(`- Median lift vs single: ${formatNumber(report.summary.medianLiftVsSingle)}x`);
  lines.push(`- Median eligible lift vs single: ${formatNumber(report.summary.medianEligibleLiftVsSingle)}x`);
  lines.push(`- Success rate: ${Math.round(report.summary.successRate * 100)}%`);
  lines.push(`- Non-degenerate success rate: ${Math.round(report.summary.nonDegenerateSuccessRate * 100)}%`);
  lines.push(`- Non-regression rate vs single baseline: ${Math.round(report.summary.nonRegressionRate * 100)}%`);
  lines.push(`- Completed seeds: ${report.summary.completedSeedCount}/${report.results.length}`);
  lines.push(`- Runtime timeouts: ${report.summary.timedOutCount}`);
  lines.push(`- Runtime errors: ${report.summary.runtimeErrorCount}`);
  lines.push(`- Single-baseline solved seeds: ${report.summary.singleSolvedCount}`);
  lines.push(`- Single-baseline fallbacks: ${report.summary.singleFallbackCount}`);
  lines.push(`- Archive injections: ${report.summary.totalArchiveInjections}`);
  lines.push(`- Hub pull successes: ${report.summary.totalHubPullSuccesses}`);
  lines.push(`- Fallbacks: ${report.summary.fallbackCount}`);
  lines.push(`- Mean NN training accuracy: ${formatNumber(report.summary.meanNnTrainingAccuracy)}`);
  lines.push('');
  lines.push('| Seed | Single MAE | Single val MAE | Single guard MAE | NN MAE | NN guard MAE | Holdout-only MAE | Lift | NN island | NN p | Archive injections | NN acc | Fallback | Outcome | Success |');
  lines.push('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|');
  for (const result of report.results) {
    lines.push(`| ${result.seed} | ${formatNumber(result.single.metrics.mae)} | ${formatNumber(result.single.validationMetrics.mae)} | ${formatNumber(result.single.guardMetrics.mae)} | ${formatNumber(result.nnSelected.metrics.mae)} | ${formatNumber(result.nnSelected.guardMae)} | ${formatNumber(result.validationSelected.metrics.mae)} | ${formatNumber(result.liftVsSingle)}x | ${result.nnSelected.islandId} | ${formatNumber(result.nnSelected.probability)} | ${result.migration.archiveInjections} | ${formatNumber(result.nn.trainingAccuracy)} | ${result.nnSelected.fallbackUsed ? result.nnSelected.fallbackReason : 'no'} | ${result.outcome} | ${result.success ? 'yes' : 'no'} |`);
  }
  lines.push('');
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

function parseSeeds(value: string): number[] {
  const parsed = value.split(',').map(item => Number(item.trim())).filter(Number.isInteger);
  if (parsed.length === 0) throw new Error('GCL_ISLAND_NN_SEEDS must contain at least one integer');
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

function median(values: number[]): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
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

main().catch(error => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});
