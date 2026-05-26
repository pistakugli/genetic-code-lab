import * as fs from 'fs';
import * as path from 'path';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { DistributedManager, IslandLaunchSpec } from '../src/core/distributed-manager';
import { EvolutionEngine } from '../src/core/evolution';
import { MLP } from '../src/core/neural-network';
import { EvolutionConfig, Genome, ProblemDefinition, TerminalDefinition } from '../src/core/types';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { MATH_FUNCTIONS, executeMathGenome, mathFitnessFunction } from '../src/solvers/math-expression';
import { resolveBenchmarkOutput, writeBenchmarkManifest } from './benchmark-paths';

type Point = { input: number[]; expected: number };
type MetricSet = { mae: number; rmse: number; r2: number; maxAbsError: number };
type HistoryRecord = {
  islandId: number;
  generation: number;
  bestFitness: number;
  bestSize: number;
  entropy: number;
  archiveSize: number;
  archiveInjections: number;
  hubPullSuccesses: number;
  validationMae: number;
  genome: Genome;
};
type FinalCandidate = HistoryRecord & { guardMetrics: MetricSet; testMetrics: MetricSet };
type GateModel = {
  models: MLP[];
  samples: number;
  positives: number;
  trainingAccuracy: number;
  holdoutSamples: number;
  holdoutAccuracy: number;
  holdoutBrier: number;
  holdoutAuc: number;
};
type ScoredCandidate = {
  candidate: FinalCandidate;
  probability: number;
  hybridScore: number;
};
type SeedResult = {
  seed: number;
  single: { validationMetrics: MetricSet; guardMetrics: MetricSet; metrics: MetricSet };
  nn: {
    samples: number;
    positives: number;
    trainingAccuracy: number;
    holdoutSamples: number;
    holdoutAccuracy: number;
    holdoutBrier: number;
    holdoutAuc: number;
  };
  policy: {
    islandId: number;
    probability: number;
    nnTopIslandId: number;
    nnTopProbability: number;
    validationBestIslandId: number;
    validationBestMae: number;
    hybridScore: number;
    fallbackUsed: boolean;
    fallbackReason: string;
    validationMae: number;
    guardMae: number;
    regretVsOracle: number;
    metrics: MetricSet;
  };
  validationOnly: { islandId: number; validationMae: number; guardMae: number; metrics: MetricSet };
  oracle: { islandId: number; validationMae: number; guardMae: number; metrics: MetricSet };
  hub: { drops: number; pullAttempts: number; pullSuccesses: number };
  candidates: Array<{
    islandId: number;
    validationMae: number;
    guardMae: number;
    testMae: number;
    archiveInjections: number;
    archiveSize: number;
  }>;
  liftVsSingle: number;
  singleSolved: boolean;
  outcome: 'single-baseline-solved' | 'single-baseline-fallback' | 'distributed-lift' | 'distributed-miss' | 'runtime-timeout' | 'runtime-error';
  success: boolean;
  completed: boolean;
  error?: string;
};

const repoRoot = path.resolve(__dirname, '..');
const seeds = parseSeeds(process.env.GCL_DISTRIBUTED_NN_SEEDS ?? '4,7,1337,2026,4242');
const benchmarkOutput = resolveBenchmarkOutput(repoRoot, 'distributed-island-nn', seeds.length >= 24 ? 'wide' : 'smoke');
const benchmarkDir = benchmarkOutput.outputDir;
const jsonOutputPath = benchmarkOutput.jsonOutputPath;
const markdownOutputPath = benchmarkOutput.markdownOutputPath;
const islandCount = parsePositiveInt(process.env.GCL_DISTRIBUTED_NN_COUNT, 4);
const islandPopulation = parsePositiveInt(process.env.GCL_DISTRIBUTED_NN_POPULATION, 180);
const singlePopulation = parsePositiveInt(process.env.GCL_DISTRIBUTED_NN_SINGLE_POPULATION, 180);
const generations = parsePositiveInt(process.env.GCL_DISTRIBUTED_NN_GENERATIONS, 100);
const successLift = Number(process.env.GCL_DISTRIBUTED_NN_MIN_LIFT ?? '1.02');
const nnAcceptThreshold = Number(process.env.GCL_DISTRIBUTED_NN_ACCEPT_P ?? '0.58');
const validationVetoRatio = Number(process.env.GCL_DISTRIBUTED_NN_VALIDATION_VETO_RATIO ?? '1.05');
const validationWeight = Number(process.env.GCL_DISTRIBUTED_NN_VALIDATION_WEIGHT ?? '0.75');
const committeeSize = parsePositiveInt(process.env.GCL_DISTRIBUTED_NN_COMMITTEE, 3);
const singleSolvedMae = Number(process.env.GCL_DISTRIBUTED_NN_SINGLE_SOLVED_MAE ?? '0.001');
const seedTimeoutMs = parsePositiveInt(process.env.GCL_DISTRIBUTED_NN_SEED_TIMEOUT_MS, 120000);
const singleBaselineVetoRatio = Number(process.env.GCL_DISTRIBUTED_NN_SINGLE_VETO_RATIO ?? '1.05');
const singleBaselineGuardVetoRatio = Number(process.env.GCL_DISTRIBUTED_NN_SINGLE_GUARD_VETO_RATIO ?? '1.20');

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
      results.push(await runSeed(seed, train, validation, guard, test));
    } catch (error) {
      results.push(makeFailedSeedResult(seed, 'runtime-error', error instanceof Error ? error.message : String(error)));
    }
  }
  const completedResults = results.filter(result => result.completed);
  const eligibleResults = completedResults.filter(result => !result.singleSolved);

  const summary = {
    successRate: results.filter(result => result.success).length / Math.max(1, results.length),
    nonDegenerateSuccessRate: eligibleResults.filter(result => result.success).length / Math.max(1, eligibleResults.length),
    nonRegressionRate: eligibleResults.filter(result => result.policy.metrics.mae <= result.single.metrics.mae * successLift).length / Math.max(1, eligibleResults.length),
    completedSeedCount: completedResults.length,
    timedOutCount: results.filter(result => result.outcome === 'runtime-timeout').length,
    runtimeErrorCount: results.filter(result => result.outcome === 'runtime-error').length,
    singleSolvedCount: completedResults.length - eligibleResults.length,
    singleFallbackCount: completedResults.filter(result => result.outcome === 'single-baseline-fallback').length,
    meanSingleMae: mean(completedResults.map(result => result.single.metrics.mae)),
    meanPolicyMae: mean(completedResults.map(result => result.policy.metrics.mae)),
    meanValidationOnlyMae: mean(completedResults.map(result => result.validationOnly.metrics.mae)),
    meanOracleMae: mean(completedResults.map(result => result.oracle.metrics.mae)),
    medianLift: median(completedResults.map(result => result.liftVsSingle)),
    medianEligibleLift: median(eligibleResults.map(result => result.liftVsSingle)),
    meanLift: mean(completedResults.map(result => result.liftVsSingle)),
    meanEligibleLift: mean(eligibleResults.map(result => result.liftVsSingle)),
    meanRegretVsOracle: mean(completedResults.map(result => result.policy.regretVsOracle)),
    policyBeatsValidationOnlyRate: completedResults.filter(result => result.policy.metrics.mae < result.validationOnly.metrics.mae).length / Math.max(1, completedResults.length),
    totalHubDrops: results.reduce((sum, result) => sum + result.hub.drops, 0),
    totalHubPullSuccesses: results.reduce((sum, result) => sum + result.hub.pullSuccesses, 0),
    fallbackCount: results.filter(result => result.policy.fallbackUsed).length,
    meanHoldoutAccuracy: mean(completedResults.map(result => result.nn.holdoutAccuracy)),
    meanHoldoutBrier: mean(completedResults.map(result => result.nn.holdoutBrier)),
    meanHoldoutAuc: mean(completedResults.map(result => result.nn.holdoutAuc))
  };

  const report = {
    generatedAt: new Date().toISOString(),
    benchmark: 'distributed-island-nn',
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
      nnAcceptThreshold,
      validationVetoRatio,
      validationWeight,
      committeeSize,
      singleSolvedMae,
      singleBaselineVetoRatio,
      singleBaselineGuardVetoRatio,
      seedTimeoutMs,
      successLift,
      roles: roles.slice(0, islandCount).map(role => role.name),
      note: 'Production runtime proof: real DistributedManager worker threads, shared MigrationHub, existing MLP scorer, validation veto/fallback policy, and a second holdout guard for the single-baseline veto.'
    },
    summary,
    results
  };

  const pass = !(
    summary.nonDegenerateSuccessRate < 2 / 3 ||
    summary.nonRegressionRate < 0.95 ||
    summary.medianEligibleLift < successLift ||
    summary.totalHubPullSuccesses === 0 ||
    summary.timedOutCount > 0 ||
    summary.runtimeErrorCount > 0
  );
  fs.writeFileSync(jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(markdownOutputPath, renderMarkdown(report));
  writeBenchmarkManifest(repoRoot, benchmarkOutput, {
    lane: 'distributed-island-nn',
    status: pass ? 'pass' : 'fail',
    summary
  });

  console.log('Distributed worker-island + NN policy benchmark');
  for (const result of results) {
    console.log([
      `seed=${result.seed}`,
      `singleMAE=${formatNumber(result.single.metrics.mae)}`,
      `singleValidationMAE=${formatNumber(result.single.validationMetrics.mae)}`,
      `policyMAE=${formatNumber(result.policy.metrics.mae)}`,
      `lift=${formatNumber(result.liftVsSingle)}x`,
      `selectedIsland=${result.policy.islandId}`,
      `nnP=${formatNumber(result.policy.probability)}`,
      `validationOnlyMAE=${formatNumber(result.validationOnly.metrics.mae)}`,
      `regret=${formatNumber(result.policy.regretVsOracle)}`,
      `fallback=${result.policy.fallbackUsed}`,
      `hubPulls=${result.hub.pullSuccesses}/${result.hub.pullAttempts}`,
      `outcome=${result.outcome}`,
      `success=${result.success}`,
      result.error ? `error=${result.error}` : ''
    ].filter(Boolean).join(' | '));
  }
  console.log([
    `medianLift=${formatNumber(summary.medianLift)}x`,
    `medianEligibleLift=${formatNumber(summary.medianEligibleLift)}x`,
    `successRate=${Math.round(summary.successRate * 100)}%`,
    `nonDegenerateSuccessRate=${Math.round(summary.nonDegenerateSuccessRate * 100)}%`,
    `nonRegressionRate=${Math.round(summary.nonRegressionRate * 100)}%`,
    `completed=${summary.completedSeedCount}/${results.length}`,
    `timeouts=${summary.timedOutCount}`,
    `errors=${summary.runtimeErrorCount}`,
    `singleSolved=${summary.singleSolvedCount}`,
    `singleFallbacks=${summary.singleFallbackCount}`,
    `fallbacks=${summary.fallbackCount}`
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
    let manager: DistributedManager | null = null;
    try {
      process.env.GCL_REGRESSION_BASIS_SEED = '0';
      const bridge = new AsyncAgentBridge(new DiscoveryAgent());
      manager = new DistributedManager(islandCount, baseConfig(seed) as EvolutionConfig, bridge);
      const specs: IslandLaunchSpec[] = Array.from({ length: islandCount }, (_, islandId) => {
        const role = roles[islandId % roles.length];
        return {
          preferredSource: (islandId + islandCount - 1) % islandCount,
          problem: createProblem(train, role.functions, `distributed-${role.name}`),
          config: roleConfig(role.name, seed + islandId * 101)
        };
      });

      await withTimeout(
        manager.start(createProblem(train, roles[0].functions, 'distributed-default'), seed, specs),
        seedTimeoutMs,
        async () => manager?.shutdown()
      );
      const history = toHistory(manager.getHistory(), validation).filter(record => record.genome.genes.length > 0);
      const finals = finalCandidates(manager.getFinalResults(), history, validation, guard, test);
      const gate = trainGate(history, generations);
      const policy = selectWithPolicy(finals, gate, generations);
      const validationOnly = selectValidationOnly(finals);
      const oracle = selectOracle(finals);
      const hub = manager.getMigrationStats();
      await manager.shutdown();

      const policyWithBaselineGuard = applySingleBaselineGuard(policy, single);
      const liftVsSingle = single.metrics.mae / Math.max(policyWithBaselineGuard.metrics.mae, 1e-12);
      const singleSolved = single.metrics.mae <= singleSolvedMae;
      const distributedLift = hub.pullSuccesses > 0 && policyWithBaselineGuard.metrics.mae < single.metrics.mae && liftVsSingle >= successLift;
      return {
        seed,
        single,
        nn: {
          samples: gate.samples,
          positives: gate.positives,
          trainingAccuracy: gate.trainingAccuracy,
          holdoutSamples: gate.holdoutSamples,
          holdoutAccuracy: gate.holdoutAccuracy,
          holdoutBrier: gate.holdoutBrier,
          holdoutAuc: gate.holdoutAuc
        },
        policy: policyWithBaselineGuard,
        validationOnly,
        oracle,
        hub,
        candidates: finals.map(candidate => ({
          islandId: candidate.islandId,
          validationMae: candidate.validationMae,
          guardMae: candidate.guardMetrics.mae,
          testMae: candidate.testMetrics.mae,
          archiveInjections: candidate.archiveInjections,
          archiveSize: candidate.archiveSize
        })),
        liftVsSingle,
        singleSolved,
        outcome: singleSolved
          ? 'single-baseline-solved'
          : policyWithBaselineGuard.fallbackReason.startsWith('single-baseline-')
            ? 'single-baseline-fallback'
          : distributedLift
            ? 'distributed-lift'
            : 'distributed-miss',
        success: distributedLift,
        completed: true
      };
    } catch (error) {
      if (error instanceof Error && error.message.includes('timed out')) {
        return makeFailedSeedResult(seed, 'runtime-timeout', error.message, single);
      }
      return makeFailedSeedResult(seed, 'runtime-error', error instanceof Error ? error.message : String(error), single);
    } finally {
      if (previousBasisSeed === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
      else process.env.GCL_REGRESSION_BASIS_SEED = previousBasisSeed;
      await manager?.shutdown().catch(() => undefined);
    }
  });
}

function makeFailedSeedResult(
  seed: number,
  outcome: 'runtime-timeout' | 'runtime-error',
  error: string,
  single: { validationMetrics: MetricSet; guardMetrics: MetricSet; metrics: MetricSet } = { validationMetrics: zeroMetrics(), guardMetrics: zeroMetrics(), metrics: zeroMetrics() }
): SeedResult {
  const metrics = zeroMetrics();
  return {
    seed,
    single,
    nn: {
      samples: 0,
      positives: 0,
      trainingAccuracy: 0,
      holdoutSamples: 0,
      holdoutAccuracy: 0,
      holdoutBrier: 0,
      holdoutAuc: 0
    },
    policy: {
      islandId: -1,
      probability: 0,
      nnTopIslandId: -1,
      nnTopProbability: 0,
      validationBestIslandId: -1,
      validationBestMae: 0,
      hybridScore: 0,
      fallbackUsed: true,
      fallbackReason: outcome,
      validationMae: 0,
      guardMae: 0,
      regretVsOracle: 0,
      metrics
    },
    validationOnly: { islandId: -1, validationMae: 0, guardMae: 0, metrics },
    oracle: { islandId: -1, validationMae: 0, guardMae: 0, metrics },
    hub: { drops: 0, pullAttempts: 0, pullSuccesses: 0 },
    candidates: [],
    liftVsSingle: 0,
    singleSolved: false,
    outcome,
    success: false,
    completed: false,
    error
  };
}

function applySingleBaselineGuard(
  policy: SeedResult['policy'],
  single: { validationMetrics: MetricSet; guardMetrics: MetricSet; metrics: MetricSet }
): SeedResult['policy'] {
  const singleWinsPrimaryHoldout = single.validationMetrics.mae <= policy.validationMae * singleBaselineVetoRatio;
  const singleWinsGuardHoldout = single.guardMetrics.mae <= policy.guardMae * singleBaselineVetoRatio;
  const policyFailsGuardAgainstSingle = policy.guardMae > single.guardMetrics.mae * singleBaselineGuardVetoRatio;
  if ((singleWinsPrimaryHoldout && singleWinsGuardHoldout) || policyFailsGuardAgainstSingle) {
    return {
      ...policy,
      islandId: -2,
      fallbackUsed: true,
      fallbackReason: policyFailsGuardAgainstSingle ? 'single-baseline-guard-veto' : 'single-baseline-veto',
      validationMae: single.validationMetrics.mae,
      guardMae: single.guardMetrics.mae,
      regretVsOracle: 0,
      metrics: single.metrics
    };
  }
  return policy;
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, onTimeout: () => Promise<unknown>): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(async () => {
          await onTimeout().catch(() => undefined);
          reject(new Error(`seed timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      })
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function baseConfig(seed: number): Partial<EvolutionConfig> {
  return {
    maxGenerations: generations,
    targetFitness: 0.999999,
    stagnationLimit: generations,
    parallelEvaluators: 1,
    islandParallelEvaluators: 1,
    seed,
    selectionMethod: 'tournament',
    population: {
      size: islandPopulation,
      eliteCount: Math.max(3, Math.floor(islandPopulation * 0.06)),
      mutationRate: 0.32,
      crossoverRate: 0.78,
      maxGenomeDepth: 4,
      maxGenomeSize: 64,
      tournamentSize: 5,
      parsimonyCoefficient: 0.00015
    },
    optimizer: {
      enabled: true,
      iterations: 10,
      learningRate: 0.04,
      interval: 5,
      targetCount: 4
    }
  };
}

function roleConfig(role: string, seed: number): Partial<EvolutionConfig> {
  return {
    seed,
    population: {
      size: islandPopulation,
      eliteCount: Math.max(3, Math.floor(islandPopulation * 0.06)),
      mutationRate: role === 'trig' ? 0.36 : 0.30,
      crossoverRate: 0.78,
      maxGenomeDepth: role === 'arithmetic' ? 3 : 4,
      maxGenomeSize: 64,
      tournamentSize: 5,
      parsimonyCoefficient: role === 'full' ? 0.00012 : 0.00018
    }
  };
}

async function runSingle(seed: number, train: Point[], validation: Point[], guard: Point[], test: Point[]): Promise<{ validationMetrics: MetricSet; guardMetrics: MetricSet; metrics: MetricSet }> {
  const previousBasisSeed = process.env.GCL_REGRESSION_BASIS_SEED;
  try {
    process.env.GCL_REGRESSION_BASIS_SEED = '0';
    const engine = new EvolutionEngine(createProblem(train, roles[0].functions, 'single'), {
      ...baseConfig(seed),
      population: {
        ...baseConfig(seed).population!,
        size: singlePopulation,
        eliteCount: Math.max(3, Math.floor(singlePopulation * 0.05))
      }
    });
    const result = await engine.evolve();
    return {
      validationMetrics: evaluateGenome(result.bestGenome, validation),
      guardMetrics: evaluateGenome(result.bestGenome, guard),
      metrics: evaluateGenome(result.bestGenome, test)
    };
  } finally {
    if (previousBasisSeed === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
    else process.env.GCL_REGRESSION_BASIS_SEED = previousBasisSeed;
  }
}

function toHistory(raw: any[], validation: Point[]): HistoryRecord[] {
  return raw
    .filter(item => item.bestGenome && Array.isArray(item.bestGenome.genes))
    .map(item => {
      const genome = item.bestGenome as Genome;
      const validationMae = evaluateGenome(genome, validation).mae;
      return {
        islandId: item.islandId,
        generation: item.gen ?? 0,
        bestFitness: item.bestFitness ?? genome.fitness ?? 0,
        bestSize: item.bestSize ?? genome.genes.length,
        entropy: item.entropy ?? 0,
        archiveSize: item.semanticArchive?.size ?? 0,
        archiveInjections: item.semanticArchive?.injectionSuccesses ?? 0,
        hubPullSuccesses: item.hubStats?.pullSuccesses ?? 0,
        validationMae,
        genome
      };
    });
}

function finalCandidates(finalResults: any[], history: HistoryRecord[], validation: Point[], guard: Point[], test: Point[]): FinalCandidate[] {
  const byIsland = new Map<number, HistoryRecord>();
  for (const record of history) byIsland.set(record.islandId, record);

  return finalResults
    .map(result => {
      const fromHistory = byIsland.get(result.islandId);
      const genome = result.bestGenomeTree || fromHistory?.genome;
      if (!genome) return null;
      const validationMae = evaluateGenome(genome, validation).mae;
      const semanticArchive = result.semanticArchive || {};
      const record: HistoryRecord = {
        islandId: result.islandId,
        generation: generations,
        bestFitness: genome.fitness ?? result.bestGenome?.fitness ?? 0,
        bestSize: genome.genes?.length ?? fromHistory?.bestSize ?? 0,
        entropy: fromHistory?.entropy ?? 0,
        archiveSize: semanticArchive.size ?? fromHistory?.archiveSize ?? 0,
        archiveInjections: semanticArchive.injectionSuccesses ?? fromHistory?.archiveInjections ?? 0,
        hubPullSuccesses: result.hubStats?.pullSuccesses ?? fromHistory?.hubPullSuccesses ?? 0,
        validationMae,
        genome
      };
      return { ...record, guardMetrics: evaluateGenome(genome, guard), testMetrics: evaluateGenome(genome, test) };
    })
    .filter((item): item is FinalCandidate => item !== null);
}

function trainGate(records: HistoryRecord[], maxGeneration: number): GateModel {
  if (records.length < 8) throw new Error('Not enough worker history to train NN policy');
  const trainCutoffGeneration = Math.max(1, Math.floor(maxGeneration * 0.75));
  let trainRecords = records.filter(record => record.generation <= trainCutoffGeneration);
  let holdoutRecords = records.filter(record => record.generation > trainCutoffGeneration);
  if (trainRecords.length < 8 || holdoutRecords.length < 4) {
    trainRecords = records.slice(0, Math.max(8, Math.floor(records.length * 0.75)));
    holdoutRecords = records.slice(trainRecords.length);
  }

  const validationScores = trainRecords.map(record => record.validationMae).sort((a, b) => a - b);
  const cutoff = validationScores[Math.max(0, Math.floor(validationScores.length * 0.35) - 1)];
  const trainX = trainRecords.map(record => features(record, maxGeneration));
  const trainY: number[] = trainRecords.map(record => record.validationMae <= cutoff ? 1 : 0);
  const allX = records.map(record => features(record, maxGeneration));
  const allY: number[] = records.map(record => record.validationMae <= cutoff ? 1 : 0);
  const holdoutX = holdoutRecords.map(record => features(record, maxGeneration));
  const holdoutY: number[] = holdoutRecords.map(record => record.validationMae <= cutoff ? 1 : 0);

  const probeModels = trainCommittee(trainX, trainY);
  const holdoutPredictions = holdoutX.map(row => predictCommittee(probeModels, row));
  const models = trainCommittee(allX, allY);
  const allPredictions = allX.map(row => predictCommittee(models, row));

  return {
    models,
    samples: records.length,
    positives: allY.reduce((sum, value) => sum + value, 0),
    trainingAccuracy: binaryAccuracy(allPredictions, allY),
    holdoutSamples: holdoutRecords.length,
    holdoutAccuracy: holdoutY.length === 0 ? 0 : binaryAccuracy(holdoutPredictions, holdoutY),
    holdoutBrier: holdoutY.length === 0 ? 0 : brierScore(holdoutPredictions, holdoutY),
    holdoutAuc: holdoutY.length === 0 ? 0.5 : aucScore(holdoutPredictions, holdoutY)
  };
}

function trainCommittee(X: number[][], Y: number[]): MLP[] {
  return Array.from({ length: committeeSize }, () => {
    const model = new MLP([X[0].length, 12, 6, 1], ['tanh', 'tanh', 'sigmoid']);
    model.train(X, Y, 420, 0.03, Math.min(16, X.length), false);
    return model;
  });
}

function predictCommittee(models: MLP[], row: number[]): number {
  return mean(models.map(model => model.predict([row])[0][0]));
}

function selectWithPolicy(candidates: FinalCandidate[], gate: GateModel, maxGeneration: number): {
  islandId: number;
  probability: number;
  nnTopIslandId: number;
  nnTopProbability: number;
  validationBestIslandId: number;
  validationBestMae: number;
  hybridScore: number;
  fallbackUsed: boolean;
  fallbackReason: string;
  validationMae: number;
  guardMae: number;
  regretVsOracle: number;
  metrics: MetricSet;
} {
  if (candidates.length === 0) throw new Error('No final island candidates');
  const validationBest = [...candidates].sort((a, b) => robustHoldoutMae(a) - robustHoldoutMae(b) || a.validationMae - b.validationMae)[0];
  const oracleBest = selectOracle(candidates);
  const scored = candidates
    .map(candidate => scoreCandidate(candidate, gate, robustHoldoutMae(validationBest), maxGeneration))
    .sort((a, b) => b.probability - a.probability || robustHoldoutMae(a.candidate) - robustHoldoutMae(b.candidate));
  const nnChoice = scored[0];
  const hybridChoice = [...scored].sort((a, b) => b.hybridScore - a.hybridScore || robustHoldoutMae(a.candidate) - robustHoldoutMae(b.candidate))[0];

  let selected = hybridChoice.candidate;
  let fallbackUsed = false;
  let fallbackReason = 'accepted-hybrid';
  if (hybridChoice.probability < nnAcceptThreshold) {
    selected = validationBest;
    fallbackUsed = true;
    fallbackReason = 'low-nn-confidence';
  } else if (robustHoldoutMae(hybridChoice.candidate) > robustHoldoutMae(validationBest) * validationVetoRatio) {
    selected = validationBest;
    fallbackUsed = true;
    fallbackReason = 'validation-veto';
  }
  const selectedScore = scored.find(item => item.candidate.islandId === selected.islandId) || hybridChoice;

  return {
    islandId: selected.islandId,
    probability: selectedScore.probability,
    nnTopIslandId: nnChoice.candidate.islandId,
    nnTopProbability: nnChoice.probability,
    validationBestIslandId: validationBest.islandId,
    validationBestMae: validationBest.validationMae,
    hybridScore: selectedScore.hybridScore,
    fallbackUsed,
    fallbackReason,
    validationMae: selected.validationMae,
    guardMae: selected.guardMetrics.mae,
    regretVsOracle: selected.testMetrics.mae - oracleBest.metrics.mae,
    metrics: selected.testMetrics
  };
}

function scoreCandidate(candidate: FinalCandidate, gate: GateModel, bestValidationMae: number, maxGeneration: number): ScoredCandidate {
  const probability = predictCommittee(gate.models, features(candidate, maxGeneration));
  const validationRatio = robustHoldoutMae(candidate) / Math.max(bestValidationMae, 1e-12);
  const validationPenalty = validationWeight * Math.log1p(Math.max(0, validationRatio - 1));
  const complexityPenalty = 0.03 * Math.min(1, candidate.bestSize / 100);
  const archiveSignal = 0.04 * Math.min(1, (candidate.archiveInjections + candidate.hubPullSuccesses) / 16);
  return {
    candidate,
    probability,
    hybridScore: probability - validationPenalty - complexityPenalty + archiveSignal
  };
}

function selectValidationOnly(candidates: FinalCandidate[]): { islandId: number; validationMae: number; guardMae: number; metrics: MetricSet } {
  const selected = [...candidates].sort((a, b) => robustHoldoutMae(a) - robustHoldoutMae(b) || a.validationMae - b.validationMae)[0];
  return {
    islandId: selected.islandId,
    validationMae: selected.validationMae,
    guardMae: selected.guardMetrics.mae,
    metrics: selected.testMetrics
  };
}

function robustHoldoutMae(candidate: FinalCandidate): number {
  return Math.max(candidate.validationMae, candidate.guardMetrics.mae);
}

function selectOracle(candidates: FinalCandidate[]): { islandId: number; validationMae: number; guardMae: number; metrics: MetricSet } {
  const selected = [...candidates].sort((a, b) => a.testMetrics.mae - b.testMetrics.mae)[0];
  return {
    islandId: selected.islandId,
    validationMae: selected.validationMae,
    guardMae: selected.guardMetrics.mae,
    metrics: selected.testMetrics
  };
}

function features(record: HistoryRecord, maxGeneration: number): number[] {
  return [
    Math.max(0, Math.min(1, record.bestFitness)),
    Math.min(1, record.bestSize / 100),
    Math.min(1, Math.max(0, record.entropy)),
    record.islandId / Math.max(1, islandCount - 1),
    record.generation / Math.max(1, maxGeneration),
    Math.min(1, record.archiveInjections / 8),
    Math.min(1, record.archiveSize / 256),
    Math.min(1, record.hubPullSuccesses / 32)
  ];
}

function createProblem(points: Point[], functionNames: string[], name: string): ProblemDefinition<number[], number> {
  return {
    name,
    description: 'Distributed island NN benchmark task',
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
      description: `Fixed distributed benchmark constant ${value}`
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

function renderMarkdown(report: any): string {
  const lines: string[] = [];
  lines.push('# Distributed worker-island + NN policy benchmark');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push('- Runs real `DistributedManager` worker-thread islands.');
  lines.push('- Uses a shared `MigrationHub` with atomic migration counters.');
  lines.push('- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.');
  lines.push('- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.');
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Mean single-island MAE: ${formatNumber(report.summary.meanSingleMae)}`);
  lines.push(`- Mean policy MAE: ${formatNumber(report.summary.meanPolicyMae)}`);
  lines.push(`- Mean robust holdout-only MAE: ${formatNumber(report.summary.meanValidationOnlyMae)}`);
  lines.push(`- Mean oracle-best MAE: ${formatNumber(report.summary.meanOracleMae)}`);
  lines.push(`- Median lift: ${formatNumber(report.summary.medianLift)}x`);
  lines.push(`- Median lift on non-degenerate seeds: ${formatNumber(report.summary.medianEligibleLift)}x`);
  lines.push(`- Success rate: ${Math.round(report.summary.successRate * 100)}%`);
  lines.push(`- Non-degenerate success rate: ${Math.round(report.summary.nonDegenerateSuccessRate * 100)}%`);
  lines.push(`- Non-regression rate vs single baseline: ${Math.round(report.summary.nonRegressionRate * 100)}%`);
  lines.push(`- Completed seeds: ${report.summary.completedSeedCount}/${report.results.length}`);
  lines.push(`- Runtime timeouts: ${report.summary.timedOutCount}`);
  lines.push(`- Runtime errors: ${report.summary.runtimeErrorCount}`);
  lines.push(`- Single-baseline solved seeds: ${report.summary.singleSolvedCount}`);
  lines.push(`- Single-baseline fallbacks: ${report.summary.singleFallbackCount}`);
  lines.push(`- Mean regret vs oracle: ${formatNumber(report.summary.meanRegretVsOracle)}`);
  lines.push(`- Policy beats robust holdout-only rate: ${Math.round(report.summary.policyBeatsValidationOnlyRate * 100)}%`);
  lines.push(`- NN holdout accuracy: ${formatNumber(report.summary.meanHoldoutAccuracy)}`);
  lines.push(`- NN holdout Brier: ${formatNumber(report.summary.meanHoldoutBrier)}`);
  lines.push(`- NN holdout AUC: ${formatNumber(report.summary.meanHoldoutAuc)}`);
  lines.push(`- Hub drops: ${report.summary.totalHubDrops}`);
  lines.push(`- Hub pull successes: ${report.summary.totalHubPullSuccesses}`);
  lines.push(`- Fallbacks: ${report.summary.fallbackCount}`);
  lines.push('');
  lines.push('| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |');
  lines.push('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|');
  for (const result of report.results) {
    lines.push(`| ${result.seed} | ${formatNumber(result.single.metrics.mae)} | ${formatNumber(result.single.validationMetrics.mae)} | ${formatNumber(result.single.guardMetrics.mae)} | ${formatNumber(result.policy.metrics.mae)} | ${formatNumber(result.policy.guardMae)} | ${formatNumber(result.validationOnly.metrics.mae)} | ${formatNumber(result.oracle.metrics.mae)} | ${formatNumber(result.liftVsSingle)}x | ${result.policy.islandId} | ${formatNumber(result.policy.probability)} | ${formatNumber(result.policy.regretVsOracle)} | ${result.policy.fallbackUsed ? result.policy.fallbackReason : 'no'} | ${result.hub.pullSuccesses}/${result.hub.pullAttempts} | ${result.outcome} |`);
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

function binaryAccuracy(predictions: number[], labels: number[]): number {
  if (labels.length === 0) return 0;
  let correct = 0;
  for (let i = 0; i < labels.length; i++) {
    if ((predictions[i] >= 0.5 ? 1 : 0) === labels[i]) correct++;
  }
  return correct / labels.length;
}

function brierScore(predictions: number[], labels: number[]): number {
  if (labels.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < labels.length; i++) {
    const error = predictions[i] - labels[i];
    sum += error * error;
  }
  return sum / labels.length;
}

function aucScore(predictions: number[], labels: number[]): number {
  const positives: number[] = [];
  const negatives: number[] = [];
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) positives.push(predictions[i]);
    else negatives.push(predictions[i]);
  }
  if (positives.length === 0 || negatives.length === 0) return 0.5;
  let wins = 0;
  for (const positive of positives) {
    for (const negative of negatives) {
      if (positive > negative) wins += 1;
      else if (positive === negative) wins += 0.5;
    }
  }
  return wins / (positives.length * negatives.length);
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
  if (parsed.length === 0) throw new Error('GCL_DISTRIBUTED_NN_SEEDS must contain at least one integer');
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
