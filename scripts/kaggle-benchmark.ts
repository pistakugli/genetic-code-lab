/**
 * Kaggle Boston Housing — FULL-POWER Symbolic Regression Benchmark
 * Uses ALL engine features: Discovery Agent, Numerical Optimizer, 
 * epsilon-lexicase, high generations, proper parsimony.
 */
import { resolvePerformanceParallelism } from '../src/core/runtime-lane';
import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { ProblemDefinition, TestCase, EvolutionConfig } from '../src/core/types';
import {
  MATH_FUNCTIONS,
  createMathTerminals,
  mathFitnessFunction
} from '../src/solvers/math-expression';
import fs from 'fs';

// ─── Parse CSV ─────────────────────────────────────────────────────────────

function parseCSV(filePath: string): { headers: string[]; rows: number[][] } {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const lines = raw.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
  const headers = lines[0].split(',').map(h => h.trim());
  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => parseFloat(v.trim()));
    if (values.length === headers.length && values.every(v => Number.isFinite(v))) {
      rows.push(values);
    }
  }
  return { headers, rows };
}

// ─── Normalize data for better GP convergence ──────────────────────────────

function normalize(values: number[]): { normalized: number[]; mean: number; std: number } {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance) || 1;
  return { normalized: values.map(v => (v - mean) / std), mean, std };
}

// ─── Build problem ─────────────────────────────────────────────────────────

function buildRegressionProblem(
  rows: number[][],
  featureIdxs: number[],
  targetIdx: number
): ProblemDefinition<number[], number> {
  // Normalize features and target
  const featureStats = featureIdxs.map(fi => {
    const vals = rows.map(r => r[fi]);
    return normalize(vals);
  });
  const targetVals = rows.map(r => r[targetIdx]);
  const targetStats = normalize(targetVals);

  const testCases: TestCase<number[], number>[] = rows.map((row, i) => ({
    input: featureIdxs.map((_, fi) => featureStats[fi].normalized[i]),
    expectedOutput: targetStats.normalized[i],
    weight: 1
  }));

  const variables = featureIdxs.map((_, i) => `x${i}`);

  return {
    name: 'Boston Housing Regression',
    description: `Symbolic regression on normalized housing data`,
    functionSet: MATH_FUNCTIONS.slice(0, 8), // add, sub, mul, div, neg, abs, sqrt, pow
    terminalSet: createMathTerminals(variables),
    testCases,
    fitnessFunction: mathFitnessFunction,
    fitnessMode: 'regression'
  };
}

// ─── Main ──────────────────────────────────────────────────────────────────

async function main() {
  const csvPath = 'data/boston-housing.csv';
  const { headers, rows } = parseCSV(csvPath);
  console.log(`\n📊 Dataset: ${csvPath} — ${rows.length} rows × ${headers.length} cols`);
  console.log(`   Columns: ${headers.join(', ')}\n`);

  // Feature selection: RM (5), LSTAT (12) → MEDV (13)
  // These are the two most predictive features for housing prices
  const featureNames = ['RM', 'LSTAT'];
  const featureIdxs = featureNames.map(n => headers.indexOf(n));
  const targetIdx = headers.indexOf('MEDV');
  
  console.log(`   Features: ${featureNames.join(', ')}`);
  console.log(`   Target: MEDV (median house value)\n`);

  const problem = buildRegressionProblem(rows, featureIdxs, targetIdx);

  // ── FULL POWER config ──
  const config: Partial<EvolutionConfig> = {
    population: {
      size: 500,
      eliteCount: 15,
      maxGenomeSize: 100,     // Keep it compact
      crossoverRate: 0.85,
      mutationRate: 0.15,
      maxGenomeDepth: 6,      // Shallow = interpretable
      tournamentSize: 5,
      parsimonyCoefficient: 0.003  // Penalize bloat
    },
    maxGenerations: 300,
    parallelEvaluators: resolvePerformanceParallelism(),
    targetFitness: 0.95,
    selectionMethod: 'epsilon-lexicase',
    epsilon: 0.05,
    stagnationLimit: 100
  };

  console.log(`⚙️  Config: pop=500, gens=300, epsilon-lexicase, parsimony=0.003`);
  console.log(`   maxDepth=6, maxSize=100, crossover=0.85, mutation=0.15\n`);
  console.log(`🧬 Starting evolution...\n`);

  const agent = new DiscoveryAgent();
  const bridge = new AsyncAgentBridge(agent);
  const engine = new EvolutionEngine(problem, config);
  engine.setDiscoveryAgent(agent);
  (engine as any).agentBridge = bridge;

  let lastPrintGen = -10;
  engine.onStats = (stats: any) => {
    if (stats.gen - lastPrintGen >= 25 || stats.bestFitness >= 0.95) {
      console.log(`   Gen ${String(stats.gen).padStart(3)} | Fitness: ${stats.bestFitness.toFixed(6)} | Size: ${stats.bestSize}`);
      lastPrintGen = stats.gen;
    }
  };

  const startTime = Date.now();
  const result = await engine.evolve();
  const elapsed = Date.now() - startTime;

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`   RESULT: ${result.success ? '✅ SUCCESS' : '📊 PARTIAL'}`);
  console.log(`   Fitness:     ${result.bestGenome.fitness.toFixed(8)}`);
  console.log(`   Generations: ${result.generationsRun}`);
  console.log(`   Time:        ${(elapsed / 1000).toFixed(1)}s`);
  console.log(`${'═'.repeat(60)}\n`);

  // ── Evaluate on raw (un-normalized) scale ──
  const bestBinary = engine.getBestBinary();
  if (bestBinary) {
    const targetVals = rows.map(r => r[targetIdx]);
    const tMean = targetVals.reduce((s, v) => s + v, 0) / targetVals.length;
    const tVar = targetVals.reduce((s, v) => s + (v - tMean) ** 2, 0) / targetVals.length;
    const tStd = Math.sqrt(tVar) || 1;

    // Calculate R² on training data
    const predictions = engine.getBestPredictions();
    if (predictions) {
      let ssRes = 0, ssTot = 0;
      for (let i = 0; i < rows.length; i++) {
        const pred = predictions[i] * tStd + tMean; // un-normalize
        const actual = rows[i][targetIdx];
        ssRes += (actual - pred) ** 2;
        ssTot += (actual - tMean) ** 2;
      }
      const r2 = 1 - ssRes / (ssTot || 1);
      const rmse = Math.sqrt(ssRes / rows.length);

      console.log(`📈 Real-Scale Metrics:`);
      console.log(`   R²:   ${r2.toFixed(4)}`);
      console.log(`   RMSE: ${rmse.toFixed(2)} (in $1000s)`);
      console.log(`   (For reference: linear regression R² ≈ 0.63 on same features)\n`);
    }
  }

  // Show the evolved expression
  const best = result.bestGenome;
  if (best && best.genes && best.genes.length > 0) {
    const { genomeToString } = require('../src/core/genome');
    console.log(`🧪 Evolved Expression:`);
    console.log(`   ${genomeToString(best)}\n`);
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
