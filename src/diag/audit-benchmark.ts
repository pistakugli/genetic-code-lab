import { EvolutionEngineTree as EvolutionEngine } from '../core/evolution-tree';
import { createSantaFeTrailProblem } from '../solvers/santa-fe-trail';
import { EvolutionConfig } from '../core/types';
import * as fs from 'fs-extra';
import { v4 as uuidv4 } from 'uuid';

/**
 * Beyond SOTA: Strict Audit Protocol
 * =================================
 * This script runs multiple evolution trials and outputs a structured JSON 
 * result for statistical verification.
 */

async function runAuditTrial(regime: 'industrial' | 'frontier', runIdx: number) {
  const problem = createSantaFeTrailProblem();
  
  const config: Partial<EvolutionConfig> = {
    solverType: 'santa-fe',
    population: {
      size: regime === 'industrial' ? 200 : 500, // Reduced for tree engine
      maxGenomeDepth: 7,
      maxGenomeSize: 150,
      mutationRate: 0.3,
      crossoverRate: 0.8,
      eliteCount: 2,
      tournamentSize: 7,
      parsimonyCoefficient: 0.001
    },
    selectionMethod: regime === 'industrial' ? 'tournament' : 'lexicase',
    parallelEvaluators: 1, // Tree engine is synchronous
    maxGenerations: regime === 'industrial' ? 50 : 15,
    targetFitness: 1.0,
    stagnationLimit: 20,
    onGeneration: (pop) => {
      const best = pop.genomes.reduce((a, b) => b.fitness > a.fitness ? b : a, pop.genomes[0]);
      console.log(`[Gen ${pop.generation}] Best Fitness: ${best.fitness.toFixed(4)} | Nodes: ${best.genes.length}`);
    }
  };

  const engine = new EvolutionEngine(problem, config);
  const startTime = Date.now();
  
  console.log(`[Audit] Starting ${regime} Run ${runIdx}...`);
  const result = await engine.evolve();
  const timeMs = Date.now() - startTime;

  const lastStats = result.stats && result.stats.length > 0 ? result.stats[result.stats.length - 1] : { diversity: 0 };
  
  const auditResult = {
    run_id: uuidv4(),
    regime,
    iteration: runIdx,
    best_fitness: result.bestGenome.fitness,
    generations: result.generationsRun,
    time_ms: timeMs,
    evals_per_sec: Math.floor(result.totalEvaluations / (timeMs / 1000 || 1)),
    diversity: lastStats.diversity,
    best_structure: result.bestGenome.genes.map(g => g.operation).join(' '),
    memory_peak_mb: Math.floor(process.memoryUsage().rss / (1024 * 1024))
  };

  return auditResult;
}

async function startAudit() {
  const regime = (process.argv[2] as 'industrial' | 'frontier') || 'industrial';
  const trials = parseInt(process.argv[3] || '3');
  const results = [];

  console.log(`\nStarting Beyond SOTA Audit [Regime: ${regime}, Trials: ${trials}]`);
  console.log('────────────────────────────────────────────────────────────');

  for (let i = 0; i < trials; i++) {
    const res = await runAuditTrial(regime, i + 1);
    results.push(res);
    console.log(`[Result] Fitness: ${(res.best_fitness * 100).toFixed(2)}% | Throughput: ${res.evals_per_sec} eval/s`);
  }

  const logFile = `./audit-${regime}-${Date.now()}.json`;
  await fs.writeJson(logFile, results, { spaces: 2 });
  console.log(`\nAudit complete. Data written to ${logFile}`);
}

startAudit().catch(err => {
  console.error('[Audit Failure]', err);
  process.exit(1);
});
