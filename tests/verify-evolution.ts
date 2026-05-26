import { EvolutionEngine } from '../src/core/evolution';
import { createSymbolicRegressionProblem } from '../src/solvers/math-expression';
import { setSeed } from '../src/utils/helpers';

/**
 * Verification script: Tests if the engine can solve a simple math problem:
 * Target: f(x) = x^2 + x + 1
 */
async function verifyMathEvolution() {
  console.log('--- VERIFYING EVOLUTION ENGINE (MATH) ---');
  setSeed(12345); // Reproducible test

  const targetFunction = (x: number) => x * x + x + 1;
  const xValues = [-5, -2, 0, 2, 5];
  const dataPoints = xValues.map(x => ({ x, y: targetFunction(x) }));

  const problem = createSymbolicRegressionProblem(dataPoints);

  const config = {
    population: {
      size: 100,
      eliteCount: 5,
      mutationRate: 0.2,
      crossoverRate: 0.8,
      maxGenomeDepth: 5,
      maxGenomeSize: 100,
      tournamentSize: 3
    },
    targetFitness: 0.99,
    maxGenerations: 40,
    stagnationLimit: 15,
    onGeneration: (pop: any, stats: any) => {
      console.log(`Gen ${pop.generation}: Best Fitness = ${stats.bestFitness.toFixed(4)}, Avg = ${stats.averageFitness.toFixed(4)}`);
    }
  };

  const engine = new EvolutionEngine(problem, config);
  const result = await engine.evolve();

  console.log('\n--- RESULT ---');
  console.log(`Success: ${result.success}`);
  console.log(`Final Generation: ${result.generationsRun}`);
  console.log(`Best Fitness: ${result.bestGenome.fitness.toFixed(4)}`);
  
  if (result.bestGenome.fitness < 0.5) {
     console.error('FAILED: Fitness did not reach acceptable threshold (> 0.5)');
     process.exit(1);
  }

  const initialBest = result.stats[0].bestFitness;
  const finalBest = result.bestGenome.fitness;
  
  if (finalBest < initialBest) {
     console.error('FAILED: Final fitness is worse than initial best fitness');
     process.exit(1);
  }

  console.log('SUCCESS: Evolution engine is functional and improving fitness!');
}

verifyMathEvolution().catch(err => {
  console.error(err);
  process.exit(1);
});
