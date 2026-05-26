/**
 * Genetic Code Lab - Sorting Network Example
 */

import { EvolutionEngineTree as EvolutionEngine, EvolutionConfig, EvolutionStats } from '../src/core/evolution-tree';
import { Population, Genome } from '../src/core/types';
import { createSortingProblem, genomeToSortingNetwork } from '../src/solvers/sorting-network';
import { formatTime } from '../src/utils/helpers';
import chalk from 'chalk';

async function runExample() {
  console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Sorting Network Example\n'));

  const arraySize = 5;
  console.log(chalk.white(`Evolving sorting network for ${arraySize} elements...`));

  // Create problem
  const problem = createSortingProblem(arraySize);

  // Configure evolution
  const config: Partial<EvolutionConfig> = {
    population: {
      size: 200,
      eliteCount: 5,
      crossoverRate: 0.8,
      mutationRate: 0.2,
      maxGenomeDepth: 6,
      maxGenomeSize: 30,
      tournamentSize: 5
    },
    maxGenerations: 100,
    targetFitness: 1.0,
    stagnationLimit: 30,
    onGeneration: (pop: Population, stats: EvolutionStats) => {
      if (stats.generation % 10 === 0) {
        console.log(chalk.gray(`  Gen ${stats.generation}: fitness = ${stats.bestFitness.toFixed(4)}`));
      }
    },
    onImprovement: (genome: Genome, gen: number) => {
      console.log(chalk.green(`  ⬆️  Improvement at gen ${gen}: ${genome.fitness.toFixed(4)}`));
    }
  };

  // Run evolution
  const engine = new EvolutionEngine(problem, config);
  const result = await engine.evolve();

  console.log(chalk.cyan('\n' + '='.repeat(50)));
  console.log(chalk.white('Results:'));
  console.log(chalk.white(`  Success: ${result.success ? chalk.green('✅') : chalk.red('❌')}`));
  console.log(chalk.white(`  Best Fitness: ${result.bestGenome.fitness.toFixed(6)}`));
  console.log(chalk.white(`  Generations: ${result.generationsRun}`));
  console.log(chalk.white(`  Evaluations: ${result.totalEvaluations}`));
  console.log(chalk.white(`  Time: ${formatTime(result.totalTimeMs)}`));

  console.log(chalk.yellow('\nEvolved Sorting Network:'));
  console.log(chalk.white(genomeToSortingNetwork(result.bestGenome, arraySize)));
}

runExample().catch(err => {
  console.error(chalk.red('Error running example:'), err);
  process.exit(1);
});
