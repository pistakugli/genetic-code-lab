/**
 * Genetic Code Lab - Pathfinding Example
 */

import { EvolutionEngineTree as EvolutionEngine, EvolutionConfig, EvolutionStats } from '../src/core/evolution-tree';
import { Population, Genome } from '../src/core/types';
import { 
  createPathfindingProblem, 
  extractMoves, 
  executePathfinding, 
  visualizePath,
  PathfindingTestCase
} from '../src/solvers/pathfinding';
import { formatTime } from '../src/utils/helpers';
import chalk from 'chalk';

async function runExample() {
  console.log(chalk.cyan.bold('\n🧬 Genetic Code Lab - Pathfinding Example\n'));

  const width = 8;
  const height = 8;
  console.log(chalk.white(`Evolving pathfinding for ${width}x${height} grid...`));

  // Create problem
  const problem = createPathfindingProblem(width, height, 3);

  // Configure evolution
  const config: Partial<EvolutionConfig> = {
    population: {
      size: 150,
      eliteCount: 3,
      crossoverRate: 0.7,
      mutationRate: 0.3,
      maxGenomeDepth: 10,
      maxGenomeSize: 40,
      tournamentSize: 4
    },
    maxGenerations: 100,
    targetFitness: 0.9,
    stagnationLimit: 25,
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

  // Visualize best solution
  const moves = extractMoves(result.bestGenome);
  console.log(chalk.yellow('\nEvolved Sequence:'));
  console.log(chalk.white(`  ${moves.join(' → ')}`));

  const firstTest = problem.testCases[0] as PathfindingTestCase;
  const { grid, start, goal } = firstTest.input;
  const pathResult = executePathfinding(grid, start, goal, moves);

  console.log(chalk.yellow('\nSample Path:'));
  console.log(chalk.white(visualizePath(grid, pathResult.path, start, goal)));
  console.log(chalk.white(`\n  Reached goal: ${pathResult.reached ? chalk.green('Yes') : chalk.red('No')}`));
  console.log(chalk.white(`  Steps: ${pathResult.steps}`));
}

runExample().catch(err => {
  console.error(chalk.red('Error running example:'), err);
  process.exit(1);
});
