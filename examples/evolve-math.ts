/**
 * Example: Evolve a mathematical expression
 *
 * This example shows how to use Genetic Code Lab to evolve
 * a mathematical expression that matches a target function.
 */

import { EvolutionEngine, EvolutionConfig } from '../src/core/evolution';
import {
  createMathProblem,
  createSymbolicRegressionProblem,
  executeMathGenome,
  genomeToString
} from '../src/solvers/math-expression';

async function main() {
  console.log('🧬 Genetic Code Lab - Math Expression Example\n');

  // =========================================================================
  // Example 1: Find expression that computes x^2 + 2x + 1
  // =========================================================================

  console.log('Example 1: Finding expression for x² + 2x + 1');
  console.log('=' .repeat(50));

  // Target function: (x + 1)^2 = x^2 + 2x + 1
  const targetFn = (inputs: number[]) => {
    const x = inputs[0];
    return x * x + 2 * x + 1;
  };

  // Create problem
  const problem = createMathProblem(targetFn, 1, 25);

  // Configure evolution
  const config: Partial<EvolutionConfig> = {
    population: {
      size: 100,
      eliteCount: 2,
      crossoverRate: 0.8,
      mutationRate: 0.2,
      maxGenomeDepth: 6,
      maxGenomeSize: 30,
      tournamentSize: 3
    },
    maxGenerations: 50,
    targetFitness: 0.99,
    stagnationLimit: 20,
    seed: 42  // For reproducibility
  };

  // Create evolution engine
  const engine = new EvolutionEngine(problem, {
    ...config,
    onGeneration: (pop, stats) => {
      if (stats.generation % 10 === 0) {
        console.log(`  Gen ${stats.generation}: fitness = ${stats.bestFitness.toFixed(4)}`);
      }
    },
    onImprovement: (genome, gen) => {
      console.log(`  ⬆️  Improvement at gen ${gen}: ${genome.fitness.toFixed(4)}`);
    }
  });

  // Run evolution
  console.log('\nEvolving...\n');
  const result = await engine.evolve();

  // Display results
  console.log('\n' + '='.repeat(50));
  console.log('Results:');
  console.log(`  Success: ${result.success ? '✅' : '❌'}`);
  console.log(`  Best Fitness: ${result.bestGenome.fitness.toFixed(6)}`);
  console.log(`  Generations: ${result.generationsRun}`);
  console.log(`  Evaluations: ${result.totalEvaluations}`);
  console.log(`  Time: ${result.totalTimeMs}ms`);

  // Show evolved expression
  console.log('\nEvolved Expression:');
  console.log(`  ${genomeToString(result.bestGenome)}`);

  // Test on values
  const testValues = [-2, -1, 0, 1, 2, 3, 5, 10];
  console.log('\nVerification:');
  console.log('  x  | Expected | Got      | Error');
  console.log('  ' + '-'.repeat(35));

  for (const x of testValues) {
    const expected = targetFn([x]);
    const got = executeMathGenome(result.bestGenome, { x: x });
    const error = Math.abs(expected - got).toFixed(4);
    console.log(`  ${x.toString().padEnd(4)} | ${expected.toFixed(4).padEnd(8)} | ${got.toFixed(4).padEnd(8)} | ${error}`);
  }

  // =========================================================================
  // Example 2: Symbolic Regression
  // =========================================================================

  console.log('\n\n' + '='.repeat(50));
  console.log('Example 2: Symbolic Regression');
  console.log('='.repeat(50));

  // Generate data points from a more complex function
  const dataPoints = [];
  for (let x = -5; x <= 5; x += 0.5) {
    dataPoints.push({
      x,
      y: Math.sin(x) + 0.5 * x  // y = sin(x) + 0.5x
    });
  }

  const regressionProblem = createSymbolicRegressionProblem(dataPoints);

  const regressionConfig: Partial<EvolutionConfig> = {
    population: {
      size: 200,
      eliteCount: 4,
      crossoverRate: 0.75,
      mutationRate: 0.25,
      maxGenomeDepth: 8,
      maxGenomeSize: 40,
      tournamentSize: 4
    },
    maxGenerations: 100,
    targetFitness: 0.95,
    stagnationLimit: 30,
    seed: 123
  };

  const regressionEngine = new EvolutionEngine(regressionProblem, {
    ...regressionConfig,
    onGeneration: (pop, stats) => {
      if (stats.generation % 20 === 0) {
        console.log(`  Gen ${stats.generation}: fitness = ${stats.bestFitness.toFixed(4)}`);
      }
    }
  });

  console.log('\nEvolving regression model...\n');
  const regressionResult = await regressionEngine.evolve();

  console.log('\nResults:');
  console.log(`  Success: ${regressionResult.success ? '✅' : '❌'}`);
  console.log(`  Best Fitness: ${regressionResult.bestGenome.fitness.toFixed(6)}`);
  console.log(`  Generations: ${regressionResult.generationsRun}`);

  console.log('\nEvolved Model:');
  console.log(`  ${genomeToString(regressionResult.bestGenome)}`);

  // Sample predictions
  console.log('\nSample Predictions:');
  console.log('  x    | Actual    | Predicted');
  console.log('  ' + '-'.repeat(35));

  for (const x of [-3, -1, 0, 1, 3]) {
    const actual = Math.sin(x) + 0.5 * x;
    const predicted = executeMathGenome(regressionResult.bestGenome, { x });
    console.log(`  ${x.toFixed(1).padEnd(4)} | ${actual.toFixed(4).padEnd(9)} | ${predicted.toFixed(4)}`);
  }
}

main().catch(console.error);
