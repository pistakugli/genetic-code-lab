/**
 * Integration Tests for Genetic Code Lab
 * Runs full evolution cycles for different problems to verify engine performance
 * and ensure all components work together in production-like scenarios.
 */

import { EvolutionEngine } from '../src/core/evolution';
import { EvolutionEngineTree } from '../src/core/evolution-tree';
import { createMathProblem } from '../src/solvers/math-expression';
import { createSortingProblem, isSorted, extractCompareExchanges, executeSortingNetwork } from '../src/solvers/sorting-network';
import { createPathfindingProblem, extractMoves, executePathfinding } from '../src/solvers/pathfinding';

describe('Evolution Integration Tests', () => {
  // Set a timeout for long-running evolution tests
  jest.setTimeout(30000);

  test('Math Expression evolution should improve fitness', async () => {
    const problem = createMathProblem((x) => (x[0] + 1) * 2, 1, 20);
    const engine = new EvolutionEngine(problem, {
      ...problem.config,
      maxGenerations: 30
    });

    const result = await engine.evolve();
    expect(result.bestGenome.fitness).toBeGreaterThan(0.1); 
    expect(result.generationsRun).toBeLessThanOrEqual(30);
  });

  test('Sorting Network evolution should improve fitness', async () => {
    const problem = createSortingProblem(4);
    const engine = new EvolutionEngineTree(problem, {
      ...problem.config,
      maxGenerations: 30
    });

    const result = await engine.evolve();
    expect(result.bestGenome.fitness).toBeGreaterThan(0.1);
    
    // Verify best genome actually sorts something
    const testArr = [3, 1, 4, 2];
    const ops = extractCompareExchanges(result.bestGenome, 4);
    const sorted = executeSortingNetwork(testArr, ops);
    // This asserted only that the output differed from the input, which any permutation
    // satisfies including one that leaves the array less sorted than it started. The
    // isSorted import at the top of this file was unused, which is what a weakened
    // assertion leaves behind. A sorting network is either correct on this input or it is
    // not, so that is what gets asserted.
    expect(sorted).toEqual([1, 2, 3, 4]);
    expect(isSorted(sorted)).toBe(true);
  });

  test('Pathfinding evolution should improve fitness', async () => {
    const problem = createPathfindingProblem(8, 8, 3);
    const engine = new EvolutionEngineTree(problem, {
      ...problem.config,
      maxGenerations: 30
    });

    const result = await engine.evolve();
    expect(result.bestGenome.fitness).toBeGreaterThan(0);
    
    // Verify moves are extracted
    const moves = extractMoves(result.bestGenome);
    expect(moves.length).toBeGreaterThan(0);
  });
});
