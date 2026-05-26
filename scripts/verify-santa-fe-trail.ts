#!/usr/bin/env ts-node

import { Genome } from '../src/core/types';
import { getSantaFeFoodCount, simulateSantaFeTrail } from '../src/solvers/santa-fe-trail';

function terminal(id: string, operation: string): Genome['genes'][number] {
  return { id, type: 'terminal', operation, operands: [] };
}

function fn(id: string, operation: string, operands: string[]): Genome['genes'][number] {
  return { id, type: 'function', operation, operands };
}

function genome(genes: Genome['genes']): Genome {
  return { id: 'verify-santa-fe', genes, fitness: -Infinity, generation: 0, age: 0 };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function main(): void {
  assert(getSantaFeFoodCount() === 89, `expected canonical Santa Fe food count 89, got ${getSantaFeFoodCount()}`);

  const singleMove = simulateSantaFeTrail(genome([terminal('m', 'move')]));
  assert(singleMove.foodCollected === 3, `single move replay should collect first 3 food cells, got ${singleMove.foodCollected}`);
  assert(singleMove.stepsUsed === 600, `single move replay should use 600 steps, got ${singleMove.stepsUsed}`);
  assert(Math.abs(singleMove.fitness - 3 / 89) < 1e-12, `single move fitness mismatch: ${singleMove.fitness}`);

  const canonical = simulateSantaFeTrail(genome([
    fn('root', 'ifFoodAhead', ['move', 'right']),
    terminal('move', 'move'),
    terminal('right', 'turnRight')
  ]));
  const legacy = simulateSantaFeTrail(genome([
    fn('root', 'if-food-ahead', ['move', 'right']),
    terminal('move', 'move'),
    terminal('right', 'right')
  ]));
  assert(legacy.foodCollected === canonical.foodCollected, 'legacy operation aliases changed food count');
  assert(legacy.stepsUsed === canonical.stepsUsed, 'legacy operation aliases changed step count');
  assert(Math.abs(legacy.fitness - canonical.fitness) < 1e-12, 'legacy operation aliases changed fitness');

  console.log(JSON.stringify({
    pass: true,
    checks: ['canonical food count', 'single move replay', 'legacy operation aliases'],
    santaFe: {
      foodCount: getSantaFeFoodCount(),
      maxSteps: 600,
      singleMove: {
        foodCollected: singleMove.foodCollected,
        stepsUsed: singleMove.stepsUsed,
        fitness: singleMove.fitness
      },
      aliasReplay: {
        foodCollected: legacy.foodCollected,
        stepsUsed: legacy.stepsUsed,
        fitness: legacy.fitness
      }
    }
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
