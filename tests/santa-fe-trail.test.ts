import { Genome } from '../src/core/types';
import { getSantaFeFoodCount, simulateSantaFeTrail } from '../src/solvers/santa-fe-trail';

function terminal(id: string, operation: string) {
  return { id, type: 'terminal' as const, operation, operands: [] };
}

function fn(id: string, operation: string, operands: string[]) {
  return { id, type: 'function' as const, operation, operands };
}

function genome(genes: Genome['genes']): Genome {
  return {
    id: 'test-genome',
    genes,
    fitness: -Infinity,
    generation: 0,
    age: 0
  };
}

describe('Santa Fe Trail solver', () => {
  test('uses the canonical 89-food Santa Fe trail', () => {
    expect(getSantaFeFoodCount()).toBe(89);
  });

  test('single move program replays against the canonical start cell', () => {
    const result = simulateSantaFeTrail(genome([terminal('m', 'move')]));

    expect(result.foodCollected).toBe(3);
    expect(result.stepsUsed).toBe(600);
    expect(result.fitness).toBeCloseTo(3 / 89, 12);
    expect(result.trace.slice(0, 3)).toEqual([
      { step: 1, x: 1, y: 0, foodCollected: 1 },
      { step: 2, x: 2, y: 0, foodCollected: 2 },
      { step: 3, x: 3, y: 0, foodCollected: 3 }
    ]);
  });

  test('legacy operation aliases replay through the same simulator', () => {
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

    expect(legacy.foodCollected).toBe(canonical.foodCollected);
    expect(legacy.stepsUsed).toBe(canonical.stepsUsed);
    expect(legacy.fitness).toBeCloseTo(canonical.fitness, 12);
    expect(legacy.foodCollected).toBeGreaterThan(3);
  });
});
