/**
 * Genetic Code Lab - Tests
 */

import {
  Genome,
  Gene,
  PopulationConfig,
  DEFAULT_POPULATION_CONFIG
} from '../src/core/types';
import {
  generateId,
  random,
  randomInt,
  randomChoice,
  mean,
  stdDev,
  clamp,
  setSeed
} from '../src/utils/helpers';
import {
  GenomeBuilder,
  GenomeFactory,
  getGenomeDepth,
  getGenomeSize,
  validateGenome,
  cloneGenome,
  genomeToString
} from '../src/core/genome';
import {
  PopulationManager,
  DEFAULT_POPULATION_CONFIG as POP_CONFIG
} from '../src/core/population';
import {
  tournamentSelection,
  subtreeCrossover,
  pointMutation
} from '../src/core/operators';
import { MATH_FUNCTIONS, createMathTerminals, executeMathGenome } from '../src/solvers/math-expression';
import { createSortingProblem, executeSortingNetwork, extractCompareExchanges } from '../src/solvers/sorting-network';

// ============================================================================
// HELPER TESTS
// ============================================================================

describe('Utility Functions', () => {
  test('generateId should return unique IDs', () => {
    const id1 = generateId();
    const id2 = generateId();
    expect(id1).not.toBe(id2);
    expect(id1.length).toBeGreaterThan(0);
  });

  test('randomInt should return integers in range', () => {
    for (let i = 0; i < 100; i++) {
      const val = randomInt(0, 10);
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(10);
      expect(Number.isInteger(val)).toBe(true);
    }
  });

  test('randomChoice should return elements from array', () => {
    const arr = [1, 2, 3, 4, 5];
    for (let i = 0; i < 10; i++) {
      const val = randomChoice(arr);
      expect(arr).toContain(val);
    }
  });

  test('mean should calculate correctly', () => {
    expect(mean([1, 2, 3, 4, 5])).toBe(3);
    expect(mean([10])).toBe(10);
    expect(mean([])).toBe(0);
  });

  test('stdDev should calculate correctly', () => {
    const values = [2, 4, 4, 4, 5, 5, 7, 9];
    const std = stdDev(values);
    expect(std).toBeCloseTo(2, 0);
  });

  test('clamp should work correctly', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  test('setSeed should make random reproducible', () => {
    setSeed(42);
    const r1 = random();
    setSeed(42);
    const r2 = random();
    expect(r1).toBe(r2);
  });
});

// ============================================================================
// GENOME TESTS
// ============================================================================

describe('Genome', () => {
  const config: PopulationConfig = DEFAULT_POPULATION_CONFIG;
  const factory = new GenomeFactory(MATH_FUNCTIONS, createMathTerminals(['x']), config);

  test('GenomeBuilder should create valid genome', () => {
    const gene: Gene = {
      id: generateId(),
      type: 'terminal',
      operation: 'x',
      operands: []
    };

    const genome = new GenomeBuilder()
      .addGene(gene)
      .setGeneration(0)
      .build();

    expect(genome.genes.length).toBe(1);
    expect(genome.generation).toBe(0);
    expect(genome.fitness).toBe(-Infinity);
    expect(genome.id).toBeDefined();
  });

  test('GenomeFactory should create random genome', () => {
    const genome = factory.createRandom();

    expect(genome.genes.length).toBeGreaterThan(0);
    expect(genome.fitness).toBe(-Infinity);
    expect(validateGenome(genome)).toBe(true);
  });

  test('GenomeFactory should create population', () => {
    const genomes = factory.createPopulation(50);

    expect(genomes.length).toBe(50);

    // All genomes should have unique IDs
    const ids = new Set(genomes.map(g => g.id));
    expect(ids.size).toBe(50);
  });

  test('getGenomeSize should return correct size', () => {
    const genome = factory.createRandom();
    expect(getGenomeSize(genome)).toBe(genome.genes.length);
  });

  test('cloneGenome should create independent copy', () => {
    const original = factory.createRandom();
    const clone = cloneGenome(original);

    expect(clone.id).not.toBe(original.id);
    expect(clone.genes).toEqual(original.genes);
    expect(clone.age).toBe(original.age + 1);
  });

  test('genomeToString should produce readable output', () => {
    const gene: Gene = {
      id: generateId(),
      type: 'terminal',
      operation: 'x',
      operands: []
    };

    const genome = new GenomeBuilder().addGene(gene).build();
    const str = genomeToString(genome);

    expect(str).toContain('x');
  });
});

// ============================================================================
// POPULATION TESTS
// ============================================================================

describe('Population', () => {
  const config: PopulationConfig = { ...POP_CONFIG, size: 50 };
  const manager = new PopulationManager(MATH_FUNCTIONS, createMathTerminals(['x']), config);

  test('PopulationManager should initialize population', () => {
    const pop = manager.initialize();

    expect(pop.genomes.length).toBe(50);
    expect(pop.generation).toBe(0);
  });

  test('getBest should return best genome', () => {
    const pop = manager.initialize();

    // Set fitness values
    pop.genomes[0].fitness = 0.9;
    pop.genomes[1].fitness = 0.5;
    pop.genomes[2].fitness = 0.3;

    const best = manager.getBest(pop);
    expect(best.fitness).toBe(0.9);
  });

  test('getElites should return top performers', () => {
    const pop = manager.initialize();

    // Set fitness values
    pop.genomes.forEach((g, i) => {
      g.fitness = i / pop.genomes.length;
    });

    const elites = manager.getElites(pop, 5);
    expect(elites.length).toBe(5);

    // All elites should have high fitness
    elites.forEach(e => {
      expect(e.fitness).toBeGreaterThanOrEqual(0.9);
    });
  });
});

// ============================================================================
// OPERATOR TESTS
// ============================================================================

describe('Genetic Operators', () => {
  const config: PopulationConfig = DEFAULT_POPULATION_CONFIG;
  const factory = new GenomeFactory(MATH_FUNCTIONS, createMathTerminals(['x']), config);

  test('tournamentSelection should select genomes', () => {
    const genomes = factory.createPopulation(20);
    const pop = {
      id: generateId(),
      generation: 0,
      genomes,
      bestFitness: 1,
      averageFitness: 0.5,
      worstFitness: 0,
      diversity: 0.5,
      config
    };

    const selected = tournamentSelection(pop, 5);

    expect(selected.length).toBe(5);
    selected.forEach(s => {
      expect(s).toBeDefined();
      expect(s.genes.length).toBeGreaterThan(0);
    });
  });

  test('subtreeCrossover should create children', () => {
    const parent1 = factory.createRandom();
    const parent2 = factory.createRandom();

    const [child1, child2] = subtreeCrossover(parent1, parent2);

    expect(child1.id).not.toBe(parent1.id);
    expect(child1.id).not.toBe(parent2.id);
    expect(child1.parents).toEqual([parent1.id, parent2.id]);
    expect(child1.fitness).toBe(-Infinity);
  });

  test('pointMutation should modify genome', () => {
    const genome = factory.createRandom();
    genome.fitness = 0.5;

    const mutated = pointMutation(genome, { ...config, mutationRate: 1.0 }, MATH_FUNCTIONS);

    expect(mutated.id).not.toBe(genome.id);
    expect(mutated.fitness).toBe(-Infinity);
  });
});

// ============================================================================
// MATH EXPRESSION TESTS
// ============================================================================

describe('Math Expression Solver', () => {
  test('executeMathGenome should evaluate simple expression', () => {
    const gene: Gene = {
      id: generateId(),
      type: 'terminal',
      operation: 'x',
      operands: []
    };

    const genome: Genome = {
      id: generateId(),
      genes: [gene],
      fitness: 0,
      generation: 0,
      age: 0
    };

    const result = executeMathGenome(genome, { x: 5 });
    expect(result).toBe(5);
  });

  test('executeMathGenome should evaluate constant', () => {
    const gene: Gene = {
      id: generateId(),
      type: 'terminal',
      operation: 'const',
      operands: [42]
    };

    const genome: Genome = {
      id: generateId(),
      genes: [gene],
      fitness: 0,
      generation: 0,
      age: 0
    };

    const result = executeMathGenome(genome, { x: 0 });
    expect(result).toBe(42);
  });
});

// ============================================================================
// SORTING TESTS
// ============================================================================

describe('Sorting Network Solver', () => {
  test('extractCompareExchanges should extract operations', () => {
    const genome: Genome = {
      id: generateId(),
      genes: [
        {
          id: generateId(),
          type: 'function',
          operation: 'compare',
          operands: [0, 2],
          metadata: { arity: 2 }
        },
        {
          id: generateId(),
          type: 'function',
          operation: 'compare',
          operands: [1, 3],
          metadata: { arity: 2 }
        }
      ],
      fitness: 0,
      generation: 0,
      age: 0
    };

    const ops = extractCompareExchanges(genome, 4);

    expect(ops.length).toBe(2);
    expect(ops[0]).toEqual({ i: 0, j: 2 });
    expect(ops[1]).toEqual({ i: 1, j: 3 });
  });

  test('executeSortingNetwork should sort array', () => {
    const arr = [3, 1, 4, 2];
    const ops = [
      { i: 0, j: 1 },  // Compare 0,1 -> [1, 3, 4, 2]
      { i: 2, j: 3 },  // Compare 2,3 -> [1, 3, 2, 4]
      { i: 1, j: 2 },  // Compare 1,2 -> [1, 2, 3, 4]
      { i: 0, j: 1 },  // Compare 0,1 -> [1, 2, 3, 4]
      { i: 2, j: 3 }   // Compare 2,3 -> [1, 2, 3, 4]
    ];

    const sorted = executeSortingNetwork(arr, ops);

    expect(sorted).toEqual([1, 2, 3, 4]);
  });
});

describe('Advanced Genetic Operators', () => {
  const { 
    stochasticUniversalSampling, 
    rankSelection,
    subtreeMutation,
    hoistMutation,
    shrinkMutation
  } = require('../src/core/operators');
  const { MATH_FUNCTIONS } = require('../src/solvers/math-expression');
  const { createMathTerminals } = require('../src/solvers/math-expression');

  const config = DEFAULT_POPULATION_CONFIG;
  const terminals = createMathTerminals(['x']);

  const createPop = (fitnesses: number[]) => {
    const genomes = fitnesses.map(f => ({
      id: generateId(),
      genes: [],
      fitness: f,
      generation: 0,
      age: 0
    }));
    return {
      genomes,
      generation: 0,
      bestFitness: Math.max(...fitnesses),
      averageFitness: fitnesses.reduce((a,b) => a+b, 0) / fitnesses.length,
      worstFitness: Math.min(...fitnesses),
      diversity: 1
    };
  };

  test('rankSelection should select higher ranked genomes', () => {
    const pop = createPop([10, 20, 30, 40, 50]);
    const selected = rankSelection(pop, 2);
    expect(selected.length).toBe(2);
    expect(selected[0].fitness).toBeGreaterThan(0);
  });

  test('stochasticUniversalSampling should select genomes correctly', () => {
    const pop = createPop([10, 20, 30, 40, 50]);
    const selected = stochasticUniversalSampling(pop, 3);
    expect(selected.length).toBe(3);
  });

  test('subtreeMutation should maintain tree structure', () => {
    const factory = new GenomeFactory(MATH_FUNCTIONS, terminals, config);
    const genome = factory.createRandom();
    const mutated = subtreeMutation(genome, { ...config, mutationRate: 1.0 }, MATH_FUNCTIONS, terminals);
    expect(mutated.id).not.toBe(genome.id);
    expect(mutated.genes.length).toBeGreaterThan(0);
  });

  test('hoistMutation should produce smaller or equal tree', () => {
    const factory = new GenomeFactory(MATH_FUNCTIONS, terminals, config);
    const genome = factory.createRandom();
    const mutated = hoistMutation(genome);
    expect(mutated.genes.length).toBeLessThanOrEqual(genome.genes.length);
  });

  test('shrinkMutation should replace node with terminal', () => {
    const factory = new GenomeFactory(MATH_FUNCTIONS, terminals, config);
    const genome = factory.createRandom();
    const mutated = shrinkMutation(genome, terminals);
    expect(mutated.genes.length).toBeLessThanOrEqual(genome.genes.length);
  });
});

