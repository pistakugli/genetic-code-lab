/**
 * Genetic Code Lab - Sorting Network Problem
 * Evolves sorting networks (compare-exchange sequences)
 */

import {
  ProblemDefinition,
  TestCase,
  FunctionDefinition,
  TerminalDefinition,
  Genome,
  Gene,
  FitnessFunction
} from '../core/types';

// ============================================================================
// SORTING NETWORK GENE TYPES
// ============================================================================

interface CompareExchange {
  i: number;
  j: number;
}

// ============================================================================
// FUNCTION SET (for sorting networks, we use compare-exchange operations)
// ============================================================================

export const SORTING_FUNCTIONS: FunctionDefinition[] = [
  {
    name: 'compare',
    arity: 2,
    implementation: () => 0, // Handled specially
    returnType: 'void',
    description: 'Compare and swap if needed'
  }
];

// ============================================================================
// TERMINAL SET
// ============================================================================

export function createSortingTerminals(arraySize: number): TerminalDefinition[] {
  const terminals: TerminalDefinition[] = [];

  // Each terminal represents an index in the array
  for (let i = 0; i < arraySize; i++) {
    terminals.push({
      name: `idx${i}`,
      type: 'constant',
      value: i,
      description: `Index ${i}`
    });
  }

  return terminals;
}

// ============================================================================
// GENOME EXECUTION
// ============================================================================

/**
 * Extract compare-exchange operations from genome
 */
export function extractCompareExchanges(
  genome: Genome,
  arraySize: number
): CompareExchange[] {
  const operations: CompareExchange[] = [];

  for (const gene of genome.genes) {
    if (gene.type === 'terminal' && gene.operands.length > 0) {
      // Single index - pair with next
      const idx = gene.operands[0] as number;
      if (idx >= 0 && idx < arraySize - 1) {
        operations.push({ i: idx, j: idx + 1 });
      }
    } else if (gene.type === 'function' && gene.operation === 'compare') {
      // Extract two indices from operands
      const indices: number[] = [];
      for (const op of gene.operands) {
        if (typeof op === 'number') {
          indices.push(Math.floor(op) % arraySize);
        }
      }

      if (indices.length >= 2 && indices[0] !== indices[1]) {
        operations.push({
          i: Math.min(indices[0], indices[1]),
          j: Math.max(indices[0], indices[1])
        });
      }
    }
  }

  return operations;
}

/**
 * Execute sorting network on an array
 */
export function executeSortingNetwork(
  arr: number[],
  operations: CompareExchange[]
): number[] {
  const result = [...arr];

  for (const op of operations) {
    if (result[op.i] > result[op.j]) {
      [result[op.i], result[op.j]] = [result[op.j], result[op.i]];
    }
  }

  return result;
}

/**
 * Check if array is sorted
 */
export function isSorted(arr: number[]): boolean {
  for (let i = 0; i < arr.length - 1; i++) {
    if (arr[i] > arr[i + 1]) return false;
  }
  return true;
}

// ============================================================================
// FITNESS FUNCTION
// ============================================================================

export const sortingFitnessFunction: FitnessFunction<number, number> = (
  genome: Genome,
  testCases: TestCase<number, number>[]
): number => {
  if (testCases.length === 0) return 0;

  // Extract array size from test cases
  const arraySize = Math.floor(testCases[0].expectedOutput) || 4;

  // Extract compare-exchange operations
  const operations = extractCompareExchanges(genome, arraySize);

  if (operations.length === 0) return 0;

  let correctSorts = 0;
  let totalInversions = 0;

  // Test on various inputs
  const testArrays = generateTestArrays(arraySize, 50);

  for (const arr of testArrays) {
    const sorted = executeSortingNetwork(arr, operations);

    if (isSorted(sorted)) {
      correctSorts++;
    } else {
      // Count remaining inversions as partial credit
      totalInversions += countInversions(sorted);
    }
  }

  // Fitness = correct sorts + partial credit for reducing inversions
  const sortRatio = correctSorts / testArrays.length;
  const avgInversions = totalInversions / testArrays.length;
  const inversionPenalty = avgInversions / (arraySize * arraySize);

  // Also penalize long networks (efficiency)
  const efficiencyBonus = Math.max(0, 1 - operations.length / (arraySize * arraySize));

  const fitness = sortRatio * 0.8 + efficiencyBonus * 0.1 + (1 - inversionPenalty) * 0.1;

  return Math.max(0, Math.min(1, fitness));
};

/**
 * Generate test arrays for sorting
 */
function generateTestArrays(size: number, count: number): number[][] {
  const arrays: number[][] = [];

  // Reverse sorted
  arrays.push(Array.from({ length: size }, (_, i) => size - i - 1));

  // Already sorted
  arrays.push(Array.from({ length: size }, (_, i) => i));

  // Random permutations
  for (let i = 0; i < count - 2; i++) {
    const arr = Array.from({ length: size }, (_, j) => j);
    // Fisher-Yates shuffle
    for (let j = arr.length - 1; j > 0; j--) {
      const k = Math.floor(Math.random() * (j + 1));
      [arr[j], arr[k]] = [arr[k], arr[j]];
    }
    arrays.push(arr);
  }

  return arrays;
}

/**
 * Count inversions in array
 */
function countInversions(arr: number[]): number {
  let count = 0;
  for (let i = 0; i < arr.length - 1; i++) {
    for (let j = i + 1; j < arr.length; j++) {
      if (arr[i] > arr[j]) count++;
    }
  }
  return count;
}

// ============================================================================
// PROBLEM FACTORY
// ============================================================================

/**
 * Create a sorting network problem
 */
export function createSortingProblem(
  arraySize: number = 4
): ProblemDefinition<number, number> {
  // Create dummy test cases (fitness function generates its own test data)
  const testCases: TestCase<number, number>[] = [
    { input: arraySize, expectedOutput: arraySize }
  ];

  return {
    name: 'Sorting Network',
    description: `Evolve a sorting network for ${arraySize} elements`,
    functionSet: SORTING_FUNCTIONS,
    terminalSet: createSortingTerminals(arraySize),
    testCases,
    fitnessFunction: sortingFitnessFunction,
    config: {
      population: {
        size: 200,
        eliteCount: 4,
        crossoverRate: 0.7,
        mutationRate: 0.3,
        maxGenomeDepth: 6,
        maxGenomeSize: arraySize * arraySize,
        tournamentSize: 5
      },
      maxGenerations: 200,
      targetFitness: 0.95,
      stagnationLimit: 30
    }
  };
}

/**
 * Convert evolved genome to human-readable sorting network
 */
export function genomeToSortingNetwork(
  genome: Genome,
  arraySize: number
): string {
  const operations = extractCompareExchanges(genome, arraySize);

  const lines: string[] = [`Sorting Network for ${arraySize} elements:`];
  lines.push(`Total compare-exchange operations: ${operations.length}`);
  lines.push('');
  lines.push('Operations:');

  for (let i = 0; i < operations.length; i++) {
    const op = operations[i];
    lines.push(`  ${i + 1}. Compare positions [${op.i}] and [${op.j}], swap if needed`);
  }

  // Visual representation
  lines.push('');
  lines.push('Visual:');
  for (let i = 0; i < arraySize; i++) {
    let line = `  [${i}] `;
    for (let j = 0; j < operations.length; j++) {
      const op = operations[j];
      if (op.i === i || op.j === i) {
        line += '═╪═ ';
      } else {
        line += '─── ';
      }
    }
    lines.push(line);
  }

  return lines.join('\n');
}
