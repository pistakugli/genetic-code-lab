/**
 * Genetic Code Lab - Math Expression Problem
 * Evolves mathematical expressions to match target values
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
import { genomeToString } from '../core/genome';
import { isValidNumber, safeDivide } from '../utils/helpers';

// Re-export genomeToString for convenience
export { genomeToString };

// ============================================================================
// FUNCTION SET
// ============================================================================

export const MATH_FUNCTIONS: FunctionDefinition[] = [
  {
    name: 'add',
    arity: 2,
    implementation: (a, b) => (a as number) + (b as number),
    returnType: 'number',
    description: 'Addition'
  },
  {
    name: 'sub',
    arity: 2,
    implementation: (a, b) => (a as number) - (b as number),
    returnType: 'number',
    description: 'Subtraction'
  },
  {
    name: 'mul',
    arity: 2,
    implementation: (a, b) => (a as number) * (b as number),
    returnType: 'number',
    description: 'Multiplication'
  },
  {
    name: 'div',
    arity: 2,
    implementation: (a, b) => safeDivide(a as number, b as number),
    returnType: 'number',
    description: 'Division (safe)'
  },
  {
    name: 'neg',
    arity: 1,
    implementation: (a) => -(a as number),
    returnType: 'number',
    description: 'Negation'
  },
  {
    name: 'abs',
    arity: 1,
    implementation: (a) => Math.abs(a as number),
    returnType: 'number',
    description: 'Absolute value'
  },
  {
    name: 'sqrt',
    arity: 1,
    implementation: (a) => {
      const v = a as number;
      return v >= 0 ? Math.sqrt(v) : 0;
    },
    returnType: 'number',
    description: 'Square root (safe)'
  },
  {
    name: 'exp',
    arity: 1,
    implementation: (a) => Math.exp(Math.max(-40, Math.min(40, a as number))),
    returnType: 'number',
    description: 'Exponential (clamped)'
  },
  {
    name: 'log',
    arity: 1,
    implementation: (a) => Math.log(Math.abs(a as number) + 1e-12),
    returnType: 'number',
    description: 'Natural logarithm (safe)'
  },
  {
    name: 'pow',
    arity: 2,
    implementation: (a, b) => {
      const result = Math.pow(a as number, b as number);
      return isValidNumber(result) ? result : 0;
    },
    returnType: 'number',
    description: 'Power'
  },
  {
    name: 'sin',
    arity: 1,
    implementation: (a) => Math.sin(a as number),
    returnType: 'number',
    description: 'Sine'
  },
  {
    name: 'cos',
    arity: 1,
    implementation: (a) => Math.cos(a as number),
    returnType: 'number',
    description: 'Cosine'
  },
  {
    name: 'tanh',
    arity: 1,
    implementation: (a) => Math.tanh(a as number),
    returnType: 'number',
    description: 'Hyperbolic tangent'
  },
  {
    name: 'min',
    arity: 2,
    implementation: (a, b) => Math.min(a as number, b as number),
    returnType: 'number',
    description: 'Minimum'
  },
  {
    name: 'max',
    arity: 2,
    implementation: (a, b) => Math.max(a as number, b as number),
    returnType: 'number',
    description: 'Maximum'
  }
];

// ============================================================================
// TERMINAL SET
// ============================================================================

export function createMathTerminals(
  variables: string[] = ['x0'],
  constantRange: [number, number] = [-10, 10]
): TerminalDefinition[] {
  const terminals: TerminalDefinition[] = [];

  // Add variables
  for (const v of variables) {
    terminals.push({
      name: v,
      type: 'variable',
      description: `Variable ${v}`
    });
  }

  // Add common constants
  terminals.push(
    { name: 'const', type: 'constant', value: 0, description: 'Zero' },
    { name: 'const', type: 'constant', value: 1, description: 'One' },
    { name: 'const', type: 'constant', value: 2, description: 'Two' },
    { name: 'const', type: 'constant', value: Math.PI, description: 'Pi' },
    { name: 'const', type: 'constant', value: Math.E, description: 'Euler' }
  );

  // Add random constants in range
  for (let i = 0; i < 5; i++) {
    const value = constantRange[0] + Math.random() * (constantRange[1] - constantRange[0]);
    terminals.push({
      name: 'const',
      type: 'constant',
      value: Math.round(value * 100) / 100,
      description: 'Random constant'
    });
  }

  return terminals;
}

// ============================================================================
// GENOME EXECUTION
// ============================================================================

/**
 * Execute a genome as a math expression
 */
export function executeMathGenome(
  genome: Genome,
  inputs: Record<string, number>
): number {
  if (genome.genes.length === 0) return 0;

  const geneMap = new Map<string, Gene>();
  genome.genes.forEach(g => geneMap.set(g.id, g));

  function executeGene(gene: Gene): number {
    if (gene.type === 'terminal') {
      // Check if it's a constant or variable
      if (gene.operation === 'const' && gene.operands.length > 0) {
        return gene.operands[0] as number;
      }
      // It's a variable
      return inputs[gene.operation] ?? 0;
    }

    // Function gene
    const func = MATH_FUNCTIONS.find(f => f.name === gene.operation);
    if (!func) return 0;

    // Evaluate operands
    const args: number[] = [];
    for (const op of gene.operands) {
      if (typeof op === 'string') {
        const childGene = geneMap.get(op);
        if (childGene) {
          args.push(executeGene(childGene));
        }
      } else if (typeof op === 'number') {
        args.push(op);
      }
    }

    // Execute function
    const result = func.implementation(...args);
    return isValidNumber(result) ? (result as number) : 0;
  }

  return executeGene(genome.genes[0]);
}

// ============================================================================
// FITNESS FUNCTION
// ============================================================================

export const mathFitnessFunction: FitnessFunction<number[], number> = (
  genome: Genome,
  testCases: TestCase<number[], number>[]
): number => {
  let totalError = 0;
  let validCases = 0;

  for (const tc of testCases) {
    const inputs: Record<string, number> = {};
    tc.input.forEach((val, idx) => {
      inputs[`x${idx === 0 ? '' : idx}`] = val;
    });

    try {
      const output = executeMathGenome(genome, inputs);
      const expected = tc.expectedOutput;

      if (isValidNumber(output) && isValidNumber(expected)) {
        // Use relative error for better scaling
        const absError = Math.abs(output - expected);
        const relError = expected !== 0
          ? absError / (Math.abs(expected) + 1)
          : absError;

        totalError += relError * (tc.weight ?? 1);
        validCases++;
      } else {
        totalError += 1000; // Penalty for invalid output
      }
    } catch {
      totalError += 1000; // Penalty for execution errors
    }
  }

  // Convert error to fitness (higher is better)
  const avgError = validCases > 0 ? totalError / validCases : 1000;
  const fitness = 1 / (1 + avgError);

  return Math.max(0, Math.min(1, fitness));
};

// ============================================================================
// PROBLEM FACTORY
// ============================================================================

/**
 * Create a math expression problem
 */
export function createMathProblem(
  targetExpression: (inputs: number[]) => number,
  variableCount: number = 1,
  testCount: number = 20
): ProblemDefinition<number[], number> {
  // Generate test cases
  const testCases: TestCase<number[], number>[] = [];

  for (let i = 0; i < testCount; i++) {
    const inputs: number[] = [];
    for (let j = 0; j < variableCount; j++) {
      inputs.push((Math.random() - 0.5) * 20); // Random values between -10 and 10
    }

    const expected = targetExpression(inputs);
    if (isValidNumber(expected)) {
      testCases.push({
        input: inputs,
        expectedOutput: expected,
        weight: 1
      });
    }
  }

  // Create variable names
  const variables = Array.from(
    { length: variableCount },
    (_, i) => `x${i}`
  );

  return {
    name: 'Math Expression',
    description: 'Evolve a mathematical expression to match target behavior',
    functionSet: MATH_FUNCTIONS,
    terminalSet: createMathTerminals(variables),
    testCases,
    fitnessFunction: mathFitnessFunction
  };
}

/**
 * Create a symbolic regression problem
 * Target: find expression that matches given data points
 */
export function createSymbolicRegressionProblem(
  dataPoints: Array<{ x: number; y: number }>
): ProblemDefinition<number[], number> {
  const testCases: TestCase<number[], number>[] = dataPoints.map(dp => ({
    input: [dp.x],
    expectedOutput: dp.y,
    weight: 1
  }));

  return {
    name: 'Symbolic Regression',
    description: 'Fit a mathematical expression to data points',
    functionSet: MATH_FUNCTIONS.slice(0, 8), // Simpler function set
    terminalSet: createMathTerminals(['x']),
    testCases,
    fitnessFunction: mathFitnessFunction
  };
}
