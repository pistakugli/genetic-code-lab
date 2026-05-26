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
// BOOLEAN FUNCTION SET
// ============================================================================

export const BOOLEAN_FUNCTIONS: FunctionDefinition[] = [
  { name: 'and', arity: 2, implementation: (a, b) => ((a as number) > 0 && (b as number) > 0) ? 1 : 0, returnType: 'number', description: 'AND' },
  { name: 'or',  arity: 2, implementation: (a, b) => ((a as number) > 0 || (b as number) > 0) ? 1 : 0, returnType: 'number', description: 'OR' },
  { name: 'not', arity: 1, implementation: (a)    => ((a as number) > 0) ? 0 : 1, returnType: 'number', description: 'NOT' },
  { name: 'nand',arity: 2, implementation: (a, b) => ((a as number) > 0 && (b as number) > 0) ? 0 : 1, returnType: 'number', description: 'NAND' },
  { name: 'nor', arity: 2, implementation: (a, b) => ((a as number) > 0 || (b as number) > 0) ? 0 : 1, returnType: 'number', description: 'NOR' },
  { name: 'xor', arity: 2, implementation: (a, b) => (((a as number) > 0) !== ((b as number) > 0)) ? 1 : 0, returnType: 'number', description: 'XOR' },
  { name: 'if_then_else', arity: 3, implementation: (cond, t, f) => ((cond as number) > 0) ? (t as number) : (f as number), returnType: 'number', description: 'IF/THEN/ELSE' }
];

export function createBooleanTerminals(variables: string[]): TerminalDefinition[] {
  const terminals: TerminalDefinition[] = [];
  for (const v of variables) {
    terminals.push({ name: v, type: 'variable', description: `Variable ${v}` });
  }
  // Boolean constants
  terminals.push({ name: 'const', type: 'constant', value: 0, description: 'False' });
  terminals.push({ name: 'const', type: 'constant', value: 1, description: 'True' });
  return terminals;
}

export function executeBooleanGenome(genome: Genome, inputs: Record<string, number>): number {
  if (genome.genes.length === 0) return 0;
  const geneMap = new Map<string, Gene>();
  genome.genes.forEach(g => geneMap.set(g.id, g));

  function executeGene(gene: Gene): number {
    if (gene.type === 'terminal') {
      if (gene.operation === 'const' && gene.operands.length > 0) return gene.operands[0] as number;
      return inputs[gene.operation] ?? 0;
    }
    const func = BOOLEAN_FUNCTIONS.find(f => f.name === gene.operation);
    if (!func) return 0;

    const args: number[] = [];
    for (const op of gene.operands) {
      if (typeof op === 'string') {
        const childGene = geneMap.get(op);
        if (childGene) args.push(executeGene(childGene));
      } else if (typeof op === 'number') {
        args.push(op);
      }
    }
    const result = func.implementation(...args);
    return isNaN(result as number) ? 0 : (result as number);
  }
  return executeGene(genome.genes[0]);
}

export const booleanFitnessFunction: FitnessFunction<number[], number> = (
  genome: Genome,
  testCases: TestCase<number[], number>[]
): number => {
  let totalError = 0;
  for (const tc of testCases) {
    const inputs: Record<string, number> = {};
    tc.input.forEach((val, idx) => { inputs[`x${idx}`] = val; });
    const output = executeBooleanGenome(genome, inputs);
    totalError += Math.abs(output - tc.expectedOutput);
  }
  return 1 / (1 + totalError / testCases.length);
};

// ============================================================================
// PROBLEM GENERATORS
// ============================================================================

export function createMajorityProblem(n: number, threshold: number): ProblemDefinition<number[], number> {
  const testCases: TestCase<number[], number>[] = [];
  const total = 1 << n;
  for (let i = 0; i < total; i++) {
    const input: number[] = [];
    let ones = 0;
    for (let bit = 0; bit < n; bit++) {
      const val = (i & (1 << bit)) !== 0 ? 1 : 0;
      input.push(val);
      if (val) ones++;
    }
    testCases.push({ input, expectedOutput: ones >= threshold ? 1 : 0, weight: 1.0 });
  }

  const variables = Array.from({ length: n }, (_, i) => `x${i}`);
  return {
    name: `Majority-${n}`,
    description: `Evaluate 1 if >= ${threshold} bits are 1`,
    functionSet: BOOLEAN_FUNCTIONS,
    terminalSet: createBooleanTerminals(variables),
    testCases,
    fitnessFunction: booleanFitnessFunction,
    fitnessMode: 'classification'
  };
}

export function createMuxProblem(selectBits: number): ProblemDefinition<number[], number> {
  const dataBits = 1 << selectBits;
  const n = selectBits + dataBits;
  const testCases: TestCase<number[], number>[] = [];
  const total = 1 << n;

  for (let i = 0; i < total; i++) {
    const input: number[] = [];
    for (let bit = 0; bit < n; bit++) input.push((i & (1 << bit)) !== 0 ? 1 : 0);
    
    // Select bits are the first `selectBits` inputs
    let sel = 0;
    for (let bit = 0; bit < selectBits; bit++) {
      if (input[bit]) sel |= (1 << bit);
    }
    // Data bits follow select bits
    const expectedOutput = input[selectBits + sel];
    testCases.push({ input, expectedOutput, weight: 1.0 });
  }

  const variables = Array.from({ length: n }, (_, i) => `x${i}`);
  return {
    name: `MUX-${n}`,
    description: `Multiplexer with ${selectBits} select bits`,
    functionSet: BOOLEAN_FUNCTIONS,
    terminalSet: createBooleanTerminals(variables),
    testCases,
    fitnessFunction: booleanFitnessFunction,
    fitnessMode: 'classification'
  };
}
