import { ProblemDefinition, TestCase } from '../core/types';
import { BOOLEAN_FUNCTIONS, createBooleanTerminals, booleanFitnessFunction } from './boolean-library';

export function createBooleanParityProblem(n: number): ProblemDefinition<number[], number> {
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
    // Odd parity (sum mod 2)
    testCases.push({ input, expectedOutput: ones % 2 === 1 ? 1 : 0, weight: 1.0 });
  }

  const variables = Array.from({ length: n }, (_, i) => `x${i}`);
  return {
    name: `Parity-${n}`,
    description: `Odd parity over ${n} bits`,
    functionSet: BOOLEAN_FUNCTIONS,
    terminalSet: createBooleanTerminals(variables),
    testCases,
    fitnessFunction: booleanFitnessFunction,
    fitnessMode: 'classification'
  };
}
