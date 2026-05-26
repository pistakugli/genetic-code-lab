import { EvolutionEngine } from '../src/core/evolution';
import { ProblemDefinition } from '../src/core/types';
import {
  MATH_FUNCTIONS,
  executeMathGenome,
  mathFitnessFunction
} from '../src/solvers/math-expression';

function cubic(x: number): number {
  return x * x * x - x + 0.5;
}

function makeCubicProblem(): ProblemDefinition<number[], number> {
  const train = Array.from({ length: 41 }, (_, index) => {
    const x = -2 + (4 * index) / 40;
    return { input: [x], expectedOutput: cubic(x), weight: 1 };
  });

  return {
    name: 'Cubic basis seed regression',
    description: 'x^3 - x + 0.5',
    functionSet: MATH_FUNCTIONS.filter(fn => ['add', 'sub', 'mul', 'div'].includes(fn.name)),
    terminalSet: [
      { name: 'x', type: 'variable', description: 'Variable x' },
      { name: 'const', type: 'constant', value: 1, description: 'One' }
    ],
    testCases: train,
    fitnessFunction: mathFitnessFunction
  };
}

describe('regression basis seeding', () => {
  const previousFlag = process.env.GCL_REGRESSION_BASIS_SEED;
  const previousScaffoldFlag = process.env.GCL_ANALYTIC_SCAFFOLD_SEED;

  afterEach(() => {
    if (previousFlag === undefined) delete process.env.GCL_REGRESSION_BASIS_SEED;
    else process.env.GCL_REGRESSION_BASIS_SEED = previousFlag;
    if (previousScaffoldFlag === undefined) delete process.env.GCL_ANALYTIC_SCAFFOLD_SEED;
    else process.env.GCL_ANALYTIC_SCAFFOLD_SEED = previousScaffoldFlag;
  });

  it('seeds a train-only cubic candidate into the core engine', async () => {
    delete process.env.GCL_REGRESSION_BASIS_SEED;
    const engine = new EvolutionEngine(makeCubicProblem(), {
      population: {
        size: 40,
        eliteCount: 2,
        mutationRate: 0.2,
        crossoverRate: 0.7,
        maxGenomeDepth: 5,
        maxGenomeSize: 64,
        tournamentSize: 3
      },
      targetFitness: 0.999999,
      maxGenerations: 2,
      stagnationLimit: 2,
      parallelEvaluators: 1,
      seed: 123
    });

    const result = await engine.evolve();
    const holdout = [-2.25, -1.37, -0.2, 0.8, 2.25];
    const mae = holdout.reduce((sum, x) => {
      const predicted = executeMathGenome(result.bestGenome, { x });
      return sum + Math.abs(predicted - cubic(x));
    }, 0) / holdout.length;

    expect(result.bestGenome.fitness).toBeGreaterThan(0.9999);
    expect(mae).toBeLessThan(1e-8);
  });

  it('derives multivariable product bases from the problem grammar', async () => {
    const testCases = [];
    for (const x0 of [-2, -1, 0, 1, 2]) {
      for (const x1 of [-1.5, -0.5, 0.5, 1.5]) {
        testCases.push({
          input: [x0, x1],
          expectedOutput: x0 * x1 + 2 * x0 - 0.25,
          weight: 1
        });
      }
    }

    const engine = new EvolutionEngine({
      name: 'Bivariate product basis seed regression',
      description: 'x0*x1 + 2*x0 - 0.25',
      functionSet: MATH_FUNCTIONS.filter(fn => ['add', 'sub', 'mul'].includes(fn.name)),
      terminalSet: [
        { name: 'x', type: 'variable', description: 'Variable x0' },
        { name: 'x1', type: 'variable', description: 'Variable x1' },
        { name: 'const', type: 'constant', value: 1, description: 'One' }
      ],
      testCases,
      fitnessFunction: mathFitnessFunction
    }, {
      population: {
        size: 48,
        eliteCount: 2,
        mutationRate: 0.2,
        crossoverRate: 0.7,
        maxGenomeDepth: 5,
        maxGenomeSize: 80,
        tournamentSize: 3
      },
      targetFitness: 0.999999,
      maxGenerations: 2,
      stagnationLimit: 2,
      parallelEvaluators: 1,
      seed: 456
    });

    const result = await engine.evolve();
    const holdout = [
      [-2.25, 1.25],
      [-0.75, -1.25],
      [0.5, 0.75],
      [1.75, -0.25]
    ];
    const mae = holdout.reduce((sum, input) => {
      const predicted = executeMathGenome(result.bestGenome, { x: input[0], x1: input[1] });
      const expected = input[0] * input[1] + 2 * input[0] - 0.25;
      return sum + Math.abs(predicted - expected);
    }, 0) / holdout.length;

    expect(result.bestGenome.fitness).toBeGreaterThan(0.9999);
    expect(mae).toBeLessThan(1e-8);
  });

  it('derives rational bases instead of relying on polynomial hardcoding', async () => {
    const fn = (x: number) => x / (1 + x * x);
    const testCases = Array.from({ length: 61 }, (_, index) => {
      const x = -3 + (6 * index) / 60;
      return { input: [x], expectedOutput: fn(x), weight: 1 };
    });

    const engine = new EvolutionEngine({
      name: 'Rational basis seed regression',
      description: 'x / (1 + x^2)',
      functionSet: MATH_FUNCTIONS.filter(op => ['add', 'sub', 'mul', 'div'].includes(op.name)),
      terminalSet: [
        { name: 'x', type: 'variable', description: 'Variable x' },
        { name: 'const', type: 'constant', value: 1, description: 'One' }
      ],
      testCases,
      fitnessFunction: mathFitnessFunction
    }, {
      population: {
        size: 48,
        eliteCount: 2,
        mutationRate: 0.2,
        crossoverRate: 0.7,
        maxGenomeDepth: 7,
        maxGenomeSize: 96,
        tournamentSize: 3
      },
      targetFitness: 0.999,
      maxGenerations: 2,
      stagnationLimit: 2,
      parallelEvaluators: 1,
      seed: 789
    });

    const result = await engine.evolve();
    const holdout = [-3.25, -1.75, -0.35, 0.65, 2.4, 3.25];
    const mae = holdout.reduce((sum, x) => {
      const predicted = executeMathGenome(result.bestGenome, { x });
      return sum + Math.abs(predicted - fn(x));
    }, 0) / holdout.length;

    expect(result.bestGenome.fitness).toBeGreaterThan(0.999);
    expect(mae).toBeLessThan(1e-3);
  });

  it('injects pure-GCL analytic rational scaffolds when train-only basis seeding is disabled', async () => {
    process.env.GCL_REGRESSION_BASIS_SEED = '0';
    delete process.env.GCL_ANALYTIC_SCAFFOLD_SEED;
    const fn = (x: number) => 1 / (1 + x * x);
    const testCases = Array.from({ length: 61 }, (_, index) => {
      const x = -3 + (6 * index) / 60;
      return { input: [x], expectedOutput: fn(x), weight: 1 };
    });

    const engine = new EvolutionEngine({
      name: 'Pure GCL analytic inverse rational scaffold',
      description: '1 / (1 + x^2)',
      functionSet: MATH_FUNCTIONS.filter(op => ['add', 'sub', 'mul', 'div'].includes(op.name)),
      terminalSet: [
        { name: 'x', type: 'variable', description: 'Variable x' },
        { name: 'const', type: 'constant', value: 1, description: 'One' }
      ],
      testCases,
      fitnessFunction: mathFitnessFunction
    }, {
      population: {
        size: 32,
        eliteCount: 2,
        mutationRate: 0.2,
        crossoverRate: 0.7,
        maxGenomeDepth: 5,
        maxGenomeSize: 64,
        tournamentSize: 3
      },
      targetFitness: 0.999999,
      maxGenerations: 1,
      stagnationLimit: 1,
      parallelEvaluators: 1,
      seed: 321
    });

    const result = await engine.evolve();
    const holdout = [-3.25, -1.75, -0.35, 0.65, 2.4, 3.25];
    const mae = holdout.reduce((sum, x) => {
      const predicted = executeMathGenome(result.bestGenome, { x });
      return sum + Math.abs(predicted - fn(x));
    }, 0) / holdout.length;

    expect(result.bestGenome.fitness).toBeGreaterThan(0.9999);
    expect(mae).toBeLessThan(1e-10);
  });

  it('injects pure-GCL analytic trig-rational scaffolds when train-only basis seeding is disabled', async () => {
    process.env.GCL_REGRESSION_BASIS_SEED = '0';
    delete process.env.GCL_ANALYTIC_SCAFFOLD_SEED;
    const fn = (x: number) => Math.sin(x) + x / (1 + x * x);
    const testCases = Array.from({ length: 71 }, (_, index) => {
      const x = -3 + (6 * index) / 70;
      return { input: [x], expectedOutput: fn(x), weight: 1 };
    });

    const engine = new EvolutionEngine({
      name: 'Pure GCL analytic trig-rational scaffold',
      description: 'sin(x) + x / (1 + x^2)',
      functionSet: MATH_FUNCTIONS.filter(op => ['add', 'sub', 'mul', 'div', 'sin', 'cos'].includes(op.name)),
      terminalSet: [
        { name: 'x', type: 'variable', description: 'Variable x' },
        { name: 'const', type: 'constant', value: 1, description: 'One' }
      ],
      testCases,
      fitnessFunction: mathFitnessFunction
    }, {
      population: {
        size: 40,
        eliteCount: 2,
        mutationRate: 0.2,
        crossoverRate: 0.7,
        maxGenomeDepth: 7,
        maxGenomeSize: 96,
        tournamentSize: 3
      },
      targetFitness: 0.999999,
      maxGenerations: 1,
      stagnationLimit: 1,
      parallelEvaluators: 1,
      seed: 654
    });

    const result = await engine.evolve();
    const holdout = [-3.25, -1.75, -0.35, 0.65, 2.4, 3.25];
    const mae = holdout.reduce((sum, x) => {
      const predicted = executeMathGenome(result.bestGenome, { x });
      return sum + Math.abs(predicted - fn(x));
    }, 0) / holdout.length;

    expect(result.bestGenome.fitness).toBeGreaterThan(0.9999);
    expect(mae).toBeLessThan(1e-10);
  });
});
