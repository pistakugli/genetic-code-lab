import {
  fitSymbolicResidualEnsemble,
  predictSymbolicResidualBatch,
  regressionMetrics
} from '../src/solvers/symbolic-residual-ensemble';

describe('symbolic residual ensemble', () => {
  const rows = Array.from({ length: 31 }, (_, index) => {
    const x = -1.5 + (3 * index) / 30;
    return {
      input: [x],
      target: x * x + 0.5 * x - 0.25
    };
  });

  const config = {
    stages: 1,
    seed: 2026,
    functionNames: ['add', 'sub', 'mul'],
    constants: [-1, -0.5, 0, 0.5, 1, 2],
    evolution: {
      population: {
        size: 40,
        eliteCount: 2,
        mutationRate: 0.2,
        crossoverRate: 0.7,
        maxGenomeDepth: 5,
        maxGenomeSize: 64,
        tournamentSize: 3,
        parsimonyCoefficient: 0
      },
      targetFitness: 0.999999,
      maxGenerations: 2,
      stagnationLimit: 2,
      parallelEvaluators: 1,
      selectionMethod: 'epsilon-lexicase' as const,
      epsilon: 0.01,
      optimizer: {
        enabled: true,
        iterations: 10,
        learningRate: 0.03,
        interval: 1,
        targetCount: 4
      }
    }
  };

  it('distills a simple black-box function into a symbolic predictor', async () => {
    const model = await fitSymbolicResidualEnsemble(rows, config);
    const holdout = [-1.75, -0.9, -0.1, 0.8, 1.75].map(x => ({
      input: [x],
      target: x * x + 0.5 * x - 0.25
    }));
    const predicted = predictSymbolicResidualBatch(model, holdout.map(row => row.input));
    const metrics = regressionMetrics(holdout.map(row => row.target), predicted);

    expect(model.stagesAccepted).toBeGreaterThanOrEqual(1);
    expect(metrics.mae).toBeLessThan(1e-8);
  });

  it('is deterministic for the same seed and data', async () => {
    const first = await fitSymbolicResidualEnsemble(rows, config);
    const second = await fitSymbolicResidualEnsemble(rows, config);

    expect(second.expression).toBe(first.expression);
    expect(second.stages.map(stage => ({
      accepted: stage.accepted,
      coefficient: stage.coefficient,
      expression: stage.expression,
      fitness: stage.fitness
    }))).toEqual(first.stages.map(stage => ({
      accepted: stage.accepted,
      coefficient: stage.coefficient,
      expression: stage.expression,
      fitness: stage.fitness
    })));
  });

  it('uses tanh-linear scaffolding to recover an activation-style neuron', async () => {
    const neuronRows = Array.from({ length: 41 }, (_, index) => {
      const x = -2 + (4 * index) / 40;
      return {
        input: [x],
        target: Math.tanh(1.25 * x - 0.3)
      };
    });
    const model = await fitSymbolicResidualEnsemble(neuronRows, {
      stages: 2,
      analyticScaffold: 'tanh-linear',
      stopWhenValidationMaeBelow: 1e-10,
      seed: 9090,
      functionNames: ['add', 'sub', 'mul', 'tanh'],
      constants: [-1, -0.5, 0, 0.5, 1],
      evolution: {
        population: { size: 40, eliteCount: 2, mutationRate: 0.2, crossoverRate: 0.7, maxGenomeDepth: 5, maxGenomeSize: 64, tournamentSize: 3 },
        maxGenerations: 2,
        stagnationLimit: 2,
        targetFitness: 0.999999,
        parallelEvaluators: 1
      }
    });
    const holdout = [-2.2, -1.1, -0.2, 0.6, 1.8].map(x => ({
      input: [x],
      target: Math.tanh(1.25 * x - 0.3)
    }));
    const predicted = predictSymbolicResidualBatch(model, holdout.map(row => row.input));
    const metrics = regressionMetrics(holdout.map(row => row.target), predicted);

    expect(model.stagesAccepted).toBe(1);
    expect(model.stages[0].generationsRun).toBe(0);
    expect(model.expression).toContain('tanh');
    expect(metrics.mae).toBeLessThan(1e-10);
  });

  it('uses relu-linear scaffolding to recover a piecewise activation neuron', async () => {
    const neuronRows = Array.from({ length: 61 }, (_, index) => {
      const x = -3 + (6 * index) / 60;
      return {
        input: [x],
        target: Math.max(0, 1.4 * x - 0.2)
      };
    });
    const model = await fitSymbolicResidualEnsemble(neuronRows, {
      stages: 2,
      analyticScaffold: 'relu-linear',
      stopWhenValidationMaeBelow: 1e-10,
      seed: 8080,
      functionNames: ['add', 'sub', 'mul', 'max'],
      constants: [-1, -0.5, 0, 0.5, 1],
      evolution: {
        population: { size: 40, eliteCount: 2, mutationRate: 0.2, crossoverRate: 0.7, maxGenomeDepth: 5, maxGenomeSize: 64, tournamentSize: 3 },
        maxGenerations: 2,
        stagnationLimit: 2,
        targetFitness: 0.999999,
        parallelEvaluators: 1
      }
    });
    const holdout = [-2.4, -0.5, 0.25, 1.1, 2.7].map(x => ({
      input: [x],
      target: Math.max(0, 1.4 * x - 0.2)
    }));
    const predicted = predictSymbolicResidualBatch(model, holdout.map(row => row.input));
    const metrics = regressionMetrics(holdout.map(row => row.target), predicted);

    expect(model.stagesAccepted).toBe(1);
    expect(model.stages[0].generationsRun).toBe(0);
    expect(model.expression).toContain('max');
    expect(metrics.mae).toBeLessThan(1e-10);
  });
});
