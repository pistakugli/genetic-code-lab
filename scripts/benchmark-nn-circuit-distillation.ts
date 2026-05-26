#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  fitSymbolicResidualEnsemble,
  predictSymbolicResidualBatch,
  regressionMetrics,
  SymbolicResidualEnsemble
} from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type CircuitRow = {
  input: number[];
  hidden: number[];
  output: number;
};

const repoRoot = path.resolve(__dirname, '..');
const activation = process.env.GCL_NN_CIRCUIT_ACTIVATION === 'relu' ? 'relu' : 'tanh';
const lane = activation === 'relu' ? 'nn-relu-circuit-distillation' : 'nn-circuit-distillation';
const output = resolveBenchmarkOutput(repoRoot, lane, 'smoke');

const readout = {
  weights: [0.85, -0.60, 0.42],
  bias: 0.10
};

function teacherCircuit(input: number[]): CircuitRow {
  const [x0, x1, x2] = input;
  const activate = activation === 'relu'
    ? (value: number) => Math.max(0, value)
    : (value: number) => Math.tanh(value);
  const hidden = [
    activate(1.15 * x0 - 0.75 * x1 + 0.35 * x2 + 0.20),
    activate(-0.55 * x0 + 1.20 * x1 - 0.25 * x2 - 0.15),
    activate(0.35 * x0 + 0.45 * x1 + 1.05 * x2)
  ];
  return {
    input,
    hidden,
    output: readout.bias + hidden.reduce((sum, value, index) => sum + readout.weights[index] * value, 0)
  };
}

function makeRows(): { train: CircuitRow[]; test: CircuitRow[] } {
  const train: CircuitRow[] = [];
  const test: CircuitRow[] = [];
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 9; j++) {
      for (let k = 0; k < 5; k++) {
        train.push(teacherCircuit([
          -1.8 + (3.6 * i) / 8,
          -1.8 + (3.6 * j) / 8,
          -1.4 + (2.8 * k) / 4
        ]));
      }
    }
  }
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      for (let k = 0; k < 4; k++) {
        test.push(teacherCircuit([
          -1.575 + (3.15 * i) / 7,
          -1.575 + (3.15 * j) / 7,
          -1.05 + (2.1 * k) / 3
        ]));
      }
    }
  }
  return { train, test };
}

function predictCircuit(models: SymbolicResidualEnsemble[], inputs: number[][]): { hidden: number[][]; output: number[] } {
  const hidden = inputs.map(input => models.map(model => predictSymbolicResidualBatch(model, [input])[0]));
  const output = hidden.map(row => readout.bias + row.reduce((sum, value, index) => sum + readout.weights[index] * value, 0));
  return { hidden, output };
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const { train, test } = makeRows();
  const stages = Math.max(1, Number(process.env.GCL_NN_CIRCUIT_STAGES ?? 2));
  const population = Math.max(40, Number(process.env.GCL_NN_CIRCUIT_POP ?? 420));
  const generations = Math.max(1, Number(process.env.GCL_NN_CIRCUIT_GEN ?? 100));
  const stagnationLimit = Math.max(1, Number(process.env.GCL_NN_CIRCUIT_STAGNATION ?? 45));
  const seed = Number(process.env.GCL_NN_CIRCUIT_SEED ?? 5151);
  const started = Date.now();

  const models: SymbolicResidualEnsemble[] = [];
  for (let neuron = 0; neuron < 3; neuron++) {
    models.push(await fitSymbolicResidualEnsemble(
      train.map(row => ({ input: row.input, target: row.hidden[neuron] })),
      {
        stages,
        analyticScaffold: activation === 'relu' ? 'relu-linear' : 'tanh-linear',
        shrinkage: 0.75,
        stopWhenValidationMaeBelow: 1e-10,
        seed: seed + neuron * 1009,
        validationEvery: 5,
        variableNames: ['x', 'x1', 'x2'],
        constants: [-2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2],
        functionNames: activation === 'relu'
          ? ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'max']
          : ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tanh'],
        firstStageCoefficient: 'least-squares',
        evolution: {
          population: {
            size: population,
            eliteCount: Math.max(4, Math.floor(population * 0.04)),
            mutationRate: 0.32,
            crossoverRate: 0.78,
            maxGenomeDepth: 7,
            maxGenomeSize: 120,
            tournamentSize: 6,
            parsimonyCoefficient: 0.00015
          },
          targetFitness: 0.9995,
          maxGenerations: generations,
          stagnationLimit,
          parallelEvaluators: 1,
          selectionMethod: 'epsilon-lexicase',
          epsilon: 0.015,
          optimizer: {
            enabled: true,
            iterations: 25,
            learningRate: 0.04,
            interval: 5,
            targetCount: 8
          }
        }
      }
    ));
  }

  const predictions = predictCircuit(models, test.map(row => row.input));
  const hiddenMetrics = models.map((_, neuron) => regressionMetrics(
    test.map(row => row.hidden[neuron]),
    predictions.hidden.map(row => row[neuron])
  ));
  const outputMetrics = regressionMetrics(test.map(row => row.output), predictions.output);
  const maxHiddenMae = Math.max(...hiddenMetrics.map(metric => metric.mae));
  const pass = models.every(model => model.stagesAccepted > 0)
    && maxHiddenMae < 0.03
    && outputMetrics.r2 > 0.995
    && outputMetrics.mae < 0.03;

  const report = {
    generatedAt: new Date().toISOString(),
    task: `${activation} activation-access symbolic circuit distillation`,
    teacher: {
      type: `3-input/3-hidden ${activation} MLP`,
      hiddenActivationsVisible: true,
      readout
    },
    config: { activation, stages, population, generations, stagnationLimit, seed },
    rows: { train: train.length, test: test.length },
    hidden: models.map((model, neuron) => ({
      neuron,
      stagesRequested: model.stagesRequested,
      stagesAccepted: model.stagesAccepted,
      expression: model.expression,
      featureUsage: model.featureUsage,
      metrics: hiddenMetrics[neuron],
      stages: model.stages.map(({ genome, ...stage }) => stage)
    })),
    output: {
      expression: `${readout.bias} + ${readout.weights.map((weight, index) => `${weight} * h${index}`).join(' + ')}`,
      metrics: outputMetrics
    },
    gates: {
      pass,
      maxHiddenMae,
      maxHiddenMaeBelow003: maxHiddenMae < 0.03,
      outputR2Above0995: outputMetrics.r2 > 0.995,
      outputMaeBelow003: outputMetrics.mae < 0.03
    },
    runtimeMs: Date.now() - started
  };
  const md = [
    '# NN Circuit Distillation Benchmark',
    '',
    `- Teacher: fixed 3-input/3-hidden ${activation} MLP`,
    '- GCL access: input-output samples for each hidden neuron plus the output readout',
    `- Train rows: ${train.length}`,
    `- Test rows: ${test.length}`,
    '',
    '| Target | Test MAE | Test RMSE | Test R2 | Stages |',
    '|---|---:|---:|---:|---:|',
    ...hiddenMetrics.map((metric, index) =>
      `| h${index} | ${metric.mae.toFixed(6)} | ${metric.rmse.toFixed(6)} | ${metric.r2.toFixed(6)} | ${models[index].stagesAccepted}/${models[index].stagesRequested} |`
    ),
    `| output via symbolic hidden circuit | ${outputMetrics.mae.toFixed(6)} | ${outputMetrics.rmse.toFixed(6)} | ${outputMetrics.r2.toFixed(6)} | - |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Max hidden MAE < 0.03: ${report.gates.maxHiddenMaeBelow003 ? 'yes' : 'no'} (${maxHiddenMae.toFixed(6)})`,
    `- Output R2 >= 0.995: ${report.gates.outputR2Above0995 ? 'yes' : 'no'} (${outputMetrics.r2.toFixed(6)})`,
    `- Output MAE < 0.03: ${report.gates.outputMaeBelow003 ? 'yes' : 'no'} (${outputMetrics.mae.toFixed(6)})`,
    '',
    '## Hidden Formulas',
    '',
    ...models.flatMap((model, index) => [
      `### h${index}`,
      '',
      '```',
      model.expression,
      '```',
      ''
    ])
  ].join('\n');

  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, md);
  writeBenchmarkManifest(repoRoot, output, {
    lane,
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      maxHiddenMae,
      outputMae: outputMetrics.mae,
      outputR2: outputMetrics.r2,
      acceptedStages: models.map(model => model.stagesAccepted)
    },
    notes: [
      'This benchmark tests symbolic circuit recovery when hidden activations are available.',
      'It is distinct from black-box output-only distillation and does not claim exact weight recovery.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
