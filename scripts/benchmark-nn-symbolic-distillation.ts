#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import {
  fitSymbolicResidualEnsemble,
  predictSymbolicResidualBatch,
  regressionMetrics
} from '../src/solvers/symbolic-residual-ensemble';
import {
  resolveBenchmarkOutput,
  writeBenchmarkManifest
} from './benchmark-paths';

type Row = {
  input: number[];
  target: number;
};

type LinearModel = {
  coefficients: number[];
  predict: (input: number[]) => number;
};

const repoRoot = path.resolve(__dirname, '..');
const output = resolveBenchmarkOutput(repoRoot, 'nn-symbolic-distillation', 'smoke');

function teacherNetwork(input: number[]): number {
  const [x0, x1, x2] = input;
  const h0 = Math.tanh(1.15 * x0 - 0.75 * x1 + 0.35 * x2 + 0.20);
  const h1 = Math.tanh(-0.55 * x0 + 1.20 * x1 - 0.25 * x2 - 0.15);
  const h2 = Math.tanh(0.35 * x0 + 0.45 * x1 + 1.05 * x2);
  return 0.85 * h0 - 0.60 * h1 + 0.42 * h2 + 0.18 * x0 * x1 - 0.12 * x1 * x2;
}

function makeRows(): { train: Row[]; test: Row[] } {
  const train: Row[] = [];
  const test: Row[] = [];
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 9; j++) {
      for (let k = 0; k < 5; k++) {
        const input = [
          -1.8 + (3.6 * i) / 8,
          -1.8 + (3.6 * j) / 8,
          -1.4 + (2.8 * k) / 4
        ];
        train.push({ input, target: teacherNetwork(input) });
      }
    }
  }
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      for (let k = 0; k < 4; k++) {
        const input = [
          -1.575 + (3.15 * i) / 7,
          -1.575 + (3.15 * j) / 7,
          -1.05 + (2.1 * k) / 3
        ];
        test.push({ input, target: teacherNetwork(input) });
      }
    }
  }
  return { train, test };
}

function fitLinear(rows: Row[], lambda = 1e-8): LinearModel {
  const width = rows[0].input.length + 1;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const row of rows) {
    const x = [1, ...row.input];
    for (let i = 0; i < width; i++) {
      xty[i] += x[i] * row.target;
      for (let j = 0; j < width; j++) xtx[i][j] += x[i] * x[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += lambda;
  const coefficients = solveLinearSystem(xtx, xty);
  return {
    coefficients,
    predict: (input: number[]) => coefficients[0] + input.reduce((sum, value, index) => sum + coefficients[index + 1] * value, 0)
  };
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, index) => [...row, rhs[index]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error('singular linear baseline matrix');
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const div = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= div;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[n]);
}

async function main(): Promise<void> {
  fs.mkdirSync(output.outputDir, { recursive: true });
  const { train, test } = makeRows();
  const stages = Math.max(1, Number(process.env.GCL_NN_DISTILL_STAGES ?? 4));
  const population = Math.max(40, Number(process.env.GCL_NN_DISTILL_POP ?? 450));
  const generations = Math.max(1, Number(process.env.GCL_NN_DISTILL_GEN ?? 90));
  const stagnationLimit = Math.max(1, Number(process.env.GCL_NN_DISTILL_STAGNATION ?? 40));
  const seed = Number(process.env.GCL_NN_DISTILL_SEED ?? 4242);
  const started = Date.now();

  const linear = fitLinear(train);
  const linearPred = test.map(row => linear.predict(row.input));
  const model = await fitSymbolicResidualEnsemble(train, {
    stages,
    shrinkage: 0.65,
    seed,
    validationEvery: 5,
    variableNames: ['x', 'x1', 'x2'],
    constants: [-2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2],
    functionNames: ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tanh'],
    firstStageCoefficient: 'least-squares',
    evolution: {
      population: {
        size: population,
        eliteCount: Math.max(4, Math.floor(population * 0.04)),
        mutationRate: 0.32,
        crossoverRate: 0.78,
        maxGenomeDepth: 8,
        maxGenomeSize: 150,
        tournamentSize: 6,
        parsimonyCoefficient: 0.0002
      },
      targetFitness: 0.9995,
      maxGenerations: generations,
      stagnationLimit,
      parallelEvaluators: 1,
      selectionMethod: 'epsilon-lexicase',
      epsilon: 0.02,
      optimizer: {
        enabled: true,
        iterations: 25,
        learningRate: 0.04,
        interval: 5,
        targetCount: 8
      }
    }
  });
  const gclPred = predictSymbolicResidualBatch(model, test.map(row => row.input));
  const actual = test.map(row => row.target);
  const linearMetrics = regressionMetrics(actual, linearPred);
  const gclMetrics = regressionMetrics(actual, gclPred);
  const pass = model.stagesAccepted > 0
    && gclMetrics.mae < linearMetrics.mae * 0.8
    && gclMetrics.r2 > 0.85;
  const report = {
    generatedAt: new Date().toISOString(),
    task: 'fixed tanh MLP teacher output distillation',
    blackBox: {
      type: '3-input/3-hidden tanh MLP teacher',
      visibilityToGcl: 'input-output only; weights are not passed to the solver'
    },
    config: { stages, population, generations, stagnationLimit, seed },
    rows: { train: train.length, test: test.length },
    baselines: {
      linear: {
        coefficients: linear.coefficients,
        metrics: linearMetrics
      }
    },
    gcl: {
      stagesRequested: model.stagesRequested,
      stagesAccepted: model.stagesAccepted,
      expression: model.expression,
      featureUsage: model.featureUsage,
      metrics: gclMetrics,
      stages: model.stages.map(({ genome, ...stage }) => stage)
    },
    gates: {
      pass,
      beatsLinearBy20Percent: gclMetrics.mae < linearMetrics.mae * 0.8,
      r2Above085: gclMetrics.r2 > 0.85
    },
    runtimeMs: Date.now() - started
  };
  const md = [
    '# NN Symbolic Distillation Benchmark',
    '',
    '- Teacher: fixed 3-input/3-hidden tanh MLP',
    '- GCL access: input-output samples only',
    `- Train rows: ${train.length}`,
    `- Test rows: ${test.length}`,
    '',
    '| Model | Test MAE | Test RMSE | Test R2 |',
    '|---|---:|---:|---:|',
    `| Linear baseline | ${linearMetrics.mae.toFixed(6)} | ${linearMetrics.rmse.toFixed(6)} | ${linearMetrics.r2.toFixed(6)} |`,
    `| GCL symbolic residual ensemble | ${gclMetrics.mae.toFixed(6)} | ${gclMetrics.rmse.toFixed(6)} | ${gclMetrics.r2.toFixed(6)} |`,
    '',
    '## Gate',
    '',
    `- Pass: ${pass ? 'yes' : 'no'}`,
    `- Accepted stages: ${model.stagesAccepted}/${model.stagesRequested}`,
    `- Beats linear by >=20%: ${report.gates.beatsLinearBy20Percent ? 'yes' : 'no'}`,
    `- R2 >= 0.85: ${report.gates.r2Above085 ? 'yes' : 'no'}`,
    '',
    '## Residual Stages',
    '',
    '| Stage | Status | Coef | Validation residual MAE | Fitness |',
    '|---:|---|---:|---:|---:|',
    ...model.stages.map(stage =>
      `| ${stage.stage} | ${stage.accepted ? 'accepted' : 'rejected'} | ${stage.coefficient.toFixed(6)} | ${stage.validationResidualMaeBefore.toFixed(6)} -> ${stage.validationResidualMaeAfter.toFixed(6)} | ${stage.fitness.toFixed(6)} |`
    ),
    '',
    '## Expression',
    '',
    '```',
    model.expression,
    '```',
    ''
  ].join('\n');

  fs.writeFileSync(output.jsonOutputPath, `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(output.markdownOutputPath, md);
  writeBenchmarkManifest(repoRoot, output, {
    lane: 'nn-symbolic-distillation',
    status: pass ? 'pass' : 'fail',
    summary: {
      pass,
      gclMae: gclMetrics.mae,
      linearMae: linearMetrics.mae,
      gclR2: gclMetrics.r2,
      stagesAccepted: model.stagesAccepted
    },
    notes: [
      'This benchmark tests transparent input-output distillation of a fixed neural teacher.',
      'It does not claim exact weight recovery or superiority over all model classes.'
    ]
  });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
