import { EvolutionEngine } from '../core/evolution';
import {
  EvolutionConfig,
  FunctionDefinition,
  Genome,
  PopulationConfig,
  ProblemDefinition,
  TerminalDefinition
} from '../core/types';
import {
  executeMathGenome,
  genomeToString,
  mathFitnessFunction,
  MATH_FUNCTIONS
} from './math-expression';

export type SymbolicResidualRow = {
  input: number[];
  target: number;
  weight?: number;
};

export type RegressionMetrics = {
  mae: number;
  rmse: number;
  r2: number;
};

export type SymbolicResidualStage = {
  stage: number;
  genome: Genome;
  expression: string;
  coefficient: number;
  fitness: number;
  generationsRun: number;
  trainResidualMaeBefore: number;
  trainResidualMaeAfter: number;
  validationResidualMaeBefore: number;
  validationResidualMaeAfter: number;
  accepted: boolean;
};

export type SymbolicResidualEnsemble = {
  stagesRequested: number;
  stagesAccepted: number;
  shrinkage: number;
  variableNames: string[];
  featureUsage: Record<string, number>;
  expression: string;
  stages: SymbolicResidualStage[];
};

type EvolutionOverrides = Partial<Omit<EvolutionConfig, 'population'>> & {
  population?: Partial<PopulationConfig>;
};

export type SymbolicResidualEnsembleConfig = {
  stages?: number;
  shrinkage?: number;
  seed?: number;
  analyticScaffold?: 'none' | 'tanh-linear' | 'relu-linear';
  validationEvery?: number;
  variableNames?: string[];
  constants?: number[];
  functionNames?: string[];
  functionSet?: FunctionDefinition[];
  terminalSet?: TerminalDefinition[];
  firstStageCoefficient?: 'fixed-one' | 'least-squares';
  deterministicMathRandom?: boolean;
  stopWhenValidationMaeBelow?: number;
  evolution?: EvolutionOverrides;
};

const DEFAULT_CONSTANTS = [-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3];
const DEFAULT_FUNCTION_NAMES = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos'];

function variableName(index: number): string {
  return index === 0 ? 'x' : `x${index}`;
}

function evaluateGenome(genome: Genome, input: number[]): number {
  const scope: Record<string, number> = {};
  for (let i = 0; i < input.length; i++) scope[variableName(i)] = input[i];
  return executeMathGenome(genome, scope);
}

function fitScale(target: number[], predicted: number[], rows: number[]): number {
  let numerator = 0;
  let denominator = 0;
  for (const row of rows) {
    const p = Number.isFinite(predicted[row]) ? predicted[row] : 0;
    numerator += p * target[row];
    denominator += p * p;
  }
  if (Math.abs(denominator) < 1e-12) return 0;
  const scale = numerator / denominator;
  return Number.isFinite(scale) ? scale : 0;
}

function mae(actual: number[], predicted: number[], rows?: number[]): number {
  const indices = rows ?? actual.map((_, index) => index);
  let total = 0;
  for (const row of indices) {
    const pred = Number.isFinite(predicted[row]) ? predicted[row] : 0;
    total += Math.abs(pred - actual[row]);
  }
  return total / Math.max(1, indices.length);
}

function residualSplit(length: number, validationEvery: number): { fit: number[]; validation: number[] } {
  const fit: number[] = [];
  const validation: number[] = [];
  const every = Math.max(2, Math.floor(validationEvery));
  for (let i = 0; i < length; i++) {
    if (i % every === every - 1) validation.push(i);
    else fit.push(i);
  }
  if (validation.length === 0) validation.push(fit.pop() ?? 0);
  return { fit, validation };
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, index) => [...row, rhs[index]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) continue;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => Number.isFinite(row[n]) ? row[n] : 0);
}

function fitLinearAtanh(rows: SymbolicResidualRow[]): number[] {
  const width = rows[0].input.length + 1;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const row of rows) {
    const clamped = Math.max(-0.999999, Math.min(0.999999, row.target));
    const target = 0.5 * Math.log((1 + clamped) / (1 - clamped));
    const x = [1, ...row.input];
    for (let i = 0; i < width; i++) {
      xty[i] += x[i] * target;
      for (let j = 0; j < width; j++) xtx[i][j] += x[i] * x[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += 1e-10;
  return solveLinearSystem(xtx, xty);
}

function fitLinearRows(rows: SymbolicResidualRow[], target: (row: SymbolicResidualRow) => number): number[] {
  const width = rows[0].input.length + 1;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const row of rows) {
    const y = target(row);
    const x = [1, ...row.input];
    for (let i = 0; i < width; i++) {
      xty[i] += x[i] * y;
      for (let j = 0; j < width; j++) xtx[i][j] += x[i] * x[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += 1e-10;
  return solveLinearSystem(xtx, xty);
}

function fitLinearRelu(rows: SymbolicResidualRow[]): number[] {
  const activeRows = rows.filter(row => row.target > 1e-9);
  if (activeRows.length >= rows[0].input.length + 1) return fitLinearRows(activeRows, row => row.target);
  return fitLinearRows(rows, row => row.target);
}

function buildActivationLinearGenome(
  activation: 'tanh' | 'relu',
  coefficients: number[],
  variableNames: string[]
): Genome {
  let id = 0;
  const nextId = (prefix: string) => `${prefix}_${id++}`;
  const genes: Genome['genes'] = [];
  const constGene = (value: number) => {
    const gene = { id: nextId('c'), type: 'terminal' as const, operation: 'const', operands: [value] };
    genes.push(gene);
    return gene.id;
  };
  const variableGene = (name: string) => {
    const gene = { id: nextId('x'), type: 'terminal' as const, operation: name, operands: [] };
    genes.push(gene);
    return gene.id;
  };
  const fnGene = (operation: string, operands: string[]) => {
    const gene = { id: nextId(operation), type: 'function' as const, operation, operands };
    genes.push(gene);
    return gene.id;
  };

  let acc = constGene(coefficients[0] ?? 0);
  for (let i = 0; i < variableNames.length; i++) {
    const term = fnGene('mul', [constGene(coefficients[i + 1] ?? 0), variableGene(variableNames[i])]);
    acc = fnGene('add', [acc, term]);
  }
  const root = activation === 'relu'
    ? fnGene('max', [constGene(0), acc])
    : fnGene('tanh', [acc]);
  const rootGene = genes.find(gene => gene.id === root);
  if (!rootGene) throw new Error(`failed to build ${activation}-linear scaffold genome`);
  return {
    id: `${activation}_linear_${coefficients.map(value => value.toPrecision(6)).join('_')}`,
    genes: [rootGene, ...genes.filter(gene => gene.id !== root)],
    fitness: 0,
    generation: 0,
    age: 0
  };
}

function buildTanhLinearGenome(coefficients: number[], variableNames: string[]): Genome {
  return buildActivationLinearGenome('tanh', coefficients, variableNames);
}

function buildReluLinearGenome(coefficients: number[], variableNames: string[]): Genome {
  return buildActivationLinearGenome('relu', coefficients, variableNames);
}

export async function withDeterministicMathRandom<T>(seed: number, action: () => Promise<T>): Promise<T> {
  const originalRandom = Math.random;
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  Math.random = () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
  try {
    return await action();
  } finally {
    Math.random = originalRandom;
  }
}

export function regressionMetrics(actual: number[], predicted: number[]): RegressionMetrics {
  const n = actual.length;
  const mean = actual.reduce((s, v) => s + v, 0) / n;
  let abs = 0;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const pred = Number.isFinite(predicted[i]) ? predicted[i] : 0;
    const err = pred - actual[i];
    abs += Math.abs(err);
    ssRes += err * err;
    const centered = actual[i] - mean;
    ssTot += centered * centered;
  }
  return {
    mae: abs / n,
    rmse: Math.sqrt(ssRes / n),
    r2: ssTot > 0 ? 1 - ssRes / ssTot : 1
  };
}

export function symbolicFeatureUsage(expressions: string[], variableNames: string[]): Record<string, number> {
  const usage: Record<string, number> = {};
  for (const name of variableNames) usage[name] = 0;
  for (const expression of expressions) {
    for (const name of variableNames) {
      const re = new RegExp(`\\b${name}\\b`, 'g');
      usage[name] += expression.match(re)?.length ?? 0;
    }
  }
  return usage;
}

export function predictSymbolicResidualEnsemble(model: SymbolicResidualEnsemble, input: number[]): number {
  let total = 0;
  for (const stage of model.stages) {
    if (!stage.accepted) continue;
    total += stage.coefficient * evaluateGenome(stage.genome, input);
  }
  return Number.isFinite(total) ? total : 0;
}

export function predictSymbolicResidualBatch(model: SymbolicResidualEnsemble, inputs: number[][]): number[] {
  return inputs.map(input => predictSymbolicResidualEnsemble(model, input));
}

export async function fitSymbolicResidualEnsemble(
  rows: SymbolicResidualRow[],
  config: SymbolicResidualEnsembleConfig = {}
): Promise<SymbolicResidualEnsemble> {
  if (rows.length < 2) throw new Error('symbolic residual ensemble requires at least two rows');
  const inputWidth = rows[0].input.length;
  if (inputWidth < 1) throw new Error('symbolic residual ensemble requires at least one input feature');
  if (rows.some(row => row.input.length !== inputWidth)) throw new Error('all rows must have the same input width');

  const variableNames = config.variableNames ?? Array.from({ length: inputWidth }, (_, index) => variableName(index));
  const constants = config.constants ?? DEFAULT_CONSTANTS;
  const functionSet = config.functionSet ?? MATH_FUNCTIONS.filter(fn => (config.functionNames ?? DEFAULT_FUNCTION_NAMES).includes(fn.name));
  const terminalSet = config.terminalSet ?? [
    ...variableNames.map((name, index) => ({
      name,
      type: 'variable' as const,
      description: `feature ${index}`
    })),
    ...constants.map(value => ({
      name: 'const',
      type: 'constant' as const,
      value,
      description: `constant ${value}`
    }))
  ];
  const residualStagesRequested = Math.max(1, Math.floor(config.stages ?? 1));
  const shrinkage = Math.max(0.05, Math.min(1, config.shrinkage ?? 0.8));
  const seed = config.seed ?? 1337;
  const deterministicMathRandom = config.deterministicMathRandom ?? true;
  const firstStageCoefficient = config.firstStageCoefficient ?? 'fixed-one';
  const split = residualSplit(rows.length, config.validationEvery ?? 5);
  const actual = rows.map(row => row.target);
  const ensemblePred = new Array(rows.length).fill(0);
  const stages: SymbolicResidualStage[] = [];

  if (config.analyticScaffold === 'tanh-linear' || config.analyticScaffold === 'relu-linear') {
    const scaffoldRows = rows.map((row, index) => ({
      input: row.input,
      target: row.target - ensemblePred[index],
      weight: row.weight
    }));
    const scaffoldGenome = config.analyticScaffold === 'relu-linear'
      ? buildReluLinearGenome(fitLinearRelu(scaffoldRows), variableNames)
      : buildTanhLinearGenome(fitLinearAtanh(scaffoldRows), variableNames);
    const scaffoldPred = rows.map(row => evaluateGenome(scaffoldGenome, row.input));
    const coefficient = fitScale(scaffoldRows.map(row => row.target), scaffoldPred, split.fit);
    const beforeFit = mae(actual, ensemblePred, split.fit);
    const beforeValidation = mae(actual, ensemblePred, split.validation);
    const trialPred = ensemblePred.slice();
    for (let i = 0; i < rows.length; i++) trialPred[i] += coefficient * scaffoldPred[i];
    const afterFit = mae(actual, trialPred, split.fit);
    const afterValidation = mae(actual, trialPred, split.validation);
    const accepted = afterValidation < beforeValidation - 1e-5
      && afterFit <= beforeFit + 1e-4
      && Number.isFinite(coefficient)
      && Math.abs(coefficient) > 1e-8;
    const expression = genomeToString(scaffoldGenome);
    stages.push({
      stage: 1,
      genome: scaffoldGenome,
      expression,
      coefficient,
      fitness: 1 / (1 + afterFit),
      generationsRun: 0,
      trainResidualMaeBefore: beforeFit,
      trainResidualMaeAfter: afterFit,
      validationResidualMaeBefore: beforeValidation,
      validationResidualMaeAfter: afterValidation,
      accepted
    });
    if (accepted) {
      for (let i = 0; i < rows.length; i++) ensemblePred[i] = trialPred[i];
    }
  }

  for (let stageIndex = 0; stageIndex < residualStagesRequested; stageIndex++) {
    if (
      config.stopWhenValidationMaeBelow !== undefined
      && mae(actual, ensemblePred, split.validation) <= config.stopWhenValidationMaeBelow
    ) {
      break;
    }
    const stageNumber = stages.length + 1;
    const stageSeed = seed + (stageNumber - 1) * 7919;
    const residualTarget = rows.map((row, index) => row.target - ensemblePred[index]);
    const problem: ProblemDefinition<number[], number> = {
      name: `Symbolic residual stage ${stageNumber}`,
      description: 'Transparent symbolic residual distillation stage',
      functionSet,
      terminalSet,
      testCases: rows.map((row, index) => ({
        input: row.input,
        expectedOutput: residualTarget[index],
        weight: row.weight ?? 1
      })),
      fitnessFunction: mathFitnessFunction
    };
    const evolutionConfig: Partial<EvolutionConfig> = {
      population: {
        size: config.evolution?.population?.size ?? 700,
        eliteCount: config.evolution?.population?.eliteCount ?? 30,
        mutationRate: config.evolution?.population?.mutationRate ?? 0.32,
        crossoverRate: config.evolution?.population?.crossoverRate ?? 0.78,
        maxGenomeDepth: config.evolution?.population?.maxGenomeDepth ?? 8,
        maxGenomeSize: config.evolution?.population?.maxGenomeSize ?? 140,
        tournamentSize: config.evolution?.population?.tournamentSize ?? 6,
        parsimonyCoefficient: config.evolution?.population?.parsimonyCoefficient ?? 0.00015,
        lexicaseRate: config.evolution?.population?.lexicaseRate,
        lexicaseEpsilon: config.evolution?.population?.lexicaseEpsilon,
        quotientMode: config.evolution?.population?.quotientMode
      },
      targetFitness: config.evolution?.targetFitness ?? 0.999,
      maxGenerations: config.evolution?.maxGenerations ?? 140,
      stagnationLimit: config.evolution?.stagnationLimit ?? 60,
      parallelEvaluators: config.evolution?.parallelEvaluators ?? 1,
      selectionMethod: config.evolution?.selectionMethod ?? 'epsilon-lexicase',
      epsilon: config.evolution?.epsilon ?? 0.03,
      seed: stageSeed,
      optimizer: config.evolution?.optimizer ?? {
        enabled: true,
        iterations: 25,
        learningRate: 0.05,
        interval: 5,
        targetCount: 8
      },
      alps: config.evolution?.alps,
      archiveSize: config.evolution?.archiveSize,
      crossoverMethod: config.evolution?.crossoverMethod,
      distributedIslands: config.evolution?.distributedIslands,
      islandMode: config.evolution?.islandMode,
      islandParallelEvaluators: config.evolution?.islandParallelEvaluators,
      multiObjective: config.evolution?.multiObjective,
      solverType: config.evolution?.solverType,
      strictQuotientValidation: config.evolution?.strictQuotientValidation,
      thresholdSelectionPolicy: config.evolution?.thresholdSelectionPolicy
    };

    const engine = new EvolutionEngine(problem, evolutionConfig);
    const evolve = () => engine.evolve();
    const result = deterministicMathRandom
      ? await withDeterministicMathRandom(stageSeed, evolve)
      : await evolve();
    const stagePred = rows.map(row => evaluateGenome(result.bestGenome, row.input));
    const fittedScale = fitScale(residualTarget, stagePred, split.fit);
    const coefficient = stageIndex === 0 && firstStageCoefficient === 'fixed-one'
      ? 1
      : shrinkage * fittedScale;
    const beforeFit = mae(actual, ensemblePred, split.fit);
    const beforeValidation = mae(actual, ensemblePred, split.validation);
    const trialPred = ensemblePred.slice();
    for (let i = 0; i < rows.length; i++) trialPred[i] += coefficient * stagePred[i];
    const afterFit = mae(actual, trialPred, split.fit);
    const afterValidation = mae(actual, trialPred, split.validation);
    const accepted = stages.length === 0 || (
      afterValidation < beforeValidation - 1e-5
      && afterFit <= beforeFit + 1e-4
      && Number.isFinite(coefficient)
      && Math.abs(coefficient) > 1e-8
    );
    const expression = genomeToString(result.bestGenome);
    stages.push({
      stage: stageNumber,
      genome: result.bestGenome,
      expression,
      coefficient,
      fitness: result.bestGenome.fitness,
      generationsRun: result.generationsRun,
      trainResidualMaeBefore: beforeFit,
      trainResidualMaeAfter: afterFit,
      validationResidualMaeBefore: beforeValidation,
      validationResidualMaeAfter: afterValidation,
      accepted
    });
    if (accepted) {
      for (let i = 0; i < rows.length; i++) ensemblePred[i] = trialPred[i];
    }
  }

  const acceptedStages = stages.filter(stage => stage.accepted);
  const expression = acceptedStages.map(stage => `${stage.coefficient.toPrecision(8)} * (${stage.expression})`).join(' + ');
  const featureUsage = symbolicFeatureUsage(acceptedStages.map(stage => stage.expression), variableNames);
  return {
    stagesRequested: residualStagesRequested + (config.analyticScaffold && config.analyticScaffold !== 'none' ? 1 : 0),
    stagesAccepted: acceptedStages.length,
    shrinkage,
    variableNames,
    featureUsage,
    expression,
    stages
  };
}
