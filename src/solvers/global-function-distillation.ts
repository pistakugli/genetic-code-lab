export type GlobalFunctionDistillationTerm = {
  normal: number[];
  offset: number;
  readout: number;
};

export type GlobalFunctionDistillationDiagnostics = {
  totalOracleCalls: number;
  sampleCount: number;
  trainingSamples: number;
  validationSamples: number;
  restartsAttempted: number;
  selectedRestart: number;
  trainingMae: number | null;
  validationMae: number | null;
  reconstructionComplete: boolean;
};

export type GlobalFunctionDistillationModel = {
  dimension: number;
  baseIntercept: number;
  baseWeights: number[];
  terms: GlobalFunctionDistillationTerm[];
  diagnostics: GlobalFunctionDistillationDiagnostics;
  reconstructionComplete: boolean;
};

export type GlobalFunctionDistillationConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  termCount: number;
  oracleBudget: number;
  randomSeed?: number;
  sampleCount?: number;
  restarts?: number;
  optimizationSteps?: number;
  batchSize?: number;
  profileOutputLayer?: boolean;
};

export type GlobalFunctionDistillationSample = { input: number[]; output: number };
type Sample = GlobalFunctionDistillationSample;
type HiddenUnit = { normal: number[]; offset: number; readout: number };
type Parameters = {
  baseIntercept: number;
  baseWeights: number[];
  units: HiddenUnit[];
};

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = state * 16807 % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function dot(first: number[], second: number[]): number {
  let result = 0;
  for (let index = 0; index < first.length; index++) result += first[index] * second[index];
  return result;
}

function norm(values: number[]): number {
  return Math.sqrt(dot(values, values));
}

function normalizeUnits(parameters: Parameters, random: () => number): void {
  for (const unit of parameters.units) {
    const length = norm(unit.normal);
    if (length < 1e-10) {
      for (let axis = 0; axis < unit.normal.length; axis++) unit.normal[axis] = 2 * random() - 1;
      const replacementLength = norm(unit.normal);
      for (let axis = 0; axis < unit.normal.length; axis++) {
        unit.normal[axis] /= replacementLength;
      }
      continue;
    }
    for (let axis = 0; axis < unit.normal.length; axis++) unit.normal[axis] /= length;
    unit.offset /= length;
    unit.readout *= length;
  }
}

function validateConfig(config: GlobalFunctionDistillationConfig): void {
  if (!Number.isSafeInteger(config.dimension) || config.dimension < 1) {
    throw new Error('Global function distillation dimension must be a positive safe integer');
  }
  if (config.ranges.length !== config.dimension
    || config.ranges.some(([minimum, maximum]) =>
      !Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= maximum)) {
    throw new Error('Global function distillation ranges must match dimension and be finite');
  }
  if (!Number.isSafeInteger(config.termCount) || config.termCount < 1) {
    throw new Error('Global function distillation termCount must be a positive safe integer');
  }
  if (!Number.isSafeInteger(config.oracleBudget) || config.oracleBudget < 0) {
    throw new Error('Global function distillation oracleBudget must be a non-negative safe integer');
  }
  for (const [name, value, minimum] of [
    ['sampleCount', config.sampleCount, 1],
    ['restarts', config.restarts, 1],
    ['optimizationSteps', config.optimizationSteps, 0],
    ['batchSize', config.batchSize, 1]
  ] as const) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < minimum)) {
      throw new Error(`Global function distillation ${name} must be a safe integer >= ${minimum}`);
    }
  }
}

export function createGlobalFunctionDistillationSampleInputs(
  config: GlobalFunctionDistillationConfig
): number[][] {
  validateConfig(config);
  const random = makeRandom(config.randomSeed ?? 1);
  const sampleCount = Math.min(config.sampleCount ?? 4096, config.oracleBudget);
  return Array.from({ length: sampleCount }, () =>
    config.ranges.map(([minimum, maximum]) => minimum + (maximum - minimum) * random()));
}

function sampleOracle(
  oracle: (input: number[]) => number,
  config: GlobalFunctionDistillationConfig
): Sample[] {
  const inputs = createGlobalFunctionDistillationSampleInputs(config);
  const samples: Sample[] = [];
  for (const input of inputs) {
    const output = oracle(input);
    if (!Number.isFinite(output)) throw new Error('Oracle returned a non-finite value');
    samples.push({ input, output });
  }
  return samples;
}

function splitSamples(samples: Sample[]): { training: Sample[]; validation: Sample[] } {
  if (samples.length < 2) return { training: samples, validation: samples };
  const validationCount = Math.max(1, Math.floor(samples.length * 0.2));
  return {
    training: samples.slice(0, samples.length - validationCount),
    validation: samples.slice(samples.length - validationCount)
  };
}

function initializeParameters(
  dimension: number,
  termCount: number,
  random: () => number,
  outputMean: number
): Parameters {
  const units = Array.from({ length: termCount }, () => {
    const normal = Array.from({ length: dimension }, () => 2 * random() - 1);
    const length = norm(normal);
    return {
      normal: normal.map(value => value / length),
      offset: (2 * random() - 1) * 0.35,
      readout: (2 * random() - 1) * 0.2
    };
  });
  return {
    baseIntercept: outputMean,
    baseWeights: new Array(dimension).fill(0),
    units
  };
}

function features(input: number[], parameters: Parameters): number[] {
  const values = [1, ...input];
  for (const unit of parameters.units) {
    values.push(Math.max(0, dot(unit.normal, input) + unit.offset));
  }
  return values;
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) {
      throw new Error('Global output-layer system is singular');
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let entry = column; entry <= size; entry++) augmented[column][entry] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let entry = column; entry <= size; entry++) {
        augmented[row][entry] -= factor * augmented[column][entry];
      }
    }
  }
  return augmented.map(row => row[size]);
}

function fitOutputLayer(samples: Sample[], parameters: Parameters): void {
  if (samples.length < parameters.baseWeights.length + parameters.units.length + 1) {
    return;
  }
  const featureCount = parameters.baseWeights.length + parameters.units.length + 1;
  const gram = Array.from({ length: featureCount }, () => new Array(featureCount).fill(0));
  const target = new Array(featureCount).fill(0);
  for (const sample of samples) {
    const row = features(sample.input, parameters);
    for (let first = 0; first < featureCount; first++) {
      target[first] += row[first] * sample.output;
      for (let second = first; second < featureCount; second++) {
        gram[first][second] += row[first] * row[second];
      }
    }
  }
  for (let first = 0; first < featureCount; first++) {
    for (let second = 0; second < first; second++) gram[first][second] = gram[second][first];
    gram[first][first] += 1e-8 * Math.max(1, gram[first][first] / samples.length);
  }
  const coefficients = solveLinearSystem(gram, target);
  parameters.baseIntercept = coefficients[0];
  parameters.baseWeights = coefficients.slice(1, 1 + parameters.baseWeights.length);
  parameters.units.forEach((unit, index) => {
    unit.readout = coefficients[1 + parameters.baseWeights.length + index];
  });
}

function predict(input: number[], parameters: Parameters): number {
  let output = parameters.baseIntercept + dot(parameters.baseWeights, input);
  for (const unit of parameters.units) {
    output += unit.readout * Math.max(0, dot(unit.normal, input) + unit.offset);
  }
  return output;
}

function mae(samples: Sample[], parameters: Parameters): number {
  if (!samples.length) return Number.POSITIVE_INFINITY;
  let total = 0;
  for (const sample of samples) total += Math.abs(sample.output - predict(sample.input, parameters));
  return total / samples.length;
}

function shuffledBatch(samples: Sample[], batchSize: number, random: () => number): Sample[] {
  const batch: Sample[] = [];
  for (let index = 0; index < batchSize; index++) {
    batch.push(samples[Math.floor(random() * samples.length)]);
  }
  return batch;
}

type AdamState = { first: number[]; second: number[]; step: number };

function updateAdam(
  values: number[],
  gradients: number[],
  state: AdamState,
  learningRate: number
): void {
  state.step++;
  const firstCorrection = 1 - 0.9 ** state.step;
  const secondCorrection = 1 - 0.999 ** state.step;
  for (let index = 0; index < values.length; index++) {
    state.first[index] = 0.9 * state.first[index] + 0.1 * gradients[index];
    state.second[index] = 0.999 * state.second[index] + 0.001 * gradients[index] ** 2;
    const first = state.first[index] / firstCorrection;
    const second = state.second[index] / secondCorrection;
    values[index] -= learningRate * first / (Math.sqrt(second) + 1e-8);
  }
}

function optimizeRestart(
  training: Sample[],
  config: GlobalFunctionDistillationConfig,
  random: () => number
): Parameters {
  const outputMean = training.reduce((sum, sample) => sum + sample.output, 0) / Math.max(1, training.length);
  const parameters = initializeParameters(config.dimension, config.termCount, random, outputMean);
  const batchSize = Math.min(config.batchSize ?? 256, training.length);
  const steps = config.optimizationSteps ?? 250;
  const profileOutputLayer = config.profileOutputLayer ?? true;
  const hiddenStates = parameters.units.map(unit => ({
    normal: { first: new Array(config.dimension).fill(0), second: new Array(config.dimension).fill(0), step: 0 },
    offset: { first: [0], second: [0], step: 0 }
  }));
  const headValues = [parameters.baseIntercept, ...parameters.baseWeights,
    ...parameters.units.map(unit => unit.readout)];
  const headState: AdamState = {
    first: new Array(headValues.length).fill(0),
    second: new Array(headValues.length).fill(0),
    step: 0
  };
  const headRefitBatchSize = Math.min(training.length, Math.max(256, batchSize * 2));
  for (let step = 0; step < steps; step++) {
    if (profileOutputLayer && step % 8 === 0) {
      fitOutputLayer(shuffledBatch(training, headRefitBatchSize, random), parameters);
    }
    const batch = shuffledBatch(training, batchSize, random);
    const normalGradients = parameters.units.map(() => new Array(config.dimension).fill(0));
    const offsetGradients = new Array(parameters.units.length).fill(0);
    const headGradients = new Array(headValues.length).fill(0);
    for (const sample of batch) {
      const activations = parameters.units.map(unit =>
        dot(unit.normal, sample.input) + unit.offset);
      let output = parameters.baseIntercept + dot(parameters.baseWeights, sample.input);
      for (let unitIndex = 0; unitIndex < parameters.units.length; unitIndex++) {
        output += parameters.units[unitIndex].readout * Math.max(0, activations[unitIndex]);
      }
      const residual = output - sample.output;
      const scale = 2 / batch.length;
      if (!profileOutputLayer) {
        headGradients[0] += scale * residual;
        for (let axis = 0; axis < config.dimension; axis++) {
          headGradients[axis + 1] += scale * residual * sample.input[axis];
        }
        for (let unitIndex = 0; unitIndex < parameters.units.length; unitIndex++) {
          headGradients[1 + config.dimension + unitIndex] +=
            scale * residual * Math.max(0, activations[unitIndex]);
        }
      }
      for (let unitIndex = 0; unitIndex < parameters.units.length; unitIndex++) {
        if (activations[unitIndex] <= 0) continue;
        const factor = scale * residual * parameters.units[unitIndex].readout;
        offsetGradients[unitIndex] += factor;
        for (let axis = 0; axis < config.dimension; axis++) {
          normalGradients[unitIndex][axis] += factor * sample.input[axis];
        }
      }
    }
    for (let unitIndex = 0; unitIndex < parameters.units.length; unitIndex++) {
      updateAdam(parameters.units[unitIndex].normal, normalGradients[unitIndex],
        hiddenStates[unitIndex].normal, 0.015);
      const offset = [parameters.units[unitIndex].offset];
      updateAdam(offset, [offsetGradients[unitIndex]],
        hiddenStates[unitIndex].offset, 0.015);
      parameters.units[unitIndex].offset = offset[0];
    }
    normalizeUnits(parameters, random);
    if (!profileOutputLayer) {
      parameters.units.forEach((unit, index) => {
        headValues[1 + config.dimension + index] = unit.readout;
      });
      updateAdam(headValues, headGradients, headState, 0.02);
      parameters.baseIntercept = headValues[0];
      parameters.baseWeights = headValues.slice(1, 1 + config.dimension);
      parameters.units.forEach((unit, index) => {
        unit.readout = headValues[1 + config.dimension + index];
      });
    }
  }
  if (profileOutputLayer) fitOutputLayer(training, parameters);
  return parameters;
}

export function decompileBlackBoxReluNdGlobalFunctionDistillation(
  oracle: (input: number[]) => number,
  config: GlobalFunctionDistillationConfig
): GlobalFunctionDistillationModel {
  validateConfig(config);
  const random = makeRandom(config.randomSeed ?? 1);
  let totalOracleCalls = 0;
  const samples = sampleOracle(input => {
    totalOracleCalls++;
    return oracle(input);
  }, config);
  const { training, validation } = splitSamples(samples);
  const restartCount = config.restarts ?? 4;
  let bestParameters: Parameters | null = null;
  let bestValidationMae = Number.POSITIVE_INFINITY;
  let bestTrainingMae = Number.POSITIVE_INFINITY;
  let selectedRestart = 0;
  for (let restart = 0; restart < (samples.length > 0 ? restartCount : 0); restart++) {
    const restartRandom = makeRandom((config.randomSeed ?? 1) + 104729 * (restart + 1));
    const parameters = optimizeRestart(training, config, restartRandom);
    const validationMae = mae(validation, parameters);
    if (validationMae < bestValidationMae) {
      bestParameters = parameters;
      bestValidationMae = validationMae;
      bestTrainingMae = mae(training, parameters);
      selectedRestart = restart;
    }
  }
  const parameters = bestParameters ?? initializeParameters(
    config.dimension, config.termCount, random, 0
  );
  const reconstructionComplete = samples.length > 0
    && Number.isFinite(bestValidationMae)
    && bestValidationMae <= 1e-5;
  const diagnostics: GlobalFunctionDistillationDiagnostics = {
    totalOracleCalls,
    sampleCount: samples.length,
    trainingSamples: training.length,
    validationSamples: validation.length,
    restartsAttempted: samples.length > 0 ? restartCount : 0,
    selectedRestart,
    trainingMae: Number.isFinite(bestTrainingMae) ? bestTrainingMae : null,
    validationMae: Number.isFinite(bestValidationMae) ? bestValidationMae : null,
    reconstructionComplete
  };
  return {
    dimension: config.dimension,
    baseIntercept: parameters.baseIntercept,
    baseWeights: parameters.baseWeights,
    terms: parameters.units.map(({ normal, offset, readout }) => ({
      normal: [...normal],
      offset,
      readout
    })),
    diagnostics,
    reconstructionComplete
  };
}

export function predictBlackBoxReluNdGlobalFunctionDistillation(
  model: GlobalFunctionDistillationModel,
  input: number[]
): number {
  if (input.length !== model.dimension) {
    throw new Error('Global function distillation prediction dimension mismatch');
  }
  const parameters: Parameters = {
    baseIntercept: model.baseIntercept,
    baseWeights: model.baseWeights,
    units: model.terms
  };
  return predict(input, parameters);
}
