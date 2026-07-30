export type SwigluFfnStructurePair = {
  valueWeight: number[];
  valueBias: number;
  valueWeightNorm: number;
  gateWeight: number[];
  gateBias: number;
  gateWeightNorm: number;
  outputWeight: number;
};

export type SwigluFfnStructureModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  pairCount: number;
  maxPairCount?: number;
  rawPairCount?: number;
  effectivePairCount?: number;
  activeRank: number;
  gradientProbeCount: number;
  trainingSampleCount: number;
  validationSampleCount: number;
  finiteDifferenceStep: number;
  querySamples: number;
  aggregation: 'mean' | 'median';
  recoveryKind: 'swiglu-ffn-structure' | 'rank-adaptive-swiglu-ffn-structure';
  activeBasis: number[][];
  pairs: SwigluFfnStructurePair[];
  rawPairs?: SwigluFfnStructurePair[];
  outputBias: number;
  trainingMse: number;
  validationMetrics: {
    mae: number;
    rmse: number;
    r2: number;
  };
  optimizer: {
    restarts: number;
    iterations: number;
    learningRate: number;
  };
  queryBudget: {
    rawOracleCalls: number;
    gradientRawOracleCalls: number;
    trainingRawOracleCalls: number;
    validationRawOracleCalls: number;
    finiteDifferenceCallsPerProbe: number;
    aggregatedProbePoints: number;
    querySamples: number;
    selectionRestarts?: number;
    selectedRestart?: number;
  };
  adaptiveSelection?: {
    pairStrengthThreshold: number;
    maxPairStrength: number;
    prunedWeakPairCount: number;
    duplicatePairCount: number;
    selectionRestarts: number;
    candidateCount: number;
    selectedCandidateKind: string;
    selectedCandidateRank: number;
    bestCandidateMetrics: {
      mae: number;
      rmse: number;
      r2: number;
    };
    unprunedValidationMetrics: {
      mae: number;
      rmse: number;
      r2: number;
    };
  };
  expression: string;
};

export type SwigluFfnStructureFinderConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  pairCount: number;
  activeRank?: number;
  gradientProbeCount?: number;
  trainingSampleCount?: number;
  validationSampleCount?: number;
  finiteDifferenceStep?: number;
  randomSeed?: number;
  restarts?: number;
  iterations?: number;
  learningRate?: number;
  ridge?: number;
  querySamples?: number;
  aggregation?: 'mean' | 'median';
};

export type RankAdaptiveSwigluFfnStructureFinderConfig = Omit<SwigluFfnStructureFinderConfig, 'pairCount'> & {
  maxPairCount: number;
  minEffectivePairCount?: number;
  pairRelativeStrengthThreshold?: number;
  pairAbsoluteStrengthThreshold?: number;
  duplicateDirectionCosine?: number;
  selectionRestarts?: number;
  pruneValidationToleranceRatio?: number;
  pruneValidationToleranceAbsolute?: number;
};

type TrainingSample = {
  input: number[];
  coords: number[];
  target: number;
};

type SwigluParams = {
  offset: number;
  outputWeights: number[];
  valueBiases: number[];
  gateBiases: number[];
  valueVectors: number[][];
  gateVectors: number[][];
};

type PairCandidate = {
  restart: number;
  kind: string;
  threshold: number;
  maxStrength: number;
  rawModel: SwigluFfnStructureModel;
  pairs: SwigluFfnStructurePair[];
  prunedWeakPairCount: number;
  duplicatePairCount: number;
  validationMetrics: { mae: number; rmse: number; r2: number };
};

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randomPoint(random: () => number, ranges: Array<[number, number]>, margin: number): number[] {
  return ranges.map(([min, max]) => {
    const boundedMargin = Math.min(margin, (max - min) * 0.2);
    return randomBetween(random, min + boundedMargin, max - boundedMargin);
  });
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function subtractProjection(vector: number[], basis: number[][]): number[] {
  const output = [...vector];
  for (const direction of basis) {
    const coefficient = dot(output, direction);
    for (let index = 0; index < output.length; index++) output[index] -= coefficient * direction[index];
  }
  return output;
}

function canonicalizeDirection(direction: number[]): number[] {
  let maxIndex = 0;
  for (let index = 1; index < direction.length; index++) {
    if (Math.abs(direction[index]) > Math.abs(direction[maxIndex])) maxIndex = index;
  }
  return direction[maxIndex] < 0 ? direction.map(value => -value) : direction;
}

function covarianceVector(gradients: number[][], vector: number[]): number[] {
  const output = new Array(vector.length).fill(0);
  for (const gradient of gradients) {
    const coefficient = dot(gradient, vector);
    for (let index = 0; index < output.length; index++) output[index] += gradient[index] * coefficient;
  }
  return output.map(value => value / Math.max(1, gradients.length));
}

function estimateCoordinateGradient(
  oracle: (input: number[]) => number,
  input: number[],
  step: number
): number[] {
  const gradient: number[] = [];
  for (let axis = 0; axis < input.length; axis++) {
    const left = [...input];
    const right = [...input];
    left[axis] -= step;
    right[axis] += step;
    gradient.push((oracle(right) - oracle(left)) / (2 * step));
  }
  return gradient;
}

function recoverActiveBasis(
  gradients: number[][],
  rank: number,
  random: () => number
): number[][] {
  const dimension = gradients[0]?.length ?? 0;
  const basis: number[][] = [];
  for (let component = 0; component < rank; component++) {
    let vector = normalize(subtractProjection(
      Array.from({ length: dimension }, () => gaussian(random)),
      basis
    ));
    for (let iteration = 0; iteration < 90; iteration++) {
      const multiplied = covarianceVector(gradients, vector);
      const orthogonal = subtractProjection(multiplied, basis);
      const length = norm(orthogonal);
      if (length < 1e-14) break;
      vector = orthogonal.map(value => value / length);
    }
    basis.push(canonicalizeDirection(normalize(subtractProjection(vector, basis))));
  }
  return basis;
}

function projectInput(basis: number[][], input: number[]): number[] {
  return basis.map(direction => dot(direction, input));
}

function silu(value: number): number {
  if (value > 40) return value;
  if (value < -40) return 0;
  return value / (1 + Math.exp(-value));
}

function siluDerivative(value: number): number {
  if (value > 40) return 1;
  if (value < -40) return 0;
  const sigmoid = 1 / (1 + Math.exp(-value));
  return sigmoid * (1 + value * (1 - sigmoid));
}

function aggregate(values: number[], method: 'mean' | 'median'): number {
  if (!values.length) return 0;
  if (method === 'mean') return values.reduce((sum, value) => sum + value, 0) / values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function evaluateParams(params: SwigluParams, coords: number[]): number {
  let value = params.offset;
  for (let pair = 0; pair < params.outputWeights.length; pair++) {
    const preValue = dot(params.valueVectors[pair], coords) + params.valueBiases[pair];
    const preGate = dot(params.gateVectors[pair], coords) + params.gateBiases[pair];
    value += params.outputWeights[pair] * preValue * silu(preGate);
  }
  return Number.isFinite(value) ? value : 0;
}

function makeInitialParams(
  random: () => number,
  pairCount: number,
  activeRank: number,
  targetMean: number,
  targetScale: number,
  restart: number
): SwigluParams {
  const valueVectors: number[][] = [];
  const gateVectors: number[][] = [];
  for (let pair = 0; pair < pairCount; pair++) {
    const valueAxis = (2 * pair + restart) % activeRank;
    const gateAxis = (2 * pair + 1 + Math.floor(restart / 2)) % activeRank;
    valueVectors.push(Array.from({ length: activeRank }, (_unused, axis) => (
      (axis === valueAxis ? 0.9 : 0) + gaussian(random) * 0.18
    )));
    gateVectors.push(Array.from({ length: activeRank }, (_unused, axis) => (
      (axis === gateAxis ? 1.1 : 0) + gaussian(random) * 0.18
    )));
  }
  return {
    offset: targetMean + gaussian(random) * targetScale * 0.04,
    outputWeights: Array.from({ length: pairCount }, () => gaussian(random) * targetScale / Math.max(1, pairCount)),
    valueBiases: Array.from({ length: pairCount }, () => randomBetween(random, -0.35, 0.35)),
    gateBiases: Array.from({ length: pairCount }, () => randomBetween(random, -0.45, 0.45)),
    valueVectors,
    gateVectors
  };
}

function cloneParams(params: SwigluParams): SwigluParams {
  return {
    offset: params.offset,
    outputWeights: [...params.outputWeights],
    valueBiases: [...params.valueBiases],
    gateBiases: [...params.gateBiases],
    valueVectors: params.valueVectors.map(vector => [...vector]),
    gateVectors: params.gateVectors.map(vector => [...vector])
  };
}

function zeroParams(pairCount: number, activeRank: number): SwigluParams {
  return {
    offset: 0,
    outputWeights: new Array(pairCount).fill(0),
    valueBiases: new Array(pairCount).fill(0),
    gateBiases: new Array(pairCount).fill(0),
    valueVectors: Array.from({ length: pairCount }, () => new Array(activeRank).fill(0)),
    gateVectors: Array.from({ length: pairCount }, () => new Array(activeRank).fill(0))
  };
}

function updateAdamScalar(
  params: { offset: number },
  m: { offset: number },
  v: { offset: number },
  key: 'offset',
  grad: number,
  iteration: number,
  learningRate: number,
  beta1: number,
  beta2: number,
  epsilon: number
): void {
  m[key] = beta1 * m[key] + (1 - beta1) * grad;
  v[key] = beta2 * v[key] + (1 - beta2) * grad ** 2;
  const mHat = m[key] / (1 - beta1 ** iteration);
  const vHat = v[key] / (1 - beta2 ** iteration);
  params[key] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
}

function updateAdamArray(
  params: number[],
  m: number[],
  v: number[],
  index: number,
  grad: number,
  iteration: number,
  learningRate: number,
  beta1: number,
  beta2: number,
  epsilon: number
): void {
  m[index] = beta1 * m[index] + (1 - beta1) * grad;
  v[index] = beta2 * v[index] + (1 - beta2) * grad ** 2;
  const mHat = m[index] / (1 - beta1 ** iteration);
  const vHat = v[index] / (1 - beta2 ** iteration);
  params[index] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
}

function trainStructure(
  samples: TrainingSample[],
  pairCount: number,
  activeRank: number,
  random: () => number,
  restarts: number,
  iterations: number,
  learningRate: number,
  ridge: number
): { params: SwigluParams; mse: number } {
  const targetMean = samples.reduce((sum, sample) => sum + sample.target, 0) / Math.max(1, samples.length);
  const targetScale = Math.sqrt(samples.reduce((sum, sample) => sum + (sample.target - targetMean) ** 2, 0) / Math.max(1, samples.length));
  let bestParams = makeInitialParams(random, pairCount, activeRank, targetMean, Math.max(1e-6, targetScale), 0);
  let bestMse = Infinity;

  for (let restart = 0; restart < restarts; restart++) {
    const params = makeInitialParams(random, pairCount, activeRank, targetMean, Math.max(1e-6, targetScale), restart);
    const m = zeroParams(pairCount, activeRank);
    const v = zeroParams(pairCount, activeRank);
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    for (let iteration = 1; iteration <= iterations; iteration++) {
      const grad = zeroParams(pairCount, activeRank);
      let mse = 0;
      for (const sample of samples) {
        const preValues = params.valueVectors.map((vector, pair) => dot(vector, sample.coords) + params.valueBiases[pair]);
        const preGates = params.gateVectors.map((vector, pair) => dot(vector, sample.coords) + params.gateBiases[pair]);
        const gateActivations = preGates.map(silu);
        const gateDerivatives = preGates.map(siluDerivative);
        let predicted = params.offset;
        for (let pair = 0; pair < pairCount; pair++) {
          predicted += params.outputWeights[pair] * preValues[pair] * gateActivations[pair];
        }
        const error = predicted - sample.target;
        mse += error ** 2;
        const scale = 2 * error / Math.max(1, samples.length);
        grad.offset += scale;
        for (let pair = 0; pair < pairCount; pair++) {
          const outputWeight = params.outputWeights[pair];
          grad.outputWeights[pair] += scale * preValues[pair] * gateActivations[pair] + ridge * outputWeight;
          const valueCommon = scale * outputWeight * gateActivations[pair];
          const gateCommon = scale * outputWeight * preValues[pair] * gateDerivatives[pair];
          grad.valueBiases[pair] += valueCommon;
          grad.gateBiases[pair] += gateCommon;
          for (let axis = 0; axis < activeRank; axis++) {
            grad.valueVectors[pair][axis] += valueCommon * sample.coords[axis] + ridge * params.valueVectors[pair][axis];
            grad.gateVectors[pair][axis] += gateCommon * sample.coords[axis] + ridge * params.gateVectors[pair][axis];
          }
        }
      }
      mse /= Math.max(1, samples.length);
      if (mse < bestMse) {
        bestMse = mse;
        bestParams = cloneParams(params);
      }

      const stepScale = learningRate * (0.3 + 0.7 * (1 - iteration / Math.max(1, iterations)));
      updateAdamScalar(params, m, v, 'offset', grad.offset, iteration, stepScale, beta1, beta2, epsilon);
      for (let pair = 0; pair < pairCount; pair++) {
        updateAdamArray(params.outputWeights, m.outputWeights, v.outputWeights, pair, grad.outputWeights[pair], iteration, stepScale, beta1, beta2, epsilon);
        updateAdamArray(params.valueBiases, m.valueBiases, v.valueBiases, pair, grad.valueBiases[pair], iteration, stepScale, beta1, beta2, epsilon);
        updateAdamArray(params.gateBiases, m.gateBiases, v.gateBiases, pair, grad.gateBiases[pair], iteration, stepScale, beta1, beta2, epsilon);
        for (let axis = 0; axis < activeRank; axis++) {
          updateAdamArray(params.valueVectors[pair], m.valueVectors[pair], v.valueVectors[pair], axis, grad.valueVectors[pair][axis], iteration, stepScale, beta1, beta2, epsilon);
          updateAdamArray(params.gateVectors[pair], m.gateVectors[pair], v.gateVectors[pair], axis, grad.gateVectors[pair][axis], iteration, stepScale, beta1, beta2, epsilon);
        }
      }
    }
  }
  return { params: bestParams, mse: bestMse };
}

function metrics(actual: number[], predicted: number[]): { mae: number; rmse: number; r2: number } {
  const count = Math.max(1, actual.length);
  const meanActual = actual.reduce((sum, value) => sum + value, 0) / count;
  let absolute = 0;
  let squared = 0;
  let total = 0;
  for (let index = 0; index < actual.length; index++) {
    const error = actual[index] - predicted[index];
    absolute += Math.abs(error);
    squared += error ** 2;
    total += (actual[index] - meanActual) ** 2;
  }
  return {
    mae: absolute / count,
    rmse: Math.sqrt(squared / count),
    // See the note in deep-blackbox-relu2d-region-graph: a zero-variance target made R2
    // report a perfect 1 no matter what was predicted, which auto-passes the R2 bar.
    r2: total > 0
      ? 1 - squared / total
      : (squared <= 1e-12 * Math.max(meanActual * meanActual * count, 1e-300) ? 1 : 0)
  };
}

function liftToInputWeight(activeBasis: number[][], activeVector: number[], dimension: number): number[] {
  const output = new Array(dimension).fill(0);
  for (let axis = 0; axis < activeVector.length; axis++) {
    const basisVector = activeBasis[axis];
    for (let inputAxis = 0; inputAxis < dimension; inputAxis++) {
      output[inputAxis] += activeVector[axis] * basisVector[inputAxis];
    }
  }
  return output;
}

function pairStrength(pair: SwigluFfnStructurePair): number {
  return Math.abs(pair.outputWeight) * pair.valueWeightNorm * pair.gateWeightNorm;
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(6);
  return value.toPrecision(8).replace(/\.?0+$/, '');
}

function buildExpression(model: Omit<SwigluFfnStructureModel, 'expression'>): string {
  const terms = [`${formatNumber(model.outputBias)}`];
  model.pairs.forEach((pair, index) => {
    terms.push(`${formatNumber(pair.outputWeight)} * (dot(v${index}, x) + ${formatNumber(pair.valueBias)}) * silu(dot(g${index}, x) + ${formatNumber(pair.gateBias)})`);
  });
  return terms.join(' + ');
}

export function decompileSwigluFfnStructure(
  oracle: (input: number[]) => number,
  config: SwigluFfnStructureFinderConfig
): SwigluFfnStructureModel {
  const dimension = Math.max(2, Math.floor(config.dimension));
  if (config.ranges.length !== dimension) throw new Error('SwiGLU FFN structure finder requires one range per dimension');
  if (config.ranges.some(([min, max]) => !(max > min))) throw new Error('SwiGLU FFN structure finder requires valid ranges');
  const pairCount = Math.max(1, Math.floor(config.pairCount));
  const activeRank = Math.max(pairCount * 2, Math.min(dimension, Math.floor(config.activeRank ?? pairCount * 2)));
  const gradientProbeCount = Math.max(activeRank + 2, Math.floor(config.gradientProbeCount ?? Math.max(16, pairCount * 9)));
  const trainingSampleCount = Math.max(pairCount * 900, Math.floor(config.trainingSampleCount ?? Math.max(1800, pairCount * 1200)));
  const validationSampleCount = Math.max(256, Math.floor(config.validationSampleCount ?? 900));
  const minRange = Math.min(...config.ranges.map(([min, max]) => max - min));
  const requestedFiniteDifferenceStep = Math.max(1e-6, Math.min(config.finiteDifferenceStep ?? minRange * 1e-4, minRange * 0.02));
  const finiteDifferenceStep = Math.max(requestedFiniteDifferenceStep, minRange * 0.08);
  const restarts = Math.max(1, Math.floor(config.restarts ?? Math.max(10, pairCount * 6)));
  const iterations = Math.max(80, Math.floor(config.iterations ?? 1000));
  const learningRate = config.learningRate ?? 0.025;
  const ridge = config.ridge ?? 1e-7;
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregation = config.aggregation ?? 'median';
  const random = makeRandom(config.randomSeed ?? 260701);
  let rawOracleCalls = 0;
  let aggregatedProbePoints = 0;
  const countedOracle = (input: number[]): number => {
    aggregatedProbePoints++;
    rawOracleCalls += querySamples;
    if (querySamples === 1) return oracle(input);
    return aggregate(
      Array.from({ length: querySamples }, () => oracle(input)),
      aggregation
    );
  };

  const gradientStart = rawOracleCalls;
  const gradients: number[][] = [];
  for (let index = 0; index < gradientProbeCount; index++) {
    gradients.push(estimateCoordinateGradient(
      countedOracle,
      randomPoint(random, config.ranges, finiteDifferenceStep * 2),
      finiteDifferenceStep
    ));
  }
  const gradientRawOracleCalls = rawOracleCalls - gradientStart;
  const activeBasis = recoverActiveBasis(gradients, activeRank, random);

  const trainingStart = rawOracleCalls;
  const trainingSamples: TrainingSample[] = [];
  for (let index = 0; index < trainingSampleCount; index++) {
    const input = randomPoint(random, config.ranges, 0);
    trainingSamples.push({
      input,
      coords: projectInput(activeBasis, input),
      target: countedOracle(input)
    });
  }
  const trainingRawOracleCalls = rawOracleCalls - trainingStart;

  const { params, mse } = trainStructure(
    trainingSamples,
    pairCount,
    activeRank,
    random,
    restarts,
    iterations,
    learningRate,
    ridge
  );

  const validationStart = rawOracleCalls;
  const validationSamples: TrainingSample[] = [];
  for (let index = 0; index < validationSampleCount; index++) {
    const input = randomPoint(random, config.ranges, 0);
    validationSamples.push({
      input,
      coords: projectInput(activeBasis, input),
      target: countedOracle(input)
    });
  }
  const validationRawOracleCalls = rawOracleCalls - validationStart;
  const validationMetrics = metrics(
    validationSamples.map(sample => sample.target),
    validationSamples.map(sample => evaluateParams(params, sample.coords))
  );

  const pairs = params.outputWeights.map((outputWeight, index) => {
    const valueWeight = liftToInputWeight(activeBasis, params.valueVectors[index], dimension);
    const gateWeight = liftToInputWeight(activeBasis, params.gateVectors[index], dimension);
    return {
      valueWeight,
      valueBias: params.valueBiases[index],
      valueWeightNorm: norm(valueWeight),
      gateWeight,
      gateBias: params.gateBiases[index],
      gateWeightNorm: norm(gateWeight),
      outputWeight
    };
  }).sort((a, b) => pairStrength(b) - pairStrength(a));

  const modelBase = {
    dimension,
    ranges: config.ranges,
    pairCount,
    activeRank,
    gradientProbeCount,
    trainingSampleCount,
    validationSampleCount,
    finiteDifferenceStep,
    querySamples,
    aggregation,
    recoveryKind: 'swiglu-ffn-structure' as const,
    activeBasis,
    pairs,
    outputBias: params.offset,
    trainingMse: mse,
    validationMetrics,
    optimizer: { restarts, iterations, learningRate },
    queryBudget: {
      rawOracleCalls,
      gradientRawOracleCalls,
      trainingRawOracleCalls,
      validationRawOracleCalls,
      finiteDifferenceCallsPerProbe: 2 * dimension,
      aggregatedProbePoints,
      querySamples
    }
  };
  return {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
}

function countDuplicatePairs(
  pairs: SwigluFfnStructurePair[],
  directionCosine: number
): number {
  let duplicates = 0;
  for (let left = 0; left < pairs.length; left++) {
    const leftValueNorm = Math.max(1e-12, norm(pairs[left].valueWeight));
    const leftGateNorm = Math.max(1e-12, norm(pairs[left].gateWeight));
    for (let right = left + 1; right < pairs.length; right++) {
      const rightValueNorm = Math.max(1e-12, norm(pairs[right].valueWeight));
      const rightGateNorm = Math.max(1e-12, norm(pairs[right].gateWeight));
      const valueCosine = Math.abs(dot(pairs[left].valueWeight, pairs[right].valueWeight) / (leftValueNorm * rightValueNorm));
      const gateCosine = Math.abs(dot(pairs[left].gateWeight, pairs[right].gateWeight) / (leftGateNorm * rightGateNorm));
      if (valueCosine >= directionCosine && gateCosine >= directionCosine) duplicates++;
    }
  }
  return duplicates;
}

function pruneWeakPairs(
  pairs: SwigluFfnStructurePair[],
  relativeThreshold: number,
  absoluteThreshold: number,
  minEffectivePairCount: number
): { pairs: SwigluFfnStructurePair[]; threshold: number; maxStrength: number; prunedWeakPairCount: number } {
  const strengths = pairs.map(pairStrength);
  const maxStrength = Math.max(0, ...strengths);
  const threshold = Math.max(absoluteThreshold, maxStrength * relativeThreshold);
  const ranked = pairs
    .map((pair, index) => ({ pair, strength: strengths[index] }))
    .sort((a, b) => b.strength - a.strength);
  const strong = ranked.filter(item => item.strength >= threshold).map(item => item.pair);
  const selected = strong.length >= minEffectivePairCount
    ? strong
    : ranked.slice(0, Math.min(pairs.length, minEffectivePairCount)).map(item => item.pair);
  return {
    pairs: selected,
    threshold,
    maxStrength,
    prunedWeakPairCount: Math.max(0, pairs.length - selected.length)
  };
}

function signaturePairs(pairs: SwigluFfnStructurePair[]): string {
  return pairs
    .map(pair => `${pairStrength(pair).toPrecision(7)}:${pair.valueBias.toPrecision(7)}:${pair.gateBias.toPrecision(7)}:${pair.valueWeightNorm.toPrecision(7)}:${pair.gateWeightNorm.toPrecision(7)}`)
    .sort()
    .join('|');
}

function makeCandidateModel(
  rawModel: SwigluFfnStructureModel,
  maxPairCount: number,
  pairs: SwigluFfnStructurePair[]
): SwigluFfnStructureModel {
  return {
    ...rawModel,
    recoveryKind: 'rank-adaptive-swiglu-ffn-structure',
    pairCount: pairs.length,
    maxPairCount,
    rawPairCount: maxPairCount,
    effectivePairCount: pairs.length,
    pairs,
    rawPairs: rawModel.pairs
  };
}

function generatePairCandidates(
  rawModel: SwigluFfnStructureModel,
  maxPairCount: number,
  minEffectivePairCount: number,
  relativeThreshold: number,
  absoluteThreshold: number,
  duplicateDirectionCosine: number,
  validationInputs: number[][],
  validationActual: number[],
  restart: number
): PairCandidate[] {
  const rawPairs = rawModel.pairs;
  const candidates: Array<{ kind: string; pairs: SwigluFfnStructurePair[]; threshold: number; maxStrength: number; prunedWeakPairCount: number }> = [];
  const thresholdPruned = pruneWeakPairs(rawPairs, relativeThreshold, absoluteThreshold, minEffectivePairCount);
  candidates.push({
    kind: `trained-${rawModel.pairCount}-strength-threshold`,
    pairs: thresholdPruned.pairs,
    threshold: thresholdPruned.threshold,
    maxStrength: thresholdPruned.maxStrength,
    prunedWeakPairCount: thresholdPruned.prunedWeakPairCount
  });

  const ranked = rawPairs
    .map(pair => ({ pair, strength: pairStrength(pair) }))
    .sort((a, b) => b.strength - a.strength);
  const maxStrength = Math.max(0, ...ranked.map(item => item.strength));
  for (let count = minEffectivePairCount; count <= rawPairs.length; count++) {
    candidates.push({
      kind: `trained-${rawModel.pairCount}-top-${count}`,
      pairs: ranked.slice(0, count).map(item => item.pair),
      threshold: ranked[Math.min(count - 1, ranked.length - 1)]?.strength ?? 0,
      maxStrength,
      prunedWeakPairCount: Math.max(0, rawPairs.length - count)
    });
  }

  const seen = new Set<string>();
  const evaluated: PairCandidate[] = [];
  for (const candidate of candidates) {
    if (candidate.pairs.length < minEffectivePairCount) continue;
    const signature = `${candidate.kind}:${signaturePairs(candidate.pairs)}`;
    if (seen.has(signature)) continue;
    seen.add(signature);
    const model = makeCandidateModel(rawModel, maxPairCount, candidate.pairs);
    evaluated.push({
      restart,
      kind: candidate.kind,
      threshold: candidate.threshold,
      maxStrength: candidate.maxStrength,
      rawModel,
      pairs: candidate.pairs,
      prunedWeakPairCount: candidate.prunedWeakPairCount,
      duplicatePairCount: countDuplicatePairs(candidate.pairs, duplicateDirectionCosine),
      validationMetrics: metrics(validationActual, predictSwigluFfnStructureBatch(model, validationInputs))
    });
  }
  return evaluated;
}

export function decompileRankAdaptiveSwigluFfnStructure(
  oracle: (input: number[]) => number,
  config: RankAdaptiveSwigluFfnStructureFinderConfig
): SwigluFfnStructureModel {
  const maxPairCount = Math.max(1, Math.floor(config.maxPairCount));
  const minEffectivePairCount = Math.max(1, Math.floor(config.minEffectivePairCount ?? 1));
  const relativeThreshold = config.pairRelativeStrengthThreshold ?? 0.035;
  const absoluteThreshold = config.pairAbsoluteStrengthThreshold ?? 1e-4;
  const duplicateDirectionCosine = config.duplicateDirectionCosine ?? 0.995;
  const selectionRestarts = Math.max(1, Math.floor(config.selectionRestarts ?? 1));
  const pruneValidationToleranceRatio = config.pruneValidationToleranceRatio ?? 0.15;
  const pruneValidationToleranceAbsolute = config.pruneValidationToleranceAbsolute ?? 0.003;
  const rawModels: SwigluFfnStructureModel[] = [];
  for (let pairCount = minEffectivePairCount; pairCount <= maxPairCount; pairCount++) {
    for (let restart = 0; restart < selectionRestarts; restart++) {
      rawModels.push(decompileSwigluFfnStructure(oracle, {
        ...config,
        pairCount,
        activeRank: config.activeRank,
        randomSeed: (config.randomSeed ?? 260701) + restart * 7919 + pairCount * 104729
      }));
    }
  }

  let adaptiveValidationRawOracleCalls = 0;
  let adaptiveValidationAggregatedProbePoints = 0;
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregation = config.aggregation ?? 'median';
  const adaptiveRandom = makeRandom((config.randomSeed ?? 260701) + 99173);
  const validationCount = Math.max(256, Math.floor(config.validationSampleCount ?? 900));
  const adaptiveValidationInputs = Array.from({ length: validationCount }, () => randomPoint(adaptiveRandom, config.ranges, 0));
  const adaptiveActual = adaptiveValidationInputs.map(input => {
    adaptiveValidationAggregatedProbePoints++;
    adaptiveValidationRawOracleCalls += querySamples;
    if (querySamples === 1) return oracle(input);
    return aggregate(
      Array.from({ length: querySamples }, () => oracle(input)),
      aggregation
    );
  });

  const candidates = rawModels.flatMap((rawModel, restart) => generatePairCandidates(
    rawModel,
    maxPairCount,
    minEffectivePairCount,
    relativeThreshold,
    absoluteThreshold,
    duplicateDirectionCosine,
    adaptiveValidationInputs,
    adaptiveActual,
    restart
  ));
  if (!candidates.length) throw new Error('rank-adaptive SwiGLU structure finder produced no valid pair candidates');

  const bestCandidateMetrics = candidates.reduce((best, candidate) => (
    candidate.validationMetrics.mae < best.mae ? candidate.validationMetrics : best
  ), candidates[0].validationMetrics);
  const maeLimit = Math.max(
    bestCandidateMetrics.mae * (1 + pruneValidationToleranceRatio),
    bestCandidateMetrics.mae + pruneValidationToleranceAbsolute
  );
  const eligible = candidates
    .filter(candidate => candidate.validationMetrics.mae <= maeLimit)
    .sort((a, b) => (
      a.pairs.length - b.pairs.length
      || a.duplicatePairCount - b.duplicatePairCount
      || a.validationMetrics.mae - b.validationMetrics.mae
    ));
  const selected = eligible[0] ?? candidates.sort((a, b) => a.validationMetrics.mae - b.validationMetrics.mae)[0];
  const rawModel = selected.rawModel;
  const budgetTotals = rawModels.reduce((total, model) => ({
    rawOracleCalls: total.rawOracleCalls + model.queryBudget.rawOracleCalls,
    gradientRawOracleCalls: total.gradientRawOracleCalls + model.queryBudget.gradientRawOracleCalls,
    trainingRawOracleCalls: total.trainingRawOracleCalls + model.queryBudget.trainingRawOracleCalls,
    validationRawOracleCalls: total.validationRawOracleCalls + model.queryBudget.validationRawOracleCalls,
    aggregatedProbePoints: total.aggregatedProbePoints + model.queryBudget.aggregatedProbePoints
  }), {
    rawOracleCalls: 0,
    gradientRawOracleCalls: 0,
    trainingRawOracleCalls: 0,
    validationRawOracleCalls: 0,
    aggregatedProbePoints: 0
  });
  const selectedCandidateRank = candidates
    .filter(candidate => candidate.validationMetrics.mae < selected.validationMetrics.mae)
    .length + 1;
  const rawAllPairModel = makeCandidateModel(rawModel, maxPairCount, rawModel.pairs);
  const unprunedValidationMetrics = metrics(
    adaptiveActual,
    predictSwigluFfnStructureBatch(rawAllPairModel, adaptiveValidationInputs)
  );

  const selectedModel: SwigluFfnStructureModel = {
    ...rawModel,
    recoveryKind: 'rank-adaptive-swiglu-ffn-structure',
    pairCount: selected.pairs.length,
    maxPairCount,
    rawPairCount: maxPairCount,
    effectivePairCount: selected.pairs.length,
    pairs: selected.pairs,
    rawPairs: rawModel.pairs,
    queryBudget: {
      ...rawModel.queryBudget,
      rawOracleCalls: budgetTotals.rawOracleCalls + adaptiveValidationRawOracleCalls,
      gradientRawOracleCalls: budgetTotals.gradientRawOracleCalls,
      trainingRawOracleCalls: budgetTotals.trainingRawOracleCalls,
      validationRawOracleCalls: budgetTotals.validationRawOracleCalls + adaptiveValidationRawOracleCalls,
      aggregatedProbePoints: budgetTotals.aggregatedProbePoints + adaptiveValidationAggregatedProbePoints,
      selectionRestarts: rawModels.length,
      selectedRestart: selected.restart
    },
    adaptiveSelection: {
      pairStrengthThreshold: selected.threshold,
      maxPairStrength: selected.maxStrength,
      prunedWeakPairCount: selected.prunedWeakPairCount,
      duplicatePairCount: selected.duplicatePairCount,
      selectionRestarts: rawModels.length,
      candidateCount: candidates.length,
      selectedCandidateKind: selected.kind,
      selectedCandidateRank,
      bestCandidateMetrics,
      unprunedValidationMetrics
    }
  };
  return {
    ...selectedModel,
    validationMetrics: selected.validationMetrics,
    expression: buildExpression(selectedModel)
  };
}

export function predictSwigluFfnStructureModel(model: SwigluFfnStructureModel, input: number[]): number {
  if (input.length !== model.dimension) throw new Error('SwiGLU FFN structure prediction dimension mismatch');
  let value = model.outputBias;
  for (const pair of model.pairs) {
    value += pair.outputWeight
      * (dot(pair.valueWeight, input) + pair.valueBias)
      * silu(dot(pair.gateWeight, input) + pair.gateBias);
  }
  return Number.isFinite(value) ? value : 0;
}

export function predictSwigluFfnStructureBatch(model: SwigluFfnStructureModel, inputs: number[][]): number[] {
  return inputs.map(input => predictSwigluFfnStructureModel(model, input));
}
