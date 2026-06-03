export type GeluFfnStructureAtom = {
  inputWeight: number[];
  inputBias: number;
  outputWeight: number;
  inputWeightNorm: number;
};

export type GeluFfnStructureModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  channelCount: number;
  maxChannelCount?: number;
  rawChannelCount?: number;
  effectiveChannelCount?: number;
  activeRank: number;
  gradientProbeCount: number;
  trainingSampleCount: number;
  gradientTrainingSampleCount?: number;
  validationSampleCount: number;
  finiteDifferenceStep: number;
  querySamples: number;
  aggregation: 'mean' | 'median';
  recoveryKind: 'gelu-ffn-structure' | 'rank-adaptive-gelu-ffn-structure';
  activeBasis: number[][];
  atoms: GeluFfnStructureAtom[];
  rawAtoms?: GeluFfnStructureAtom[];
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
    scalarRefinementRestarts?: number;
    scalarRefinementIterations?: number;
    scalarRefinementLearningRate?: number;
    gradientTrainingWeight?: number;
  };
	  queryBudget: {
	    rawOracleCalls: number;
	    gradientRawOracleCalls: number;
	    trainingRawOracleCalls: number;
	    gradientTrainingRawOracleCalls?: number;
	    validationRawOracleCalls: number;
	    finiteDifferenceCallsPerProbe: number;
	    aggregatedProbePoints: number;
	    querySamples: number;
	    selectionRestarts?: number;
	    selectedRestart?: number;
	  };
	  adaptiveSelection?: {
	    atomStrengthThreshold: number;
	    maxAtomStrength: number;
	    prunedWeakAtomCount: number;
	    duplicatePairCount: number;
	    nearParallelPairCount: number;
	    duplicateDirectionCosine: number;
	    duplicateBiasTolerance: number;
	    nearParallelDirectionCosine: number;
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

export type GeluFfnCanonicalAtom = {
  direction: number[];
  inputWeightNorm: number;
  inputBias: number;
  outputWeight: number;
  strength: number;
};

export type GeluFfnCanonicalSignature = {
  atoms: GeluFfnCanonicalAtom[];
  canonicalAtomCount: number;
  removedWeakAtomCount: number;
  mergedDuplicateCount: number;
  removedCancellationCount: number;
};

export type GeluFfnCanonicalSignatureConfig = {
  atomRelativeStrengthThreshold?: number;
  atomAbsoluteStrengthThreshold?: number;
  duplicateDirectionCosine?: number;
  duplicateBiasTolerance?: number;
  duplicateNormTolerance?: number;
  cancellationOutputTolerance?: number;
};

export type GeluFfnCanonicalSignatureComparison = {
  atomCountDelta: number;
  signatureDistance: number;
  maxAtomDistance: number;
  meanAtomDistance: number;
  matchedPairs: Array<{
    left: number;
    right: number;
    distance: number;
  }>;
};

export type GeluFfnStructureFinderConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  channelCount: number;
  activeRank?: number;
  gradientProbeCount?: number;
  trainingSampleCount?: number;
  gradientTrainingSampleCount?: number;
  gradientTrainingWeight?: number;
  activeGradientFiniteDifferenceStep?: number;
  validationSampleCount?: number;
  finiteDifferenceStep?: number;
  randomSeed?: number;
  restarts?: number;
  iterations?: number;
  learningRate?: number;
  scalarRefinementRestarts?: number;
  scalarRefinementIterations?: number;
  scalarRefinementLearningRate?: number;
  ridge?: number;
  querySamples?: number;
  aggregation?: 'mean' | 'median';
};

export type RankAdaptiveGeluFfnStructureFinderConfig = Omit<GeluFfnStructureFinderConfig, 'channelCount'> & {
  maxChannelCount: number;
  minEffectiveChannelCount?: number;
  candidateChannelCounts?: number[];
  atomRelativeStrengthThreshold?: number;
	  atomAbsoluteStrengthThreshold?: number;
	  duplicateDirectionCosine?: number;
	  duplicateBiasTolerance?: number;
	  nearParallelDirectionCosine?: number;
	  selectionRestarts?: number;
	  pruneValidationToleranceRatio?: number;
	  pruneValidationToleranceAbsolute?: number;
	};

type TrainingSample = {
  input: number[];
  coords: number[];
  target: number;
};

type GradientTrainingSample = {
  coords: number[];
  targetGradient: number[];
};

type StructureParams = {
  offset: number;
  outputWeights: number[];
  biases: number[];
  vectors: number[][];
};

type FixedDirectionScalarParams = {
  offset: number;
  outputWeights: number[];
  biases: number[];
  logScales: number[];
  directions: number[][];
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

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function geluDerivative(value: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const inner = c * (value + 0.044715 * value ** 3);
  const tanh = Math.tanh(inner);
  const sech2 = 1 - tanh ** 2;
  return 0.5 * (1 + tanh) + 0.5 * value * sech2 * c * (1 + 3 * 0.044715 * value ** 2);
}

function geluSecondDerivative(value: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const cubic = 0.044715;
  const inner = c * (value + cubic * value ** 3);
  const tanh = Math.tanh(inner);
  const sech2 = 1 - tanh ** 2;
  const first = c * (1 + 3 * cubic * value ** 2);
  const second = c * 6 * cubic * value;
  return sech2 * (first + 0.5 * value * (second - 2 * tanh * first ** 2));
}

function aggregate(values: number[], method: 'mean' | 'median'): number {
  if (!values.length) return 0;
  if (method === 'mean') return values.reduce((sum, value) => sum + value, 0) / values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function evaluateParams(params: StructureParams, coords: number[]): number {
  let value = params.offset;
  for (let atom = 0; atom < params.outputWeights.length; atom++) {
    value += params.outputWeights[atom] * gelu(dot(params.vectors[atom], coords) + params.biases[atom]);
  }
  return Number.isFinite(value) ? value : 0;
}

function estimateActiveBasisGradient(
  oracle: (input: number[]) => number,
  input: number[],
  activeBasis: number[][],
  step: number
): number[] {
  return activeBasis.map(direction => {
    const left = input.map((value, index) => value - step * direction[index]);
    const right = input.map((value, index) => value + step * direction[index]);
    return (oracle(right) - oracle(left)) / (2 * step);
  });
}

function makeInitialParams(
  random: () => number,
  channelCount: number,
  activeRank: number,
  targetMean: number,
  targetScale: number,
  restart: number
): StructureParams {
  if (channelCount === 1 && activeRank === 1 && restart < 8) {
    const sign = restart % 2 === 0 ? 1 : -1;
    const scale = [0.75, 1.0, 1.35, 1.75][Math.floor(restart / 2)] ?? 1;
    return {
      offset: targetMean,
      outputWeights: [Math.max(0.1, targetScale)],
      biases: [0],
      vectors: [[sign * scale]]
    };
  }
  return {
    offset: targetMean + gaussian(random) * targetScale * 0.05,
    outputWeights: Array.from({ length: channelCount }, () => gaussian(random) * targetScale / Math.max(1, channelCount)),
    biases: Array.from({ length: channelCount }, () => randomBetween(random, -0.7, 0.7)),
    vectors: Array.from({ length: channelCount }, (_, atom) => (
      Array.from({ length: activeRank }, (_unused, axis) => {
        const axisSeed = axis === (atom + restart) % activeRank ? 1 : 0;
        return axisSeed + gaussian(random) * 0.25;
      })
    ))
  };
}

function cloneParams(params: StructureParams): StructureParams {
  return {
    offset: params.offset,
    outputWeights: [...params.outputWeights],
    biases: [...params.biases],
    vectors: params.vectors.map(vector => [...vector])
  };
}

function trainStructure(
  samples: TrainingSample[],
  gradientSamples: GradientTrainingSample[],
  channelCount: number,
  activeRank: number,
  random: () => number,
  restarts: number,
  iterations: number,
  learningRate: number,
  gradientTrainingWeight: number,
  ridge: number
): { params: StructureParams; mse: number } {
  const targetMean = samples.reduce((sum, sample) => sum + sample.target, 0) / Math.max(1, samples.length);
  const targetScale = Math.sqrt(samples.reduce((sum, sample) => sum + (sample.target - targetMean) ** 2, 0) / Math.max(1, samples.length));
  let bestParams = makeInitialParams(random, channelCount, activeRank, targetMean, Math.max(1e-6, targetScale), 0);
  let bestMse = Infinity;

  for (let restart = 0; restart < restarts; restart++) {
    const params = makeInitialParams(random, channelCount, activeRank, targetMean, Math.max(1e-6, targetScale), restart);
    const m = {
      offset: 0,
      outputWeights: new Array(channelCount).fill(0),
      biases: new Array(channelCount).fill(0),
      vectors: Array.from({ length: channelCount }, () => new Array(activeRank).fill(0))
    };
    const v = cloneParams(m);
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    for (let iteration = 1; iteration <= iterations; iteration++) {
      const grad = {
        offset: 0,
        outputWeights: new Array(channelCount).fill(0),
        biases: new Array(channelCount).fill(0),
        vectors: Array.from({ length: channelCount }, () => new Array(activeRank).fill(0))
      };
      let mse = 0;
      for (const sample of samples) {
        const pre = params.vectors.map((vector, atom) => dot(vector, sample.coords) + params.biases[atom]);
        const activations = pre.map(gelu);
        const derivatives = pre.map(geluDerivative);
        let predicted = params.offset;
        for (let atom = 0; atom < channelCount; atom++) predicted += params.outputWeights[atom] * activations[atom];
        const error = predicted - sample.target;
        mse += error ** 2;
        const scale = 2 * error / Math.max(1, samples.length);
        grad.offset += scale;
        for (let atom = 0; atom < channelCount; atom++) {
          grad.outputWeights[atom] += scale * activations[atom] + ridge * params.outputWeights[atom];
          const common = scale * params.outputWeights[atom] * derivatives[atom];
          grad.biases[atom] += common;
          for (let axis = 0; axis < activeRank; axis++) {
            grad.vectors[atom][axis] += common * sample.coords[axis] + ridge * params.vectors[atom][axis];
          }
        }
      }
      if (gradientTrainingWeight > 0 && gradientSamples.length) {
        const gradientScale = 2 * gradientTrainingWeight / Math.max(1, gradientSamples.length * activeRank);
        for (const sample of gradientSamples) {
          const pre = params.vectors.map((vector, atom) => dot(vector, sample.coords) + params.biases[atom]);
          const derivatives = pre.map(geluDerivative);
          const secondDerivatives = pre.map(geluSecondDerivative);
          const predictedGradient = new Array(activeRank).fill(0);
          for (let atom = 0; atom < channelCount; atom++) {
            const outputWeight = params.outputWeights[atom];
            const derivative = derivatives[atom];
            for (let axis = 0; axis < activeRank; axis++) {
              predictedGradient[axis] += outputWeight * derivative * params.vectors[atom][axis];
            }
          }
          const gradientErrors = predictedGradient.map((value, axis) => value - sample.targetGradient[axis]);
          for (let atom = 0; atom < channelCount; atom++) {
            const vector = params.vectors[atom];
            const outputWeight = params.outputWeights[atom];
            const derivative = derivatives[atom];
            const secondDerivative = secondDerivatives[atom];
            const errorDotVector = dot(gradientErrors, vector);
            grad.outputWeights[atom] += gradientScale * derivative * errorDotVector;
            grad.biases[atom] += gradientScale * outputWeight * secondDerivative * errorDotVector;
            for (let axis = 0; axis < activeRank; axis++) {
              grad.vectors[atom][axis] += gradientScale * outputWeight * (
                secondDerivative * sample.coords[axis] * errorDotVector
                + derivative * gradientErrors[axis]
              );
            }
          }
        }
      }
      mse /= Math.max(1, samples.length);
      if (mse < bestMse) {
        bestMse = mse;
        bestParams = cloneParams(params);
      }

      const stepScale = learningRate * (0.35 + 0.65 * (1 - iteration / Math.max(1, iterations)));
      updateAdamScalar(params, m, v, 'offset', grad.offset, iteration, stepScale, beta1, beta2, epsilon);
      for (let atom = 0; atom < channelCount; atom++) {
        updateAdamArray(params.outputWeights, m.outputWeights, v.outputWeights, atom, grad.outputWeights[atom], iteration, stepScale, beta1, beta2, epsilon);
        updateAdamArray(params.biases, m.biases, v.biases, atom, grad.biases[atom], iteration, stepScale, beta1, beta2, epsilon);
        for (let axis = 0; axis < activeRank; axis++) {
          updateAdamArray(params.vectors[atom], m.vectors[atom], v.vectors[atom], axis, grad.vectors[atom][axis], iteration, stepScale, beta1, beta2, epsilon);
        }
      }
    }
  }
  return { params: bestParams, mse: bestMse };
}

function makeFixedDirectionScalarParams(
  base: StructureParams,
  activeRank: number,
  random: () => number,
  restart: number
): FixedDirectionScalarParams {
  const directions: number[][] = [];
  const logScales: number[] = [];
  for (const vector of base.vectors) {
    const length = norm(vector);
    const direction = length > 1e-10
      ? vector.map(value => value / length)
      : normalize(Array.from({ length: activeRank }, () => gaussian(random)));
    directions.push(direction);
    logScales.push(Math.log(Math.max(0.03, length)) + (restart === 0 ? 0 : gaussian(random) * 0.25));
  }
  return {
    offset: base.offset + (restart === 0 ? 0 : gaussian(random) * 0.01),
    outputWeights: base.outputWeights.map(value => value * (restart === 0 ? 1 : 1 + gaussian(random) * 0.12)),
    biases: base.biases.map(value => value + (restart === 0 ? 0 : gaussian(random) * 0.08)),
    logScales,
    directions
  };
}

function cloneFixedDirectionScalarParams(params: FixedDirectionScalarParams): FixedDirectionScalarParams {
  return {
    offset: params.offset,
    outputWeights: [...params.outputWeights],
    biases: [...params.biases],
    logScales: [...params.logScales],
    directions: params.directions.map(direction => [...direction])
  };
}

function evaluateFixedDirectionScalarParams(params: FixedDirectionScalarParams, coords: number[]): number {
  let value = params.offset;
  for (let atom = 0; atom < params.outputWeights.length; atom++) {
    const scale = Math.exp(params.logScales[atom]);
    value += params.outputWeights[atom] * gelu(scale * dot(params.directions[atom], coords) + params.biases[atom]);
  }
  return Number.isFinite(value) ? value : 0;
}

function scalarParamsToStructureParams(params: FixedDirectionScalarParams): StructureParams {
  return {
    offset: params.offset,
    outputWeights: [...params.outputWeights],
    biases: [...params.biases],
    vectors: params.directions.map((direction, atom) => {
      const scale = Math.exp(params.logScales[atom]);
      return direction.map(value => value * scale);
    })
  };
}

function trainFixedDirectionScalars(
  samples: TrainingSample[],
  gradientSamples: GradientTrainingSample[],
  base: StructureParams,
  activeRank: number,
  random: () => number,
  restarts: number,
  iterations: number,
  learningRate: number,
  gradientTrainingWeight: number,
  ridge: number
): { params: StructureParams; mse: number } {
  const channelCount = base.outputWeights.length;
  let bestParams = makeFixedDirectionScalarParams(base, activeRank, random, 0);
  let bestMse = Infinity;

  for (let restart = 0; restart < restarts; restart++) {
    const params = makeFixedDirectionScalarParams(base, activeRank, random, restart);
    const m = {
      offset: 0,
      outputWeights: new Array(channelCount).fill(0),
      biases: new Array(channelCount).fill(0),
      logScales: new Array(channelCount).fill(0)
    };
    const v = {
      offset: 0,
      outputWeights: new Array(channelCount).fill(0),
      biases: new Array(channelCount).fill(0),
      logScales: new Array(channelCount).fill(0)
    };
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    for (let iteration = 1; iteration <= iterations; iteration++) {
      const grad = {
        offset: 0,
        outputWeights: new Array(channelCount).fill(0),
        biases: new Array(channelCount).fill(0),
        logScales: new Array(channelCount).fill(0)
      };
      let mse = 0;
      for (const sample of samples) {
        const projections = params.directions.map(direction => dot(direction, sample.coords));
        const scales = params.logScales.map(Math.exp);
        const pre = projections.map((projection, atom) => scales[atom] * projection + params.biases[atom]);
        const activations = pre.map(gelu);
        const derivatives = pre.map(geluDerivative);
        let predicted = params.offset;
        for (let atom = 0; atom < channelCount; atom++) predicted += params.outputWeights[atom] * activations[atom];
        const error = predicted - sample.target;
        mse += error ** 2;
        const sampleScale = 2 * error / Math.max(1, samples.length);
        grad.offset += sampleScale;
        for (let atom = 0; atom < channelCount; atom++) {
          grad.outputWeights[atom] += sampleScale * activations[atom] + ridge * params.outputWeights[atom];
          const common = sampleScale * params.outputWeights[atom] * derivatives[atom];
          grad.biases[atom] += common;
          grad.logScales[atom] += common * scales[atom] * projections[atom] + ridge * params.logScales[atom] * 0.1;
        }
      }
      if (gradientTrainingWeight > 0 && gradientSamples.length) {
        const gradientScale = 2 * gradientTrainingWeight / Math.max(1, gradientSamples.length * activeRank);
        for (const sample of gradientSamples) {
          const projections = params.directions.map(direction => dot(direction, sample.coords));
          const scales = params.logScales.map(Math.exp);
          const pre = projections.map((projection, atom) => scales[atom] * projection + params.biases[atom]);
          const derivatives = pre.map(geluDerivative);
          const secondDerivatives = pre.map(geluSecondDerivative);
          const predictedGradient = new Array(activeRank).fill(0);
          for (let atom = 0; atom < channelCount; atom++) {
            const outputWeight = params.outputWeights[atom];
            const derivative = derivatives[atom];
            const scale = scales[atom];
            for (let axis = 0; axis < activeRank; axis++) {
              predictedGradient[axis] += outputWeight * derivative * scale * params.directions[atom][axis];
            }
          }
          const gradientErrors = predictedGradient.map((value, axis) => value - sample.targetGradient[axis]);
          for (let atom = 0; atom < channelCount; atom++) {
            const scale = scales[atom];
            const vector = params.directions[atom].map(value => value * scale);
            const errorDotVector = dot(gradientErrors, vector);
            const derivative = derivatives[atom];
            const secondDerivative = secondDerivatives[atom];
            grad.outputWeights[atom] += gradientScale * derivative * errorDotVector;
            grad.biases[atom] += gradientScale * params.outputWeights[atom] * secondDerivative * errorDotVector;
            grad.logScales[atom] += gradientScale * params.outputWeights[atom] * (
              derivative * errorDotVector
              + secondDerivative * scale * projections[atom] * errorDotVector
            );
          }
        }
      }
      mse /= Math.max(1, samples.length);
      if (mse < bestMse) {
        bestMse = mse;
        bestParams = cloneFixedDirectionScalarParams(params);
      }

      const stepScale = learningRate * (0.25 + 0.75 * (1 - iteration / Math.max(1, iterations)));
      updateAdamScalar(params, m, v, 'offset', grad.offset, iteration, stepScale, beta1, beta2, epsilon);
      for (let atom = 0; atom < channelCount; atom++) {
        updateAdamArray(params.outputWeights, m.outputWeights, v.outputWeights, atom, grad.outputWeights[atom], iteration, stepScale, beta1, beta2, epsilon);
        updateAdamArray(params.biases, m.biases, v.biases, atom, grad.biases[atom], iteration, stepScale, beta1, beta2, epsilon);
        updateAdamArray(params.logScales, m.logScales, v.logScales, atom, grad.logScales[atom], iteration, stepScale, beta1, beta2, epsilon);
        params.logScales[atom] = Math.max(Math.log(0.03), Math.min(Math.log(20), params.logScales[atom]));
        params.biases[atom] = Math.max(-8, Math.min(8, params.biases[atom]));
        params.outputWeights[atom] = Math.max(-8, Math.min(8, params.outputWeights[atom]));
      }
    }
  }
  return { params: scalarParamsToStructureParams(bestParams), mse: bestMse };
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
    r2: total > 0 ? 1 - squared / total : 1
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

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(6);
  return value.toPrecision(8).replace(/\.?0+$/, '');
}

function buildExpression(model: Omit<GeluFfnStructureModel, 'expression'>): string {
  const terms = [`${formatNumber(model.outputBias)}`];
  model.atoms.forEach((atom, index) => {
    terms.push(`${formatNumber(atom.outputWeight)} * gelu(dot(w${index}, x) + ${formatNumber(atom.inputBias)})`);
  });
  return terms.join(' + ');
}

export function decompileGeluFfnStructure(
  oracle: (input: number[]) => number,
  config: GeluFfnStructureFinderConfig
): GeluFfnStructureModel {
  const dimension = Math.max(2, Math.floor(config.dimension));
  if (config.ranges.length !== dimension) throw new Error('GELU FFN structure finder requires one range per dimension');
  if (config.ranges.some(([min, max]) => !(max > min))) throw new Error('GELU FFN structure finder requires valid ranges');
  const channelCount = Math.max(1, Math.floor(config.channelCount));
  const activeRank = Math.max(channelCount, Math.min(dimension, Math.floor(config.activeRank ?? channelCount)));
  const gradientProbeCount = Math.max(activeRank + 2, Math.floor(config.gradientProbeCount ?? Math.max(12, channelCount * 5)));
  const trainingSampleCount = Math.max(channelCount * 80, Math.floor(config.trainingSampleCount ?? Math.max(1400, channelCount * 700)));
  const gradientTrainingSampleCount = Math.max(0, Math.floor(config.gradientTrainingSampleCount ?? 0));
  const gradientTrainingWeight = Math.max(0, config.gradientTrainingWeight ?? 0);
  const validationSampleCount = Math.max(256, Math.floor(config.validationSampleCount ?? 1000));
  const minRange = Math.min(...config.ranges.map(([min, max]) => max - min));
  const requestedFiniteDifferenceStep = Math.max(1e-6, Math.min(config.finiteDifferenceStep ?? minRange * 1e-4, minRange * 0.02));
  const finiteDifferenceStep = config.finiteDifferenceStep === undefined
    ? Math.max(requestedFiniteDifferenceStep, minRange * 0.1)
    : requestedFiniteDifferenceStep;
  const activeGradientFiniteDifferenceStep = Math.max(
    1e-6,
    Math.min(config.activeGradientFiniteDifferenceStep ?? finiteDifferenceStep, minRange * 0.02)
  );
  const restarts = Math.max(1, Math.floor(config.restarts ?? Math.max(6, channelCount * 3)));
  const iterations = Math.max(50, Math.floor(config.iterations ?? 1200));
  const learningRate = config.learningRate ?? 0.03;
  const scalarRefinementIterations = Math.max(0, Math.floor(config.scalarRefinementIterations ?? 0));
  const scalarRefinementRestarts = Math.max(1, Math.floor(config.scalarRefinementRestarts ?? 1));
  const scalarRefinementLearningRate = config.scalarRefinementLearningRate ?? learningRate * 0.5;
  const ridge = config.ridge ?? 1e-7;
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregation = config.aggregation ?? 'median';
  const random = makeRandom(config.randomSeed ?? 260631);
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

  const gradientTrainingStart = rawOracleCalls;
  const gradientTrainingSamples: GradientTrainingSample[] = [];
  if (gradientTrainingWeight > 0 && gradientTrainingSampleCount > 0) {
    for (let index = 0; index < gradientTrainingSampleCount; index++) {
      const input = randomPoint(random, config.ranges, activeGradientFiniteDifferenceStep * 2);
      gradientTrainingSamples.push({
        coords: projectInput(activeBasis, input),
        targetGradient: estimateActiveBasisGradient(countedOracle, input, activeBasis, activeGradientFiniteDifferenceStep)
      });
    }
  }
  const gradientTrainingRawOracleCalls = rawOracleCalls - gradientTrainingStart;

  let { params, mse } = trainStructure(
    trainingSamples,
    gradientTrainingSamples,
    channelCount,
    activeRank,
    random,
    restarts,
    iterations,
    learningRate,
    gradientTrainingWeight,
    ridge
  );
  if (scalarRefinementIterations > 0) {
    const refined = trainFixedDirectionScalars(
      trainingSamples,
      gradientTrainingSamples,
      params,
      activeRank,
      random,
      scalarRefinementRestarts,
      scalarRefinementIterations,
      scalarRefinementLearningRate,
      gradientTrainingWeight,
      ridge
    );
    params = refined.params;
    mse = refined.mse;
  }

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

  const atoms = params.vectors.map((vector, index) => {
    const inputWeight = liftToInputWeight(activeBasis, vector, dimension);
    return {
      inputWeight,
      inputBias: params.biases[index],
      outputWeight: params.outputWeights[index],
      inputWeightNorm: norm(inputWeight)
    };
  }).sort((a, b) => Math.abs(b.outputWeight) * b.inputWeightNorm - Math.abs(a.outputWeight) * a.inputWeightNorm);

  const modelBase = {
    dimension,
    ranges: config.ranges,
    channelCount,
    activeRank,
    gradientProbeCount,
    trainingSampleCount,
    gradientTrainingSampleCount,
    validationSampleCount,
    finiteDifferenceStep,
    querySamples,
    aggregation,
    recoveryKind: 'gelu-ffn-structure' as const,
    activeBasis,
    atoms,
    outputBias: params.offset,
    trainingMse: mse,
    validationMetrics,
    optimizer: {
      restarts,
      iterations,
      learningRate,
      scalarRefinementRestarts: scalarRefinementIterations > 0 ? scalarRefinementRestarts : undefined,
      scalarRefinementIterations,
      scalarRefinementLearningRate: scalarRefinementIterations > 0 ? scalarRefinementLearningRate : undefined,
      gradientTrainingWeight: gradientTrainingWeight > 0 ? gradientTrainingWeight : undefined
    },
    queryBudget: {
      rawOracleCalls,
      gradientRawOracleCalls,
      trainingRawOracleCalls,
      gradientTrainingRawOracleCalls,
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

function atomStrength(atom: GeluFfnStructureAtom): number {
  return Math.abs(atom.outputWeight) * atom.inputWeightNorm;
}

function canonicalAtomStrength(atom: GeluFfnCanonicalAtom): number {
  return Math.abs(atom.outputWeight) * atom.inputWeightNorm;
}

function sameCanonicalSupport(
  left: GeluFfnCanonicalAtom,
  right: GeluFfnCanonicalAtom,
  directionCosine: number,
  biasTolerance: number,
  normTolerance: number
): boolean {
  const cosine = dot(left.direction, right.direction);
  return cosine >= directionCosine
    && Math.abs(left.inputBias - right.inputBias) <= biasTolerance
    && Math.abs(left.inputWeightNorm - right.inputWeightNorm) <= normTolerance;
}

function mergeCanonicalGroup(group: GeluFfnCanonicalAtom[]): GeluFfnCanonicalAtom {
  const totalAbsOutput = group.reduce((sum, atom) => sum + Math.abs(atom.outputWeight), 0);
  const weightedDirection = new Array(group[0].direction.length).fill(0);
  for (const atom of group) {
    const weight = Math.abs(atom.outputWeight) / Math.max(1e-12, totalAbsOutput);
    for (let index = 0; index < weightedDirection.length; index++) {
      weightedDirection[index] += atom.direction[index] * weight;
    }
  }
  const outputWeight = group.reduce((sum, atom) => sum + atom.outputWeight, 0);
  const inputWeightNorm = group.reduce((sum, atom) => sum + atom.inputWeightNorm * Math.abs(atom.outputWeight), 0) / Math.max(1e-12, totalAbsOutput);
  const inputBias = group.reduce((sum, atom) => sum + atom.inputBias * Math.abs(atom.outputWeight), 0) / Math.max(1e-12, totalAbsOutput);
  const direction = normalize(weightedDirection);
  return {
    direction,
    inputWeightNorm,
    inputBias,
    outputWeight,
    strength: Math.abs(outputWeight) * inputWeightNorm
  };
}

function sortCanonicalAtoms(atoms: GeluFfnCanonicalAtom[]): GeluFfnCanonicalAtom[] {
  return [...atoms].sort((a, b) => (
    b.strength - a.strength
    || a.inputBias - b.inputBias
    || a.inputWeightNorm - b.inputWeightNorm
    || a.outputWeight - b.outputWeight
  ));
}

export function canonicalizeGeluFfnStructure(
  model: Pick<GeluFfnStructureModel, 'atoms'>,
  config: GeluFfnCanonicalSignatureConfig = {}
): GeluFfnCanonicalSignature {
  const relativeThreshold = config.atomRelativeStrengthThreshold ?? 0.02;
  const absoluteThreshold = config.atomAbsoluteStrengthThreshold ?? 1e-5;
  const duplicateDirectionCosine = config.duplicateDirectionCosine ?? 0.9995;
  const duplicateBiasTolerance = config.duplicateBiasTolerance ?? 0.03;
  const duplicateNormTolerance = config.duplicateNormTolerance ?? 0.04;
  const cancellationOutputTolerance = config.cancellationOutputTolerance ?? 0.04;
  const canonicalAtoms = model.atoms.map(atom => {
    const inputWeightNorm = Math.max(1e-12, norm(atom.inputWeight));
    return {
      direction: atom.inputWeight.map(value => value / inputWeightNorm),
      inputWeightNorm,
      inputBias: atom.inputBias,
      outputWeight: atom.outputWeight,
      strength: atomStrength(atom)
    };
  });
  const maxStrength = Math.max(0, ...canonicalAtoms.map(atom => atom.strength));
  const strengthThreshold = Math.max(absoluteThreshold, maxStrength * relativeThreshold);
  let removedWeakAtomCount = 0;
  let removedCancellationCount = 0;
  const strongAtoms = canonicalAtoms.filter(atom => {
    const keep = atom.strength >= strengthThreshold;
    if (!keep) removedWeakAtomCount++;
    return keep;
  });

  const groups: GeluFfnCanonicalAtom[][] = [];
  for (const atom of sortCanonicalAtoms(strongAtoms)) {
    const group = groups.find(items => sameCanonicalSupport(
      items[0],
      atom,
      duplicateDirectionCosine,
      duplicateBiasTolerance,
      duplicateNormTolerance
    ));
    if (group) group.push(atom);
    else groups.push([atom]);
  }

  const merged: GeluFfnCanonicalAtom[] = [];
  let mergedDuplicateCount = 0;
  for (const group of groups) {
    mergedDuplicateCount += Math.max(0, group.length - 1);
    const atom = group.length === 1 ? group[0] : mergeCanonicalGroup(group);
    const beforeStrength = group.reduce((sum, item) => sum + canonicalAtomStrength(item), 0);
    const cancellationThreshold = Math.max(absoluteThreshold, beforeStrength * cancellationOutputTolerance);
    if (atom.strength < Math.max(strengthThreshold, cancellationThreshold)) {
      removedCancellationCount += group.length > 1 ? group.length : 0;
      continue;
    }
    merged.push({ ...atom, strength: canonicalAtomStrength(atom) });
  }

  const atoms = sortCanonicalAtoms(merged);
  return {
    atoms,
    canonicalAtomCount: atoms.length,
    removedWeakAtomCount,
    mergedDuplicateCount,
    removedCancellationCount
  };
}

function canonicalAtomDistance(left: GeluFfnCanonicalAtom, right: GeluFfnCanonicalAtom): number {
  const directionDistance = Math.sqrt(left.direction.reduce((sum, value, index) => (
    sum + (value - right.direction[index]) ** 2
  ), 0));
  const normDistance = Math.abs(left.inputWeightNorm - right.inputWeightNorm) / Math.max(1, left.inputWeightNorm, right.inputWeightNorm);
  const biasDistance = Math.abs(left.inputBias - right.inputBias);
  const outputDistance = Math.abs(left.outputWeight - right.outputWeight) / Math.max(1, Math.abs(left.outputWeight), Math.abs(right.outputWeight));
  return Math.sqrt(
    directionDistance ** 2
    + normDistance ** 2
    + biasDistance ** 2
    + outputDistance ** 2
  );
}

function bestCanonicalAssignment(
  distances: number[][],
  row: number,
  used: boolean[],
  pairs: Array<{ left: number; right: number; distance: number }>
): { total: number; pairs: Array<{ left: number; right: number; distance: number }> } {
  if (row >= distances.length) return { total: 0, pairs: [...pairs] };
  let best = { total: Infinity, pairs: [] as Array<{ left: number; right: number; distance: number }> };
  for (let col = 0; col < distances[row].length; col++) {
    if (used[col]) continue;
    used[col] = true;
    pairs.push({ left: row, right: col, distance: distances[row][col] });
    const rest = bestCanonicalAssignment(distances, row + 1, used, pairs);
    const total = distances[row][col] + rest.total;
    if (total < best.total) best = { total, pairs: rest.pairs };
    pairs.pop();
    used[col] = false;
  }
  return best;
}

export function compareGeluFfnCanonicalSignatures(
  left: GeluFfnCanonicalSignature,
  right: GeluFfnCanonicalSignature
): GeluFfnCanonicalSignatureComparison {
  const atomCountDelta = Math.abs(left.canonicalAtomCount - right.canonicalAtomCount);
  const count = Math.min(left.atoms.length, right.atoms.length);
  if (count === 0) {
    return {
      atomCountDelta,
      signatureDistance: atomCountDelta,
      maxAtomDistance: atomCountDelta,
      meanAtomDistance: atomCountDelta,
      matchedPairs: []
    };
  }
  const distances = left.atoms.slice(0, count).map(atom => (
    right.atoms.slice(0, count).map(other => canonicalAtomDistance(atom, other))
  ));
  const assignment = bestCanonicalAssignment(distances, 0, new Array(count).fill(false), []);
  const matchedDistances = assignment.pairs.map(pair => pair.distance);
  const meanAtomDistance = matchedDistances.reduce((sum, value) => sum + value, 0) / Math.max(1, matchedDistances.length);
  const maxAtomDistance = Math.max(0, ...matchedDistances);
  return {
    atomCountDelta,
    signatureDistance: meanAtomDistance + atomCountDelta,
    maxAtomDistance,
    meanAtomDistance,
    matchedPairs: assignment.pairs
  };
}

function countDuplicatePairs(
  atoms: GeluFfnStructureAtom[],
  directionCosine: number,
  biasTolerance: number
): number {
  let duplicates = 0;
  for (let left = 0; left < atoms.length; left++) {
    const leftNorm = Math.max(1e-12, norm(atoms[left].inputWeight));
    for (let right = left + 1; right < atoms.length; right++) {
      const rightNorm = Math.max(1e-12, norm(atoms[right].inputWeight));
      const cosine = Math.abs(dot(atoms[left].inputWeight, atoms[right].inputWeight) / (leftNorm * rightNorm));
      const biasDelta = Math.abs(atoms[left].inputBias - atoms[right].inputBias);
      if (cosine >= directionCosine && biasDelta <= biasTolerance) duplicates++;
    }
  }
  return duplicates;
}

function countNearParallelPairs(
  atoms: GeluFfnStructureAtom[],
  directionCosine: number
): number {
  let pairs = 0;
  for (let left = 0; left < atoms.length; left++) {
    const leftNorm = Math.max(1e-12, norm(atoms[left].inputWeight));
    for (let right = left + 1; right < atoms.length; right++) {
      const rightNorm = Math.max(1e-12, norm(atoms[right].inputWeight));
      const cosine = Math.abs(dot(atoms[left].inputWeight, atoms[right].inputWeight) / (leftNorm * rightNorm));
      if (cosine >= directionCosine) pairs++;
    }
  }
  return pairs;
}

function pruneWeakAtoms(
  atoms: GeluFfnStructureAtom[],
  relativeThreshold: number,
  absoluteThreshold: number,
  minEffectiveChannelCount: number
): { atoms: GeluFfnStructureAtom[]; threshold: number; maxStrength: number; prunedWeakAtomCount: number } {
  const strengths = atoms.map(atomStrength);
  const maxStrength = Math.max(0, ...strengths);
  const threshold = Math.max(absoluteThreshold, maxStrength * relativeThreshold);
  const ranked = atoms
    .map((atom, index) => ({ atom, strength: strengths[index] }))
    .sort((a, b) => b.strength - a.strength);
  const strong = ranked.filter(item => item.strength >= threshold).map(item => item.atom);
  const selected = strong.length >= minEffectiveChannelCount
    ? strong
    : ranked.slice(0, Math.min(atoms.length, minEffectiveChannelCount)).map(item => item.atom);
  return {
    atoms: selected,
    threshold,
    maxStrength,
    prunedWeakAtomCount: Math.max(0, atoms.length - selected.length)
  };
}

type AtomCandidate = {
  restart: number;
  kind: string;
  threshold: number;
  maxStrength: number;
  rawModel: GeluFfnStructureModel;
  atoms: GeluFfnStructureAtom[];
  prunedWeakAtomCount: number;
  duplicatePairCount: number;
  nearParallelPairCount: number;
  validationMetrics: { mae: number; rmse: number; r2: number };
};

function signatureAtoms(atoms: GeluFfnStructureAtom[]): string {
  return atoms
    .map(atom => `${atomStrength(atom).toPrecision(7)}:${atom.inputBias.toPrecision(7)}:${atom.inputWeightNorm.toPrecision(7)}`)
    .sort()
    .join('|');
}

function makeCandidateModel(
  rawModel: GeluFfnStructureModel,
  maxChannelCount: number,
  atoms: GeluFfnStructureAtom[]
): GeluFfnStructureModel {
  return {
    ...rawModel,
    recoveryKind: 'rank-adaptive-gelu-ffn-structure',
    channelCount: atoms.length,
    maxChannelCount,
    rawChannelCount: maxChannelCount,
    effectiveChannelCount: atoms.length,
    atoms,
    rawAtoms: rawModel.atoms
  };
}

function generateAtomCandidates(
  rawModel: GeluFfnStructureModel,
  maxChannelCount: number,
  minEffectiveChannelCount: number,
  relativeThreshold: number,
  absoluteThreshold: number,
  duplicateDirectionCosine: number,
  duplicateBiasTolerance: number,
  nearParallelDirectionCosine: number,
  validationInputs: number[][],
  validationActual: number[],
  restart: number
): AtomCandidate[] {
  const rawAtoms = rawModel.atoms;
  const candidates: Array<{ kind: string; atoms: GeluFfnStructureAtom[]; threshold: number; maxStrength: number; prunedWeakAtomCount: number }> = [];
  const thresholdPruned = pruneWeakAtoms(rawAtoms, relativeThreshold, absoluteThreshold, minEffectiveChannelCount);
  candidates.push({
    kind: `trained-${rawModel.channelCount}-strength-threshold`,
    atoms: thresholdPruned.atoms,
    threshold: thresholdPruned.threshold,
    maxStrength: thresholdPruned.maxStrength,
    prunedWeakAtomCount: thresholdPruned.prunedWeakAtomCount
  });

  const ranked = rawAtoms
    .map(atom => ({ atom, strength: atomStrength(atom) }))
    .sort((a, b) => b.strength - a.strength);
  const maxStrength = Math.max(0, ...ranked.map(item => item.strength));
  for (let count = minEffectiveChannelCount; count <= rawAtoms.length; count++) {
    candidates.push({
      kind: `trained-${rawModel.channelCount}-top-${count}`,
      atoms: ranked.slice(0, count).map(item => item.atom),
      threshold: ranked[Math.min(count - 1, ranked.length - 1)]?.strength ?? 0,
      maxStrength,
      prunedWeakAtomCount: Math.max(0, rawAtoms.length - count)
    });
  }

  const seen = new Set<string>();
  const evaluated: AtomCandidate[] = [];
  for (const candidate of candidates) {
    if (candidate.atoms.length < minEffectiveChannelCount) continue;
    const signature = `${candidate.kind}:${signatureAtoms(candidate.atoms)}`;
    if (seen.has(signature)) continue;
    seen.add(signature);
    const model = makeCandidateModel(rawModel, maxChannelCount, candidate.atoms);
    evaluated.push({
      restart,
      kind: candidate.kind,
      threshold: candidate.threshold,
      maxStrength: candidate.maxStrength,
      rawModel,
      atoms: candidate.atoms,
      prunedWeakAtomCount: candidate.prunedWeakAtomCount,
      duplicatePairCount: countDuplicatePairs(candidate.atoms, duplicateDirectionCosine, duplicateBiasTolerance),
      nearParallelPairCount: countNearParallelPairs(candidate.atoms, nearParallelDirectionCosine),
      validationMetrics: metrics(validationActual, predictGeluFfnStructureBatch(model, validationInputs))
    });
  }
  return evaluated;
}

export function decompileRankAdaptiveGeluFfnStructure(
  oracle: (input: number[]) => number,
  config: RankAdaptiveGeluFfnStructureFinderConfig
): GeluFfnStructureModel {
  const maxChannelCount = Math.max(1, Math.floor(config.maxChannelCount));
  const minEffectiveChannelCount = Math.max(1, Math.floor(config.minEffectiveChannelCount ?? 1));
  const relativeThreshold = config.atomRelativeStrengthThreshold ?? 0.03;
  const absoluteThreshold = config.atomAbsoluteStrengthThreshold ?? 1e-4;
  const duplicateDirectionCosine = config.duplicateDirectionCosine ?? 0.995;
  const duplicateBiasTolerance = config.duplicateBiasTolerance ?? 0.08;
  const nearParallelDirectionCosine = config.nearParallelDirectionCosine ?? duplicateDirectionCosine;
  const selectionRestarts = Math.max(1, Math.floor(config.selectionRestarts ?? 1));
  const pruneValidationToleranceRatio = config.pruneValidationToleranceRatio ?? 0.2;
  const pruneValidationToleranceAbsolute = config.pruneValidationToleranceAbsolute ?? 0.002;
  const candidateChannelCounts = (config.candidateChannelCounts?.length
    ? [...new Set(config.candidateChannelCounts.map(value => Math.floor(value)))]
    : Array.from(
      { length: maxChannelCount - minEffectiveChannelCount + 1 },
      (_unused, index) => minEffectiveChannelCount + index
    ))
    .filter(value => value >= minEffectiveChannelCount && value <= maxChannelCount)
    .sort((a, b) => a - b);
  if (!candidateChannelCounts.length) {
    throw new Error('rank-adaptive GELU structure finder requires at least one candidate channel count');
  }
  const rawModels: GeluFfnStructureModel[] = [];
  for (const channelCount of candidateChannelCounts) {
    for (let restart = 0; restart < selectionRestarts; restart++) {
      rawModels.push(decompileGeluFfnStructure(oracle, {
        ...config,
        channelCount,
        activeRank: Math.max(maxChannelCount, Math.floor(config.activeRank ?? maxChannelCount)),
        randomSeed: (config.randomSeed ?? 260631) + restart * 7919 + channelCount * 104729
      }));
    }
  }

  let adaptiveValidationRawOracleCalls = 0;
  let adaptiveValidationAggregatedProbePoints = 0;
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregation = config.aggregation ?? 'median';
  const adaptiveRandom = makeRandom((config.randomSeed ?? 260631) + 99173);
  const validationCount = Math.max(256, Math.floor(config.validationSampleCount ?? 1000));
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

  const candidates = rawModels.flatMap((rawModel, restart) => generateAtomCandidates(
    rawModel,
    maxChannelCount,
    minEffectiveChannelCount,
    relativeThreshold,
    absoluteThreshold,
    duplicateDirectionCosine,
    duplicateBiasTolerance,
    nearParallelDirectionCosine,
    adaptiveValidationInputs,
    adaptiveActual,
    restart
  ));
  if (!candidates.length) throw new Error('rank-adaptive GELU structure finder produced no valid atom candidates');

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
      a.atoms.length - b.atoms.length
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
  const rawAllAtomModel = makeCandidateModel(rawModel, maxChannelCount, rawModel.atoms);
  const unprunedValidationMetrics = metrics(
    adaptiveActual,
    predictGeluFfnStructureBatch(rawAllAtomModel, adaptiveValidationInputs)
  );

  const selectedModel: GeluFfnStructureModel = {
    ...rawModel,
    recoveryKind: 'rank-adaptive-gelu-ffn-structure',
    channelCount: selected.atoms.length,
    maxChannelCount,
    rawChannelCount: maxChannelCount,
    effectiveChannelCount: selected.atoms.length,
    atoms: selected.atoms,
    rawAtoms: rawModel.atoms,
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
      atomStrengthThreshold: selected.threshold,
      maxAtomStrength: selected.maxStrength,
      prunedWeakAtomCount: selected.prunedWeakAtomCount,
      duplicatePairCount: selected.duplicatePairCount,
      nearParallelPairCount: selected.nearParallelPairCount,
      duplicateDirectionCosine,
      duplicateBiasTolerance,
      nearParallelDirectionCosine,
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

export function predictGeluFfnStructureModel(model: GeluFfnStructureModel, input: number[]): number {
  if (input.length !== model.dimension) throw new Error('GELU FFN structure prediction dimension mismatch');
  let value = model.outputBias;
  for (const atom of model.atoms) value += atom.outputWeight * gelu(dot(atom.inputWeight, input) + atom.inputBias);
  return Number.isFinite(value) ? value : 0;
}

export function predictGeluFfnStructureBatch(model: GeluFfnStructureModel, inputs: number[][]): number[] {
  return inputs.map(input => predictGeluFfnStructureModel(model, input));
}
