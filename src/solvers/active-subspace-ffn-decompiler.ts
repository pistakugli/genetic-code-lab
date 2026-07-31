export type ActiveSubspaceFfnTrainingSample = {
  coords: number[];
  target: number;
};

export type ActiveSubspaceFfnDecompilerModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  activeRank: number;
  gradientProbeCount: number;
  trainingSampleCount: number;
  finiteDifferenceStep: number;
  neighborCount: number;
  ridge: number;
  localModel: 'local-quadratic-knn' | 'local-affine-knn' | 'kernel-mean-knn';
  recoveryKind: 'active-subspace-ffn-surrogate';
  directionEstimator: 'finite-difference' | 'moment-sir';
  directionCandidateScores: Array<{
    estimator: 'finite-difference' | 'moment-sir';
    validationMae: number;
    validationCount: number;
  }>;
  activeDirections: number[][];
  gradientEigenvalues: number[];
  totalGradientEnergy: number;
  selectedGradientEnergyRatio: number;
  meanGradientResidualRatio: number;
  samples: ActiveSubspaceFfnTrainingSample[];
  queryBudget: {
    rawOracleCalls: number;
    gradientRawOracleCalls: number;
    trainingRawOracleCalls: number;
    finiteDifferenceCallsPerProbe: number;
  };
  expression: string;
};

export type ActiveSubspaceFfnDecompilerConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  activeRank: number;
  gradientProbeCount?: number;
  trainingSampleCount?: number;
  finiteDifferenceStep?: number;
  neighborCount?: number;
  randomSeed?: number;
  ridge?: number;
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

type RawTrainingSample = {
  input: number[];
  target: number;
};

type DirectionCandidate = {
  estimator: 'finite-difference' | 'moment-sir';
  directions: number[][];
  eigenvalues: number[];
  totalEnergy: number;
};

function coordinateCenters(ranges: Array<[number, number]>): number[] {
  return ranges.map(([min, max]) => (min + max) / 2);
}

function coordinateHalfRanges(ranges: Array<[number, number]>): number[] {
  return ranges.map(([min, max]) => Math.max(1e-12, (max - min) / 2));
}

function standardizeInput(input: number[], centers: number[], halfRanges: number[]): number[] {
  return input.map((value, index) => (value - centers[index]) / halfRanges[index]);
}

function originalDirectionFromStandardized(direction: number[], halfRanges: number[]): number[] {
  return normalize(direction.map((value, index) => value / halfRanges[index]));
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function centeredFeature(values: number[], transform: (value: number) => number): number[] {
  const transformed = values.map(transform);
  const transformedMean = mean(transformed);
  return transformed.map(value => value - transformedMean);
}

function buildMomentVector(
  standardizedInputs: number[][],
  feature: number[]
): { vector: number[]; energy: number } {
  const dimension = standardizedInputs[0]?.length ?? 0;
  const vector = new Array(dimension).fill(0);
  for (let row = 0; row < standardizedInputs.length; row++) {
    const coefficient = feature[row];
    for (let axis = 0; axis < dimension; axis++) vector[axis] += standardizedInputs[row][axis] * coefficient;
  }
  for (let axis = 0; axis < dimension; axis++) vector[axis] /= Math.max(1, standardizedInputs.length);
  return { vector, energy: dot(vector, vector) };
}

function recoverMomentDirections(
  rawSamples: RawTrainingSample[],
  ranges: Array<[number, number]>,
  rank: number
): { directions: number[][]; eigenvalues: number[]; totalEnergy: number } {
  const dimension = ranges.length;
  const centers = coordinateCenters(ranges);
  const halfRanges = coordinateHalfRanges(ranges);
  const standardizedInputs = rawSamples.map(sample => standardizeInput(sample.input, centers, halfRanges));
  const targets = rawSamples.map(sample => sample.target);
  const targetMean = mean(targets);
  const targetVariance = mean(targets.map(value => (value - targetMean) ** 2));
  const targetScale = Math.sqrt(Math.max(1e-12, targetVariance));
  const z = targets.map(value => (value - targetMean) / targetScale);
  const featureSets = [
    centeredFeature(z, value => value),
    centeredFeature(z, value => value ** 2),
    centeredFeature(z, value => value ** 3),
    centeredFeature(z, value => Math.tanh(value)),
    centeredFeature(z, value => Math.max(0, value)),
    centeredFeature(z, value => value >= 0 ? 1 : -1)
  ];
  const candidateVectors = featureSets
    .map(feature => buildMomentVector(standardizedInputs, feature))
    .filter(candidate => candidate.energy > 1e-18)
    .sort((a, b) => b.energy - a.energy);

  const directions: number[][] = [];
  const eigenvalues: number[] = [];
  for (const candidate of candidateVectors) {
    const orthogonal = subtractProjection(candidate.vector, directions);
    if (norm(orthogonal) < 1e-10) continue;
    directions.push(canonicalizeDirection(originalDirectionFromStandardized(normalize(orthogonal), halfRanges)));
    eigenvalues.push(candidate.energy);
    if (directions.length >= rank) break;
  }
  while (directions.length < rank) {
    const fallback = new Array(dimension).fill(0);
    fallback[directions.length % dimension] = 1;
    const orthogonal = subtractProjection(fallback, directions);
    directions.push(canonicalizeDirection(normalize(orthogonal)));
    eigenvalues.push(0);
  }
  const totalEnergy = candidateVectors.reduce((sum, candidate) => sum + candidate.energy, 0);
  return { directions, eigenvalues, totalEnergy };
}

function recoverActiveDirections(
  gradients: number[][],
  rank: number,
  random: () => number
): { directions: number[][]; eigenvalues: number[]; totalEnergy: number } {
  const dimension = gradients[0]?.length ?? 0;
  const totalEnergy = gradients.reduce((sum, gradient) => sum + dot(gradient, gradient), 0) / Math.max(1, gradients.length);
  const directions: number[][] = [];
  const eigenvalues: number[] = [];
  for (let component = 0; component < rank; component++) {
    let vector = normalize(subtractProjection(
      Array.from({ length: dimension }, () => randomBetween(random, -1, 1)),
      directions
    ));
    for (let iteration = 0; iteration < 80; iteration++) {
      const multiplied = covarianceVector(gradients, vector);
      const orthogonal = subtractProjection(multiplied, directions);
      const length = norm(orthogonal);
      if (length < 1e-14) break;
      vector = orthogonal.map(value => value / length);
    }
    vector = canonicalizeDirection(normalize(subtractProjection(vector, directions)));
    const eigenvalue = Math.max(0, dot(vector, covarianceVector(gradients, vector)));
    directions.push(vector);
    eigenvalues.push(eigenvalue);
  }
  return { directions, eigenvalues, totalEnergy };
}

function projectInput(directions: number[][], input: number[]): number[] {
  return directions.map(direction => dot(direction, input));
}

function squaredDistance(a: number[], b: number[]): number {
  let distance = 0;
  for (let index = 0; index < a.length; index++) {
    const delta = a[index] - b[index];
    distance += delta * delta;
  }
  return distance;
}

function insertNeighbor(
  neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }>,
  candidate: { distance: number; sample: ActiveSubspaceFfnTrainingSample },
  maxCount: number
): void {
  let index = 0;
  while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
  if (index >= maxCount) return;
  neighbors.splice(index, 0, candidate);
  if (neighbors.length > maxCount) neighbors.pop();
}

function solveSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, index) => [...row, rhs[index]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    if (Math.abs(divisor) < 1e-12) throw new Error('active-subspace local quadratic system is singular');
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      if (Math.abs(factor) < 1e-20) continue;
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[n]);
}

function meanTarget(samples: ActiveSubspaceFfnTrainingSample[]): number {
  return samples.reduce((sum, sample) => sum + sample.target, 0) / Math.max(1, samples.length);
}

function buildExpression(model: Omit<ActiveSubspaceFfnDecompilerModel, 'expression'>): string {
  return [
    'active_subspace_ffn(',
    `dimension=${model.dimension}`,
    `rank=${model.activeRank}`,
    `gradient_probes=${model.gradientProbeCount}`,
    `samples=${model.trainingSampleCount}`,
    `estimator=${model.directionEstimator}`,
    `energy=${model.selectedGradientEnergyRatio.toPrecision(6)}`,
    `queries=${model.queryBudget.rawOracleCalls}`,
    ')'
  ].join(' ');
}

function projectRawSamples(directions: number[][], rawSamples: RawTrainingSample[]): ActiveSubspaceFfnTrainingSample[] {
  return rawSamples.map(sample => ({
    coords: projectInput(directions, sample.input),
    target: sample.target
  }));
}

function quadraticFeature(values: number[]): number[] {
  const row = [1, ...values];
  for (let i = 0; i < values.length; i++) {
    for (let j = i; j < values.length; j++) row.push(values[i] * values[j]);
  }
  return row;
}

function localFeatureDelta(delta: number[], degree: 1 | 2): number[] {
  const row = [1, ...delta];
  if (degree === 2) {
    for (let i = 0; i < delta.length; i++) {
      for (let j = i; j < delta.length; j++) row.push(delta[i] * delta[j]);
    }
  }
  return row;
}

function chooseLocalModel(
  activeRank: number,
  neighborCount: number
): ActiveSubspaceFfnDecompilerModel['localModel'] {
  const quadraticWidth = 1 + activeRank + (activeRank * (activeRank + 1)) / 2;
  const affineWidth = 1 + activeRank;
  if (activeRank <= 12 && neighborCount >= Math.ceil(quadraticWidth * 1.3)) return 'local-quadratic-knn';
  if (neighborCount >= Math.ceil(affineWidth * 1.5)) return 'local-affine-knn';
  return 'kernel-mean-knn';
}

function neighborMean(neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }>): number {
  return meanTarget(neighbors.map(neighbor => neighbor.sample));
}

function kernelMean(neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }>): number {
  if (!neighbors.length) return 0;
  const bandwidth = Math.max(1e-10, neighbors[neighbors.length - 1].distance);
  let weightedSum = 0;
  let weightTotal = 0;
  for (const neighbor of neighbors) {
    const weight = Math.exp(-neighbor.distance / (2 * bandwidth)) + 1e-4;
    weightedSum += weight * neighbor.sample.target;
    weightTotal += weight;
  }
  return weightTotal > 0 ? weightedSum / weightTotal : neighborMean(neighbors);
}

function localRegression(
  neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }>,
  coords: number[],
  degree: 1 | 2,
  ridge: number
): number | null {
  const width = degree === 1
    ? 1 + coords.length
    : quadraticFeature(coords).length;
  if (neighbors.length < width) return null;
  const bandwidth = Math.max(1e-10, neighbors[neighbors.length - 1].distance);
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const neighbor of neighbors) {
    const delta = neighbor.sample.coords.map((value, index) => value - coords[index]);
    const row = localFeatureDelta(delta, degree);
    const weight = Math.exp(-neighbor.distance / (2 * bandwidth)) + 1e-4;
    for (let i = 0; i < width; i++) {
      xty[i] += weight * row[i] * neighbor.sample.target;
      for (let j = 0; j < width; j++) xtx[i][j] += weight * row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += ridge;
  try {
    const coefficients = solveSystem(xtx, xty);
    return Number.isFinite(coefficients[0]) ? coefficients[0] : null;
  } catch {
    return null;
  }
}

/**
 * Counts the times a prediction could not use the declared local model.
 *
 * Both fallbacks below are silent: too few neighbours returns the global mean, and a
 * singular local system falls through to a kernel mean, while the model keeps advertising
 * `localModel: 'local-quadratic-knn'`. The reported error metrics are honestly recomputed
 * against the degraded predictions, so no number is fabricated, but a reader cannot tell a
 * model that used its declared method from one that mostly did not. Counting makes that
 * visible without changing what is predicted.
 */
export type LocalModelFallbackCounts = {
  insufficientNeighbors: number;
  singularLocalSystem: number;
  total: number;
};

function predictFromSamples(
  samples: ActiveSubspaceFfnTrainingSample[],
  activeRank: number,
  neighborCount: number,
  ridge: number,
  localModel: ActiveSubspaceFfnDecompilerModel['localModel'],
  coords: number[],
  fallbacks?: LocalModelFallbackCounts
): number {
  const neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }> = [];
  for (const sample of samples) {
    insertNeighbor(neighbors, { distance: squaredDistance(coords, sample.coords), sample }, neighborCount);
  }
  if (neighbors.length < activeRank + 1) {
    if (fallbacks) { fallbacks.insufficientNeighbors++; fallbacks.total++; }
    return meanTarget(samples);
  }
  const predicted = localModel === 'local-quadratic-knn'
    ? localRegression(neighbors, coords, 2, ridge) ?? localRegression(neighbors, coords, 1, ridge)
    : localModel === 'local-affine-knn'
      ? localRegression(neighbors, coords, 1, ridge)
      : null;
  if (predicted === null && localModel !== 'kernel-mean-knn' && fallbacks) {
    fallbacks.singularLocalSystem++;
    fallbacks.total++;
  }
  return predicted ?? kernelMean(neighbors);
}

function scoreDirectionCandidate(
  candidate: DirectionCandidate,
  rawSamples: RawTrainingSample[],
  activeRank: number,
  neighborCount: number,
  ridge: number,
  localModel: ActiveSubspaceFfnDecompilerModel['localModel']
): { validationMae: number; validationCount: number } {
  const validationStride = Math.max(5, Math.floor(rawSamples.length / Math.max(24, activeRank * 16)));
  const trainRaw: RawTrainingSample[] = [];
  const validationRaw: RawTrainingSample[] = [];
  for (let index = 0; index < rawSamples.length; index++) {
    if (index % validationStride === 0) validationRaw.push(rawSamples[index]);
    else trainRaw.push(rawSamples[index]);
  }
  const trainSamples = projectRawSamples(candidate.directions, trainRaw.length >= activeRank + 4 ? trainRaw : rawSamples);
  const validation = validationRaw.length ? validationRaw : rawSamples.slice(0, Math.min(rawSamples.length, 64));
  let absoluteError = 0;
  for (const sample of validation) {
    const coords = projectInput(candidate.directions, sample.input);
    absoluteError += Math.abs(sample.target - predictFromSamples(trainSamples, activeRank, neighborCount, ridge, localModel, coords));
  }
  return {
    validationMae: absoluteError / Math.max(1, validation.length),
    validationCount: validation.length
  };
}

export function decompileActiveSubspaceFfn(
  oracle: (input: number[]) => number,
  config: ActiveSubspaceFfnDecompilerConfig
): ActiveSubspaceFfnDecompilerModel {
  const dimension = Math.max(2, Math.floor(config.dimension));
  if (config.ranges.length !== dimension) throw new Error('active-subspace FFN decompiler requires one range per dimension');
  if (config.ranges.some(([min, max]) => !(max > min))) throw new Error('active-subspace FFN decompiler requires valid ranges');
  const activeRank = Math.max(1, Math.min(dimension, Math.floor(config.activeRank)));
  const gradientProbeCount = Math.max(activeRank + 2, Math.floor(config.gradientProbeCount ?? 16));
  const trainingSampleCount = Math.max(activeRank * 16, Math.floor(config.trainingSampleCount ?? 2400));
  const minRange = Math.min(...config.ranges.map(([min, max]) => max - min));
  const requestedFiniteDifferenceStep = Math.max(1e-6, Math.min(config.finiteDifferenceStep ?? minRange * 1e-4, minRange * 0.02));
  const robustStepFraction = activeRank === 1 && dimension >= 128 ? 0.2 : 0.1;
  const finiteDifferenceStep = Math.max(requestedFiniteDifferenceStep, minRange * robustStepFraction);
  const neighborCount = Math.max(activeRank + 2, Math.floor(config.neighborCount ?? Math.max(32, activeRank * 24)));
  const ridge = config.ridge ?? 1e-8;
  const localModel = chooseLocalModel(activeRank, neighborCount);
  const random = makeRandom(config.randomSeed ?? 260601);
  let rawOracleCalls = 0;
  const countedOracle = (input: number[]): number => {
    rawOracleCalls++;
    return oracle(input);
  };

  const gradients: number[][] = [];
  const gradientCallsStart = rawOracleCalls;
  for (let index = 0; index < gradientProbeCount; index++) {
    const point = randomPoint(random, config.ranges, finiteDifferenceStep * 2);
    gradients.push(estimateCoordinateGradient(countedOracle, point, finiteDifferenceStep));
  }
  const gradientRawOracleCalls = rawOracleCalls - gradientCallsStart;

  const finiteDifferenceRecovery = recoverActiveDirections(gradients, activeRank, random);

  const trainingCallsStart = rawOracleCalls;
  const rawSamples: RawTrainingSample[] = [];
  for (let index = 0; index < trainingSampleCount; index++) {
    const input = randomPoint(random, config.ranges, 0);
    rawSamples.push({
      input,
      target: countedOracle(input)
    });
  }
  const trainingRawOracleCalls = rawOracleCalls - trainingCallsStart;

  const candidates: DirectionCandidate[] = [
    { estimator: 'finite-difference', ...finiteDifferenceRecovery }
  ];
  if (activeRank === 1) {
    const momentRecovery = recoverMomentDirections(rawSamples, config.ranges, activeRank);
    candidates.push({ estimator: 'moment-sir', ...momentRecovery });
  }
  const scoredCandidates = candidates.map(candidate => ({
    candidate,
    score: scoreDirectionCandidate(candidate, rawSamples, activeRank, neighborCount, ridge, localModel)
  }));
  scoredCandidates.sort((a, b) => a.score.validationMae - b.score.validationMae);
  const selectedCandidate = scoredCandidates[0].candidate;
  const directions = selectedCandidate.directions;
  const eigenvalues = selectedCandidate.eigenvalues;
  const totalEnergy = selectedCandidate.totalEnergy;
  const samples = projectRawSamples(directions, rawSamples);
  const directionCandidateScores = scoredCandidates.map(({ candidate, score }) => ({
    estimator: candidate.estimator,
    validationMae: score.validationMae,
    validationCount: score.validationCount
  }));

  const selectedEnergy = eigenvalues.reduce((sum, value) => sum + value, 0);
  const selectedGradientEnergyRatio = Math.min(1, selectedEnergy / Math.max(1e-12, totalEnergy));
  let residualRatioSum = 0;
  for (const gradient of gradients) {
    const projection = directions.reduce((sum, direction) => sum + dot(gradient, direction) ** 2, 0);
    residualRatioSum += Math.sqrt(Math.max(0, dot(gradient, gradient) - projection)) / Math.max(1e-12, norm(gradient));
  }
  const meanGradientResidualRatio = residualRatioSum / Math.max(1, gradients.length);

  const modelBase = {
    dimension,
    ranges: config.ranges,
    activeRank,
    gradientProbeCount,
    trainingSampleCount,
    finiteDifferenceStep,
    neighborCount,
    ridge,
    localModel,
    recoveryKind: 'active-subspace-ffn-surrogate' as const,
    directionEstimator: selectedCandidate.estimator,
    directionCandidateScores,
    activeDirections: directions,
    gradientEigenvalues: eigenvalues,
    totalGradientEnergy: totalEnergy,
    selectedGradientEnergyRatio,
    meanGradientResidualRatio,
    samples,
    queryBudget: {
      rawOracleCalls,
      gradientRawOracleCalls,
      trainingRawOracleCalls,
      finiteDifferenceCallsPerProbe: 2 * dimension
    }
  };
  return {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
}

export function predictActiveSubspaceFfnModel(model: ActiveSubspaceFfnDecompilerModel, input: number[]): number {
  if (input.length !== model.dimension) throw new Error('active-subspace FFN prediction dimension mismatch');
  const coords = projectInput(model.activeDirections, input);
  return predictFromSamples(model.samples, model.activeRank, model.neighborCount, model.ridge, model.localModel, coords);
}

/**
 * Report how often a set of inputs could not be predicted with the model's declared local
 * method, and had to fall back to a global or kernel mean.
 *
 * `model.localModel` states the intended method, but two paths in `predictFromSamples`
 * silently degrade: too few neighbours returns the global mean, and a singular local system
 * falls through to a kernel mean. Nothing in the model distinguished a run that used its
 * declared method throughout from one that mostly did not, which is the difference between
 * a local model and a smoothed average wearing its name.
 *
 * The predictions themselves are unchanged; this only measures them. `appliedRate` is the
 * fraction of inputs that were predicted with the declared method.
 */
export function auditActiveSubspaceFfnFallbacks(
  model: ActiveSubspaceFfnDecompilerModel,
  inputs: number[][]
): LocalModelFallbackCounts & { predictionCount: number; appliedRate: number } {
  const fallbacks: LocalModelFallbackCounts = { insufficientNeighbors: 0, singularLocalSystem: 0, total: 0 };
  for (const input of inputs) {
    if (input.length !== model.dimension) throw new Error('active-subspace FFN prediction dimension mismatch');
    const coords = projectInput(model.activeDirections, input);
    predictFromSamples(model.samples, model.activeRank, model.neighborCount, model.ridge, model.localModel, coords, fallbacks);
  }
  const predictionCount = inputs.length;
  return {
    ...fallbacks,
    predictionCount,
    appliedRate: predictionCount === 0 ? 1 : (predictionCount - fallbacks.total) / predictionCount
  };
}

export function predictActiveSubspaceFfnBatch(model: ActiveSubspaceFfnDecompilerModel, inputs: number[][]): number[] {
  return inputs.map(input => predictActiveSubspaceFfnModel(model, input));
}
