export type DeepBlackBoxReluNdRegionSamplerRegion = {
  id: number;
  intercept: number;
  weights: number[];
  support: number;
  centroid: number[];
  fitRmse: number;
};

export type DeepBlackBoxReluNdRegionSamplerAssignment = {
  input: number[];
  regionId: number;
  target: number;
};

export type DeepBlackBoxReluNdRegionSamplerModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  sampleCount: number;
  samplingMode: 'random' | 'latin-hypercube';
  minRegionSupport: number;
  retainedSampleCount: number;
  droppedSampleCount: number;
  gradientStep: number;
  querySamples: number;
  aggregation: 'mean' | 'median' | 'trimmed-mean';
  trimFraction: number;
  neighborCount: number;
  predictionMode: 'region-vote' | 'local-affine-knn';
  localAffineNeighborCount: number;
  ridge: number;
  recoveryKind: 'output-equivalent-nd-region-sampler';
  regions: DeepBlackBoxReluNdRegionSamplerRegion[];
  assignments: DeepBlackBoxReluNdRegionSamplerAssignment[];
  regionCount: number;
  expression: string;
};

export type DeepBlackBoxReluNdRegionSamplerConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  sampleCount?: number;
  gradientStep?: number;
  slopeTolerance?: number;
  interceptTolerance?: number;
  minRegionSupport?: number;
  minRegionSupportFloor?: number;
  querySamples?: number;
  aggregation?: 'mean' | 'median' | 'trimmed-mean';
  trimFraction?: number;
  neighborCount?: number;
  predictionMode?: 'region-vote' | 'local-affine-knn';
  localAffineNeighborCount?: number;
  samplingMode?: 'random' | 'latin-hypercube';
  randomSeed?: number;
  ridge?: number;
};

type GradientSample = {
  input: number[];
  target: number;
  weights: number[];
  intercept: number;
  clusterId: number;
};

type GradientCluster = {
  id: number;
  weights: number[];
  intercept: number;
  samples: GradientSample[];
};

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function aggregate(
  values: number[],
  mode: 'mean' | 'median' | 'trimmed-mean',
  trimFraction: number
): number {
  if (mode === 'mean') return mean(values);
  const sorted = [...values].sort((a, b) => a - b);
  if (mode === 'median') return sorted[Math.floor(sorted.length / 2)];
  const trim = Math.min(Math.floor(sorted.length / 2), Math.floor(sorted.length * trimFraction));
  return mean(sorted.slice(trim, sorted.length - trim));
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

function shuffledIndexes(random: () => number, count: number): number[] {
  const indexes = Array.from({ length: count }, (_, index) => index);
  for (let index = indexes.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [indexes[index], indexes[swapIndex]] = [indexes[swapIndex], indexes[index]];
  }
  return indexes;
}

function makeLatinHypercubePoints(
  random: () => number,
  ranges: Array<[number, number]>,
  count: number,
  margin: number
): number[][] {
  const dimension = ranges.length;
  const permutations = Array.from({ length: dimension }, () => shuffledIndexes(random, count));
  return Array.from({ length: count }, (_, sampleIndex) => (
    ranges.map(([min, max], axis) => {
      const boundedMargin = Math.min(margin, (max - min) * 0.2);
      const low = min + boundedMargin;
      const high = max - boundedMargin;
      const stratum = (permutations[axis][sampleIndex] + random()) / count;
      return low + (high - low) * stratum;
    })
  ));
}

function maxAbsWeightDelta(a: number[], b: number[]): number {
  let delta = 0;
  for (let index = 0; index < a.length; index++) delta = Math.max(delta, Math.abs(a[index] - b[index]));
  return delta;
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(8);
  return value.toPrecision(10).replace(/\.?0+$/, '');
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
    if (Math.abs(divisor) < 1e-12) throw new Error('ND region sampler linear system is singular');
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

function fitAffine(
  samples: GradientSample[],
  dimension: number,
  ridge: number,
  fallbackWeights: number[],
  fallbackIntercept: number
): { intercept: number; weights: number[]; rmse: number } {
  const width = dimension + 1;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const sample of samples) {
    const row = [1, ...sample.input];
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * sample.target;
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += ridge;
  let coefficients: number[];
  try {
    coefficients = solveSystem(xtx, xty);
  } catch {
    coefficients = [fallbackIntercept, ...fallbackWeights];
  }
  let sse = 0;
  for (const sample of samples) {
    const predicted = coefficients[0] + dot(coefficients.slice(1), sample.input);
    const error = predicted - sample.target;
    sse += error * error;
  }
  return {
    intercept: coefficients[0],
    weights: coefficients.slice(1),
    rmse: Math.sqrt(sse / Math.max(1, samples.length))
  };
}

function centroid(samples: GradientSample[], dimension: number): number[] {
  const center = new Array(dimension).fill(0);
  for (const sample of samples) {
    for (let index = 0; index < dimension; index++) center[index] += sample.input[index];
  }
  return center.map(value => value / Math.max(1, samples.length));
}

function buildExpression(model: Omit<DeepBlackBoxReluNdRegionSamplerModel, 'expression'>): string {
  return [
    'nd_region_sampler(',
    `dimension=${model.dimension}`,
    `regions=${model.regionCount}`,
    `samples=${model.retainedSampleCount}/${model.sampleCount}`,
    `sampling=${model.samplingMode}`,
    `support=${model.minRegionSupport}`,
    `k=${model.neighborCount}`,
    `mode=${model.predictionMode}`,
    ')'
  ].join(' ');
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
  neighbors: Array<{ distance: number; regionId: number }>,
  candidate: { distance: number; regionId: number },
  maxCount: number
): void {
  let index = 0;
  while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
  if (index >= maxCount) return;
  neighbors.splice(index, 0, candidate);
  if (neighbors.length > maxCount) neighbors.pop();
}

export function decompileDeepBlackBoxReluNdRegionSampler(
  oracle: (input: number[]) => number,
  config: DeepBlackBoxReluNdRegionSamplerConfig
): DeepBlackBoxReluNdRegionSamplerModel {
  const dimension = Math.max(2, Math.floor(config.dimension));
  if (config.ranges.length !== dimension) throw new Error('ND region sampler requires one range per dimension');
  if (config.ranges.some(([min, max]) => !(max > min))) throw new Error('ND region sampler requires valid ranges');

  const sampleCount = Math.max(128, Math.floor(config.sampleCount ?? 6000));
  const minRange = Math.min(...config.ranges.map(([min, max]) => max - min));
  const gradientStep = Math.max(1e-8, Math.min(config.gradientStep ?? minRange * 1e-6, minRange * 0.05));
  const slopeTolerance = config.slopeTolerance ?? 1e-5;
  const interceptTolerance = config.interceptTolerance ?? 1e-5;
  const minRegionSupportFloor = Math.max(1, Math.floor(config.minRegionSupportFloor ?? dimension + 2));
  const minRegionSupport = Math.max(minRegionSupportFloor, Math.floor(config.minRegionSupport ?? dimension + 3));
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregationMode = config.aggregation ?? 'mean';
  const trimFraction = Math.max(0, Math.min(0.45, config.trimFraction ?? 0.2));
  const neighborCount = Math.max(1, Math.floor(config.neighborCount ?? 11));
  const predictionMode = config.predictionMode ?? 'region-vote';
  const localAffineNeighborCount = Math.max(dimension + 2, Math.floor(config.localAffineNeighborCount ?? Math.max(neighborCount, 3 * (dimension + 1))));
  const samplingMode = config.samplingMode ?? 'random';
  const ridge = config.ridge ?? 1e-10;
  const random = makeRandom(config.randomSeed ?? 260526);

  const sampleOracle = (input: number[]): number => {
    if (querySamples === 1) return oracle(input);
    return aggregate(Array.from({ length: querySamples }, () => oracle(input)), aggregationMode, trimFraction);
  };

  const clusters: GradientCluster[] = [];
  const samples: GradientSample[] = [];
  const margin = gradientStep * 4;
  const samplePoints = samplingMode === 'latin-hypercube'
    ? makeLatinHypercubePoints(random, config.ranges, sampleCount, margin)
    : Array.from({ length: sampleCount }, () => randomPoint(random, config.ranges, margin));
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const input = samplePoints[sampleIndex];
    const center = sampleOracle(input);
    const weights: number[] = [];
    for (let axis = 0; axis < dimension; axis++) {
      const left = [...input];
      const right = [...input];
      left[axis] -= gradientStep;
      right[axis] += gradientStep;
      weights.push((sampleOracle(right) - sampleOracle(left)) / (2 * gradientStep));
    }
    const intercept = center - dot(weights, input);
    let cluster = clusters.find(candidate => (
      maxAbsWeightDelta(candidate.weights, weights) <= slopeTolerance
      && Math.abs(candidate.intercept - intercept) <= interceptTolerance
    ));
    if (!cluster) {
      cluster = {
        id: clusters.length,
        weights: [...weights],
        intercept,
        samples: []
      };
      clusters.push(cluster);
    }
    const gradientSample: GradientSample = {
      input,
      target: center,
      weights,
      intercept,
      clusterId: cluster.id
    };
    cluster.samples.push(gradientSample);
    samples.push(gradientSample);
    const support = cluster.samples.length;
    cluster.weights = cluster.weights.map((value, index) => value + (weights[index] - value) / support);
    cluster.intercept += (intercept - cluster.intercept) / support;
  }

  const survivors = clusters
    .filter(cluster => cluster.samples.length >= minRegionSupport)
    .sort((a, b) => b.samples.length - a.samples.length);
  const clusterToRegion = new Map<number, number>();
  const regions: DeepBlackBoxReluNdRegionSamplerRegion[] = survivors.map((cluster, regionId) => {
    clusterToRegion.set(cluster.id, regionId);
    const fit = fitAffine(cluster.samples, dimension, ridge, cluster.weights, cluster.intercept);
    return {
      id: regionId,
      intercept: fit.intercept,
      weights: fit.weights,
      support: cluster.samples.length,
      centroid: centroid(cluster.samples, dimension),
      fitRmse: fit.rmse
    };
  });
  if (regions.length === 0) throw new Error('ND region sampler recovered no supported affine regions');

  const assignments: DeepBlackBoxReluNdRegionSamplerAssignment[] = [];
  for (const sample of samples) {
    const regionId = clusterToRegion.get(sample.clusterId);
    if (regionId !== undefined) assignments.push({ input: sample.input, regionId, target: sample.target });
  }
  if (assignments.length === 0) throw new Error('ND region sampler retained no region assignments');

  const modelBase = {
    dimension,
    ranges: config.ranges,
    sampleCount,
    samplingMode,
    minRegionSupport,
    retainedSampleCount: assignments.length,
    droppedSampleCount: sampleCount - assignments.length,
    gradientStep,
    querySamples,
    aggregation: aggregationMode,
    trimFraction,
    neighborCount,
    predictionMode,
    localAffineNeighborCount,
    ridge,
    recoveryKind: 'output-equivalent-nd-region-sampler' as const,
    regions,
    assignments,
    regionCount: regions.length
  };
  return {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
}

export function predictDeepBlackBoxReluNdRegionSamplerModel(
  model: DeepBlackBoxReluNdRegionSamplerModel,
  input: number[]
): number {
  if (input.length !== model.dimension) throw new Error('ND region sampler prediction dimension mismatch');
  if (model.predictionMode === 'local-affine-knn') return predictLocalAffineKnn(model, input);
  const neighbors: Array<{ distance: number; regionId: number }> = [];
  for (const assignment of model.assignments) {
    insertNeighbor(
      neighbors,
      { distance: squaredDistance(input, assignment.input), regionId: assignment.regionId },
      model.neighborCount
    );
  }
  if (neighbors.length === 0) throw new Error('ND region sampler model has no assignments');
  const votes = new Map<number, number>();
  for (const neighbor of neighbors) {
    votes.set(neighbor.regionId, (votes.get(neighbor.regionId) ?? 0) + 1 / (neighbor.distance + 1e-12));
  }
  const selectedRegionId = [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
  const region = model.regions[selectedRegionId];
  const value = region.intercept + dot(region.weights, input);
  return Number.isFinite(value) ? value : 0;
}

function predictLocalAffineKnn(model: DeepBlackBoxReluNdRegionSamplerModel, input: number[]): number {
  const neighbors: Array<{ distance: number; assignment: DeepBlackBoxReluNdRegionSamplerAssignment }> = [];
  for (const assignment of model.assignments) {
    const candidate = { distance: squaredDistance(input, assignment.input), assignment };
    let index = 0;
    while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
    if (index >= model.localAffineNeighborCount) continue;
    neighbors.splice(index, 0, candidate);
    if (neighbors.length > model.localAffineNeighborCount) neighbors.pop();
  }
  if (neighbors.length < model.dimension + 1) return predictRegionVoteFallback(model, input);
  const width = model.dimension + 1;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const neighbor of neighbors) {
    const row = [1, ...neighbor.assignment.input];
    const weight = 1 / Math.max(1e-8, neighbor.distance);
    for (let i = 0; i < width; i++) {
      xty[i] += weight * row[i] * neighbor.assignment.target;
      for (let j = 0; j < width; j++) xtx[i][j] += weight * row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += model.ridge;
  try {
    const coefficients = solveSystem(xtx, xty);
    const value = coefficients[0] + dot(coefficients.slice(1), input);
    return Number.isFinite(value) ? value : predictRegionVoteFallback(model, input);
  } catch {
    return predictRegionVoteFallback(model, input);
  }
}

function predictRegionVoteFallback(model: DeepBlackBoxReluNdRegionSamplerModel, input: number[]): number {
  const previousMode = model.predictionMode;
  const fallbackModel = { ...model, predictionMode: 'region-vote' as const };
  const value = predictDeepBlackBoxReluNdRegionSamplerModel(fallbackModel, input);
  return Number.isFinite(value) ? value : previousMode === 'local-affine-knn' ? 0 : value;
}

export function predictDeepBlackBoxReluNdRegionSamplerBatch(
  model: DeepBlackBoxReluNdRegionSamplerModel,
  inputs: number[][]
): number[] {
  return inputs.map(input => predictDeepBlackBoxReluNdRegionSamplerModel(model, input));
}
