export type AsadTerm = {
  normal: number[];
  offset: number;
  readout: number;
  support: number;
  confidence: number;
};

export type AsadDiagnostics = {
  totalOracleCalls: number;
  oracleBudget: number;
  budgetExhausted: boolean;
  stoppedPhase: 'gradient-probe' | 'acquisition' | 'fit' | null;
  callsByPhase: Record<'gradient-probe' | 'fit', number>;
  gradientProbes: number;
  activeSubspaceRank: number;
  activeSubspaceEigenvalues: number[];
  activeSubspaceEnergyRatio: number;
  rankUncertainty: number;
  hypothesisDisagreement: number;
  disagreementQueries: number;
  activeSubspaceQueries: number;
  orthogonalExplorationQueries: number;
  globalExplorationQueries: number;
  strategySwitches: number;
  candidatesRecovered: number;
  candidateUncertainty: number;
  fitResidualRms: number | null;
  fitResidualRelative: number | null;
  fitValidationObservations: number;
  requiredFitValidationObservations: number;
  reconstructionComplete: boolean;
  reasons: string[];
};

export type ActiveSubspaceAdaptiveDiscoveryModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  baseIntercept: number;
  baseWeights: number[];
  terms: AsadTerm[];
  expression: string;
  reconstructionComplete: boolean;
  diagnostics: AsadDiagnostics;
};

export type ActiveSubspaceAdaptiveDiscoveryConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  termCount: number;
  oracleBudget: number;
  randomSeed?: number;
  finiteDifferenceStep?: number;
  maxGradientProbes?: number;
  acquisitionPoolSize?: number;
  activeRankThreshold?: number;
  minimumGradientJump?: number;
  minimumCandidateConfidence?: number;
  maximumRelativeFitResidual?: number;
  ridge?: number;
};

type Observation = {
  point: number[];
  value: number;
  gradient: number[];
};

type Plane = {
  normal: number[];
  offset: number;
  support: number;
  confidence: number;
};

type Eigensystem = {
  values: number[];
  vectors: number[][];
  trace: number;
};

class BudgetExhausted extends Error {
  constructor() {
    super('ASAD oracle budget exhausted');
    this.name = 'BudgetExhausted';
  }
}

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

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = norm(values);
  return length > 1e-14 ? values.map(value => value / length) : values.map(() => 0);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function randomPoint(random: () => number, ranges: Array<[number, number]>, margin: number): number[] {
  return ranges.map(([min, max]) => min + margin + (max - min - 2 * margin) * random());
}

function canonicalize(normalInput: number[], offsetInput: number): { normal: number[]; offset: number } {
  const length = norm(normalInput);
  if (length <= 1e-14) return { normal: normalInput, offset: offsetInput };
  let normal = normalInput.map(value => value / length);
  let offset = offsetInput / length;
  const pivot = normal.findIndex(value => Math.abs(value) > 1e-10);
  if (pivot >= 0 && normal[pivot] < 0) {
    normal = normal.map(value => -value);
    offset = -offset;
  }
  return { normal, offset };
}

function covarianceEigensystem(gradients: number[][]): Eigensystem {
  const dimension = gradients[0]?.length ?? 0;
  if (dimension === 0 || gradients.length < 2) {
    return { values: new Array(dimension).fill(0), vectors: [], trace: 0 };
  }
  const center = Array.from({ length: dimension }, (_, axis) => mean(gradients.map(row => row[axis])));
  const matrix = Array.from({ length: dimension }, () => new Array(dimension).fill(0));
  for (const gradient of gradients) {
    const centered = gradient.map((value, axis) => value - center[axis]);
    for (let row = 0; row < dimension; row++) {
      for (let col = row; col < dimension; col++) {
        matrix[row][col] += centered[row] * centered[col] / gradients.length;
      }
    }
  }
  for (let row = 0; row < dimension; row++) {
    for (let col = 0; col < row; col++) matrix[row][col] = matrix[col][row];
  }

  const eigenvectors: number[][] = Array.from({ length: dimension }, (_, row) =>
    Array.from({ length: dimension }, (_, col) => row === col ? 1 : 0));
  for (let iteration = 0; iteration < dimension * dimension * 12; iteration++) {
    let p = 0;
    let q = 1;
    let largest = 0;
    for (let row = 0; row < dimension; row++) {
      for (let col = row + 1; col < dimension; col++) {
        if (Math.abs(matrix[row][col]) > largest) {
          largest = Math.abs(matrix[row][col]);
          p = row;
          q = col;
        }
      }
    }
    if (largest < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * matrix[p][q], matrix[q][q] - matrix[p][p]);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const app = matrix[p][p];
    const aqq = matrix[q][q];
    const apq = matrix[p][q];
    matrix[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
    matrix[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
    matrix[p][q] = 0;
    matrix[q][p] = 0;
    for (let index = 0; index < dimension; index++) {
      if (index === p || index === q) continue;
      const aip = matrix[index][p];
      const aiq = matrix[index][q];
      matrix[index][p] = cosine * aip - sine * aiq;
      matrix[p][index] = matrix[index][p];
      matrix[index][q] = sine * aip + cosine * aiq;
      matrix[q][index] = matrix[index][q];
    }
    for (let row = 0; row < dimension; row++) {
      const vip = eigenvectors[row][p];
      const viq = eigenvectors[row][q];
      eigenvectors[row][p] = cosine * vip - sine * viq;
      eigenvectors[row][q] = sine * vip + cosine * viq;
    }
  }
  const ordered = Array.from({ length: dimension }, (_, index) => ({
    value: Math.max(0, matrix[index][index]),
    vector: eigenvectors.map(row => row[index])
  })).sort((a, b) => b.value - a.value);
  return {
    values: ordered.map(entry => entry.value),
    vectors: ordered.map(entry => normalize(entry.vector)),
    trace: ordered.reduce((sum, entry) => sum + entry.value, 0)
  };
}

function subtractProjection(vector: number[], basis: number[][]): number[] {
  const result = [...vector];
  for (const direction of basis) {
    const projection = dot(result, direction);
    for (let index = 0; index < result.length; index++) result[index] -= projection * direction[index];
  }
  return result;
}

function estimateRank(eigensystem: Eigensystem, dimension: number, sampleCount: number, threshold: number): {
  rank: number;
  uncertainty: number;
  energyRatio: number;
} {
  if (!eigensystem.values.length || eigensystem.values[0] <= 1e-12) {
    return { rank: 0, uncertainty: 1, energyRatio: 0 };
  }
  const cutoff = Math.max(1e-12, eigensystem.values[0] * threshold);
  const rank = Math.min(dimension, eigensystem.values.filter(value => value >= cutoff).length);
  const boundary = eigensystem.values[Math.max(0, rank - 1)] / Math.max(1e-12, eigensystem.values[0]);
  const spectralUncertainty = Math.max(0, 1 - Math.min(1, Math.abs(Math.log(boundary / threshold)) / Math.log(4)));
  const uncertainty = sampleCount < dimension + 1 ? 1 : spectralUncertainty;
  const selectedEnergy = eigensystem.values.slice(0, rank).reduce((sum, value) => sum + value, 0);
  return {
    rank,
    uncertainty,
    energyRatio: selectedEnergy / Math.max(1e-12, eigensystem.trace)
  };
}

function evaluateGradient(
  point: number[],
  step: number,
  query: (input: number[], phase: 'gradient-probe' | 'fit') => number
): Observation {
  const value = query(point, 'gradient-probe');
  const gradient = new Array(point.length);
  for (let axis = 0; axis < point.length; axis++) {
    const left = [...point];
    const right = [...point];
    left[axis] -= step;
    right[axis] += step;
    gradient[axis] = (query(right, 'gradient-probe') - query(left, 'gradient-probe')) / (2 * step);
  }
  return { point, value, gradient };
}

function affineIntercept(observation: Observation): number {
  return observation.value - dot(observation.gradient, observation.point);
}

function candidatePlanes(
  observations: Observation[],
  existing: Plane[],
  minimumJump: number,
  minimumConfidence: number
): Plane[] {
  if (observations.length < 2) return existing;
  const latestIndex = observations.length - 1;
  const latest = observations[latestIndex];
  const nearbyTransitions = observations.slice(0, latestIndex)
    .map((observation, index) => ({
      observation,
      distance: norm(latest.gradient.map((value, axis) => value - observation.gradient[axis])),
      index
    }))
    .filter(candidate => candidate.distance >= minimumJump)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 4);
  const candidates = [...existing];
  for (const transition of nearbyTransitions) {
    const provisional = canonicalize(
      latest.gradient.map((value, axis) => value - transition.observation.gradient[axis]),
      affineIntercept(latest) - affineIntercept(transition.observation)
    );
    const localObservations = observations
      .map(item => ({ item, distance: Math.abs(dot(provisional.normal, item.point) + provisional.offset) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, Math.min(observations.length, Math.max(8, 2 * provisional.normal.length)))
      .map(entry => entry.item);
    const negative = localObservations.filter(item => dot(provisional.normal, item.point) + provisional.offset < 0);
    const positive = localObservations.filter(item => dot(provisional.normal, item.point) + provisional.offset >= 0);
    if (negative.length < 2 || positive.length < 2) continue;

    const meanGradient = (items: Observation[]): number[] => Array.from(
      { length: provisional.normal.length },
      (_, axis) => mean(items.map(item => item.gradient[axis]))
    );
    const negativeGradient = meanGradient(negative);
    const positiveGradient = meanGradient(positive);
    const jump = positiveGradient.map((value, axis) => value - negativeGradient[axis]);
    const jumpNorm = norm(jump);
    if (jumpNorm < minimumJump) continue;
    const alignment = Math.abs(dot(normalize(jump), provisional.normal));

    const globalMean = meanGradient(localObservations);
    let totalEnergy = 0;
    let withinEnergy = 0;
    for (const item of localObservations) {
      totalEnergy += item.gradient.reduce((sum, value, axis) => sum + (value - globalMean[axis]) ** 2, 0);
    }
    for (const [items, center] of [[negative, negativeGradient], [positive, positiveGradient]] as const) {
      for (const item of items) {
        withinEnergy += item.gradient.reduce((sum, value, axis) => sum + (value - center[axis]) ** 2, 0);
      }
    }
    const purity = totalEnergy <= 1e-12 ? 0 : Math.max(0, 1 - withinEnergy / totalEnergy);
    const localFlatness = 1 / (1 + withinEnergy / (localObservations.length * jumpNorm * jumpNorm));
    const confidence = alignment * purity * localFlatness;
    if (confidence < minimumConfidence) continue;

    const negativeIntercept = mean(negative.map(affineIntercept));
    const positiveIntercept = mean(positive.map(affineIntercept));
    const canonical = canonicalize(jump, positiveIntercept - negativeIntercept);
    if (!Number.isFinite(canonical.offset)) continue;
    const candidate = {
      normal: canonical.normal,
      offset: canonical.offset,
      support: Math.min(negative.length, positive.length),
      confidence
    };
    const duplicate = candidates.find(existingPlane =>
      Math.abs(dot(existingPlane.normal, candidate.normal)) >= 0.995
      && Math.abs(existingPlane.offset - candidate.offset) <= 0.04);
    if (!duplicate) candidates.push(candidate);
    else if (candidate.confidence > duplicate.confidence) Object.assign(duplicate, candidate);
  }
  return candidates
    .sort((a, b) => b.confidence * b.support - a.confidence * a.support)
    .slice(0, Math.max(1, observations[0]?.point.length ?? 1));
}

function orthogonalDirection(random: () => number, basis: number[][], dimension: number): number[] {
  for (let attempt = 0; attempt < 8; attempt++) {
    const direction = normalize(Array.from({ length: dimension }, () => 2 * random() - 1));
    const residual = normalize(subtractProjection(direction, basis));
    if (norm(residual) > 1e-8) return residual;
  }
  for (let axis = 0; axis < dimension; axis++) {
    const direction = new Array(dimension).fill(0);
    direction[axis] = 1;
    const residual = normalize(subtractProjection(direction, basis));
    if (norm(residual) > 1e-8) return residual;
  }
  return new Array(dimension).fill(0);
}

function predictAffine(observation: Observation, point: number[]): number {
  return observation.value + dot(observation.gradient, point.map((value, axis) => value - observation.point[axis]));
}

function selectAcquisition(
  observations: Observation[],
  eigensystem: Eigensystem,
  activeRank: number,
  ranges: Array<[number, number]>,
  random: () => number,
  poolSize: number,
  minimumMargin: number
): { point: number[]; strategy: 'active-subspace' | 'orthogonal-exploration' | 'global-exploration'; disagreement: number } {
  const dimension = ranges.length;
  const activeBasis = eigensystem.vectors.slice(0, activeRank);
  const candidates: Array<{ point: number[]; strategy: 'active-subspace' | 'orthogonal-exploration' | 'global-exploration' }> = [];
  const margin = Math.max(minimumMargin, Math.min(...ranges.map(([min, max]) => (max - min) * 1e-5)));
  const addCandidate = (point: number[], strategy: typeof candidates[number]['strategy']): void => {
    candidates.push({
      point: point.map((value, axis) => Math.max(ranges[axis][0] + margin, Math.min(ranges[axis][1] - margin, value))),
      strategy
    });
  };

  for (let index = 0; index < poolSize; index++) {
    if (activeRank > 0 && activeRank < dimension && index % 3 !== 2) {
      const direction = index % 3 === 0
        ? activeBasis[index % activeBasis.length]
        : orthogonalDirection(random, activeBasis, dimension);
      const anchor = observations[Math.floor(random() * observations.length)].point;
      const scale = Math.sqrt(dimension) * mean(ranges.map(([min, max]) => (max - min) / 2));
      const amount = (2 * random() - 1) * scale;
      addCandidate(anchor.map((value, axis) => value + amount * direction[axis]),
        index % 3 === 0 ? 'active-subspace' : 'orthogonal-exploration');
    } else {
      addCandidate(randomPoint(random, ranges, margin), 'global-exploration');
    }
  }

  const hypotheses = observations.slice(-Math.min(24, observations.length));
  const outputVariance = mean(observations.map(item => (item.value - mean(observations.map(sample => sample.value))) ** 2));
  let best = candidates[0];
  let bestScore = -Infinity;
  let bestDisagreement = 0;
  const domainDiameter = Math.sqrt(ranges.reduce((sum, [min, max]) => sum + (max - min) ** 2, 0));
  for (const candidate of candidates) {
    const predictions = hypotheses.map(hypothesis => predictAffine(hypothesis, candidate.point));
    const predictedMean = mean(predictions);
    const disagreement = Math.sqrt(mean(predictions.map(value => (value - predictedMean) ** 2)))
      / Math.max(1e-8, Math.sqrt(outputVariance));
    const nearestDistance = Math.min(...observations.map(item =>
      norm(candidate.point.map((value, axis) => value - item.point[axis]))));
    const novelty = nearestDistance / Math.max(1e-12, domainDiameter);
    const score = disagreement + 0.12 * novelty;
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
      bestDisagreement = disagreement;
    }
  }
  return { ...best, disagreement: bestDisagreement };
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const size = rhs.length;
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[pivot], augmented[column]] = [augmented[column], augmented[pivot]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) < 1e-12) throw new Error('ASAD coefficient system is singular');
    for (let col = column; col <= size; col++) augmented[column][col] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let col = column; col <= size; col++) augmented[row][col] -= factor * augmented[column][col];
    }
  }
  return augmented.map(row => row[size]);
}

function fitCoefficients(
  observations: Observation[],
  planes: Plane[],
  ridge: number
): number[] {
  const dimension = observations[0].point.length;
  const width = 1 + dimension + planes.length;
  const matrix = Array.from({ length: width }, () => new Array(width).fill(0));
  const vector = new Array(width).fill(0);
  for (const observation of observations) {
    const features = [
      1,
      ...observation.point,
      ...planes.map(plane => Math.max(0, dot(plane.normal, observation.point) + plane.offset))
    ];
    for (let row = 0; row < width; row++) {
      vector[row] += features[row] * observation.value;
      for (let col = 0; col < width; col++) matrix[row][col] += features[row] * features[col];
    }
  }
  for (let index = 1; index < width; index++) matrix[index][index] += ridge;
  return solveLinearSystem(matrix, vector);
}

function buildExpression(model: Omit<ActiveSubspaceAdaptiveDiscoveryModel, 'expression'>): string {
  const terms = [
    model.baseIntercept.toPrecision(8),
    ...model.baseWeights.map((weight, axis) => `${weight.toPrecision(8)} * x${axis}`),
    ...model.terms.map(term =>
      `${term.readout.toPrecision(8)} * max(0, ${term.normal.map((value, axis) => `${value.toPrecision(8)} * x${axis}`).join(' + ')} + ${term.offset.toPrecision(8)})`)
  ];
  return terms.join(' + ');
}

export function decompileBlackBoxReluNdActiveSubspaceAdaptive(
  oracle: (input: number[]) => number,
  config: ActiveSubspaceAdaptiveDiscoveryConfig
): ActiveSubspaceAdaptiveDiscoveryModel {
  const dimension = Math.floor(config.dimension);
  if (!Number.isSafeInteger(dimension) || dimension < 2) throw new Error('ASAD requires dimension >= 2');
  if (config.ranges.length !== dimension || config.ranges.some(([min, max]) => !(max > min))) {
    throw new Error('ASAD requires one valid range per dimension');
  }
  if (!Number.isSafeInteger(config.termCount) || config.termCount < 1) throw new Error('ASAD requires termCount >= 1');
  if (!Number.isSafeInteger(config.oracleBudget) || config.oracleBudget < 0) {
    throw new Error('ASAD requires a non-negative safe-integer oracleBudget');
  }

  const random = makeRandom(config.randomSeed ?? 20261029);
  const step = config.finiteDifferenceStep ?? Math.min(...config.ranges.map(([min, max]) => (max - min) * 1e-4));
  if (!(step > 0) || config.ranges.some(([min, max]) => 2 * step >= max - min)) {
    throw new Error('ASAD finite-difference step must fit inside every input range');
  }
  const maxProbes = Math.max(0, Math.floor(config.maxGradientProbes ?? config.oracleBudget / (2 * dimension + 1)));
  const poolSize = Math.max(8, Math.floor(config.acquisitionPoolSize ?? 48));
  const rankThreshold = Math.min(0.5, Math.max(1e-4, config.activeRankThreshold ?? 0.025));
  const minimumJump = Math.max(1e-12, config.minimumGradientJump ?? 1e-6);
  const minimumConfidence = Math.min(1, Math.max(0, config.minimumCandidateConfidence ?? 0.92));
  const observations: Observation[] = [];
  const reasons: string[] = [];
  let totalOracleCalls = 0;
  let stoppedPhase: AsadDiagnostics['stoppedPhase'] = null;
  let budgetExhausted = false;
  const callsByPhase: AsadDiagnostics['callsByPhase'] = { 'gradient-probe': 0, fit: 0 };
  const query = (input: number[], phase: 'gradient-probe' | 'fit'): number => {
    if (totalOracleCalls >= config.oracleBudget) {
      budgetExhausted = true;
      stoppedPhase ??= phase;
      throw new BudgetExhausted();
    }
    totalOracleCalls++;
    callsByPhase[phase]++;
    return oracle(input);
  };
  const margin = Math.min(...config.ranges.map(([min, max]) => Math.max(step * 1.1, (max - min) * 1e-5)));
  const initialPoints = [
    config.ranges.map(([min, max]) => (min + max) / 2),
    randomPoint(random, config.ranges, margin)
  ];
  let eigensystem: Eigensystem = { values: new Array(dimension).fill(0), vectors: [], trace: 0 };
  let rankEstimate = { rank: 0, uncertainty: 1, energyRatio: 0 };
  let hypothesisDisagreement = 0;
  let disagreementQueries = 0;
  let activeSubspaceQueries = 0;
  let orthogonalExplorationQueries = 0;
  let globalExplorationQueries = 0;
  let strategySwitches = 0;
  let lastStrategy: string | null = null;
  let acquisitionCompleted = false;
  let observedPlanes: Plane[] = [];
  const budgetCanProbe = (): boolean => totalOracleCalls + (2 * dimension + 1) <= config.oracleBudget;

  for (let probeIndex = 0; probeIndex < maxProbes; probeIndex++) {
    if (!budgetCanProbe()) {
      budgetExhausted = config.oracleBudget === totalOracleCalls;
      stoppedPhase = 'gradient-probe';
      break;
    }
    let point: number[];
    let strategy: 'active-subspace' | 'orthogonal-exploration' | 'global-exploration';
    if (observations.length < 2) {
      point = initialPoints[observations.length];
      strategy = 'global-exploration';
    } else {
      const selected = selectAcquisition(
        observations,
        eigensystem,
        rankEstimate.rank,
        config.ranges,
        random,
        poolSize,
        step * 1.1
      );
      if (rankEstimate.uncertainty >= 0.85) {
        point = randomPoint(random, config.ranges, margin);
        strategy = 'global-exploration';
      } else {
        point = selected.point;
        strategy = selected.strategy;
      }
      hypothesisDisagreement = selected.disagreement;
      if (selected.disagreement > 1e-5) disagreementQueries++;
    }
    if (strategy !== lastStrategy && lastStrategy !== null) strategySwitches++;
    lastStrategy = strategy;
    if (strategy === 'active-subspace') activeSubspaceQueries++;
    else if (strategy === 'orthogonal-exploration') orthogonalExplorationQueries++;
    else globalExplorationQueries++;

    try {
      observations.push(evaluateGradient(point, step, query));
    } catch (error) {
      if (!(error instanceof BudgetExhausted)) throw error;
      break;
    }
    eigensystem = covarianceEigensystem(observations.map(observation => observation.gradient));
    rankEstimate = estimateRank(eigensystem, dimension, observations.length, rankThreshold);
    const planes = candidatePlanes(
      observations,
      observedPlanes,
      minimumJump,
      minimumConfidence
    );
    observedPlanes = planes;
    const fitWidth = 1 + dimension + planes.length;
    const fitValidationCount = Math.max(4, Math.ceil(fitWidth * 0.5));
    if (planes.length === config.termCount
      && observations.length >= fitWidth + fitValidationCount
      && rankEstimate.uncertainty < 0.85) {
      acquisitionCompleted = true;
      break;
    }
  }

  if (!acquisitionCompleted && stoppedPhase === null && observations.length < maxProbes && !budgetExhausted) {
    budgetExhausted = true;
    stoppedPhase = 'gradient-probe';
  } else if (!acquisitionCompleted && stoppedPhase === null
    && totalOracleCalls + (2 * dimension + 1) > config.oracleBudget) {
    budgetExhausted = true;
    stoppedPhase = 'gradient-probe';
  }
  if (observations.length < 2) reasons.push('insufficient_gradient_observations_to_separate_hypotheses');
  if (rankEstimate.uncertainty >= 0.85) {
    reasons.push('active_subspace_rank_uncertain_switched_to_orthogonal_or_global_exploration');
    if (strategySwitches === 0 && observations.length > 1) strategySwitches++;
  }
  const planes = observedPlanes.slice(0, config.termCount);
  let coefficients = new Array(1 + dimension + planes.length).fill(0);
  const fitWidth = coefficients.length;
  const requiredFitValidationObservations = Math.max(4, Math.ceil(fitWidth * 0.5));
  const fitValidationObservations = Math.max(0, observations.length - fitWidth);
  const enoughFitSamples = observations.length >= fitWidth;
  let coefficientFitSucceeded = enoughFitSamples;
  if (enoughFitSamples) {
    try {
      coefficients = fitCoefficients(observations, planes, config.ridge ?? 1e-9);
    } catch (error) {
      if (!(error instanceof BudgetExhausted)) {
        coefficientFitSucceeded = false;
        reasons.push(`coefficient_fit_failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }
  const fitResidualRms = coefficientFitSucceeded
    ? Math.sqrt(mean(observations.map(observation => {
      const prediction = coefficients[0]
        + dot(coefficients.slice(1, 1 + dimension), observation.point)
        + planes.reduce((sum, plane, index) =>
          sum + coefficients[1 + dimension + index] * Math.max(0, dot(plane.normal, observation.point) + plane.offset), 0);
      return (observation.value - prediction) ** 2;
    })))
    : null;
  const targetMean = mean(observations.map(observation => observation.value));
  const targetScale = Math.sqrt(mean(observations.map(observation =>
    (observation.value - targetMean) ** 2)));
  const fitResidualRelative = fitResidualRms === null
    ? null
    : fitResidualRms / Math.max(1e-8, targetScale);
  const fitComplete = coefficientFitSucceeded
    && fitValidationObservations >= requiredFitValidationObservations
    && fitResidualRelative !== null
    && fitResidualRelative <= (config.maximumRelativeFitResidual ?? 0.02);
  const candidateUncertainty = planes.length
    ? 1 - mean(planes.map(plane => plane.confidence))
    : 1;
  const reconstructionComplete = planes.length === config.termCount
    && fitComplete
    && candidateUncertainty <= 1 - minimumConfidence
    && stoppedPhase === null;
  if (planes.length < config.termCount) reasons.push(`only_${planes.length}_of_${config.termCount}_effective_hyperplanes_recovered`);
  budgetExhausted ||= totalOracleCalls >= config.oracleBudget;
  if (stoppedPhase !== null) reasons.push(`oracle_budget_exhausted_during_${stoppedPhase}`);
  if (observations.length < coefficients.length) reasons.push('insufficient_observations_for_coefficient_fit');
  if (fitValidationObservations < requiredFitValidationObservations) {
    reasons.push('insufficient_overdetermined_observations_to_validate_fit');
  }
  if (fitResidualRelative !== null && fitResidualRelative > (config.maximumRelativeFitResidual ?? 0.02)) {
    reasons.push('local_affine_model_residual_exceeded_confidence_gate');
  }
  const terms: AsadTerm[] = planes.map((plane, index) => ({
    normal: plane.normal,
    offset: plane.offset,
    readout: coefficientFitSucceeded ? coefficients[1 + dimension + index] : 0,
    support: plane.support,
    confidence: plane.confidence
  }));
  const diagnostics: AsadDiagnostics = {
    totalOracleCalls,
    oracleBudget: config.oracleBudget,
    budgetExhausted,
    stoppedPhase,
    callsByPhase,
    gradientProbes: observations.length,
    activeSubspaceRank: rankEstimate.rank,
    activeSubspaceEigenvalues: eigensystem.values,
    activeSubspaceEnergyRatio: rankEstimate.energyRatio,
    rankUncertainty: rankEstimate.uncertainty,
    hypothesisDisagreement,
    disagreementQueries,
    activeSubspaceQueries,
    orthogonalExplorationQueries,
    globalExplorationQueries,
    strategySwitches,
    candidatesRecovered: planes.length,
    candidateUncertainty,
    fitResidualRms,
    fitResidualRelative,
    fitValidationObservations,
    requiredFitValidationObservations,
    reconstructionComplete,
    reasons
  };
  const modelBase = {
    dimension,
    ranges: config.ranges,
    baseIntercept: coefficientFitSucceeded ? coefficients[0] : 0,
    baseWeights: coefficientFitSucceeded ? coefficients.slice(1, 1 + dimension) : new Array(dimension).fill(0),
    terms,
    reconstructionComplete,
    diagnostics
  };
  return { ...modelBase, expression: buildExpression(modelBase) };
}

export function predictBlackBoxReluNdActiveSubspaceAdaptive(
  model: ActiveSubspaceAdaptiveDiscoveryModel,
  input: number[]
): number {
  if (input.length !== model.dimension) throw new Error('ASAD prediction dimension mismatch');
  let value = model.baseIntercept + dot(model.baseWeights, input);
  for (const term of model.terms) value += term.readout * Math.max(0, dot(term.normal, input) + term.offset);
  return Number.isFinite(value) ? value : 0;
}
