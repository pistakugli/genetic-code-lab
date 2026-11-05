export type BcadAcquisitionStrategy =
  | 'boundary-uncertainty'
  | 'prediction-disagreement'
  | 'gradient-jump';

export type BcadRankStrategy =
  | 'fixed-initial'
  | 'stable-adaptive'
  | 'delayed-reduction'
  | 'full-dimensional';

export type BcadTerm = {
  normal: number[];
  offset: number;
  readout: number;
  support: number;
  confidence: number;
};

export type BcadDiagnostics = {
  totalOracleCalls: number;
  oracleBudget: number;
  budgetExhausted: boolean;
  stoppedReason: 'oracle-budget' | 'probe-limit' | null;
  gradientProbes: number;
  acquisitionStrategy: BcadAcquisitionStrategy;
  rankStrategy: BcadRankStrategy;
  acquisitionQueries: Record<BcadAcquisitionStrategy, number>;
  competingHypothesesForAcquisition: boolean;
  globalExplorationQueries: number;
  activeSubspaceQueries: number;
  boundaryConditionedQueries: number;
  boundaryValidationQueries: number;
  gradientJumpQueries: number;
  estimatedActiveRank: number;
  effectiveQueryRank: number;
  eigenvalues: number[];
  rankStability: number;
  rankMisestimated: null;
  boundaryHypotheses: number;
  maximumBoundaryHypotheses: number;
  boundaryHypothesisConfidences: number[];
  boundaryHypothesisSupport: number[];
  boundaryHypothesisNormals: number[][];
  boundaryHypothesisOffsets: number[];
  boundaryValidationConfidences: number[];
  boundaryValidationSupport: number[];
  boundaryValidationNegativeSupport: number[];
  boundaryValidationPositiveSupport: number[];
  candidateUncertainty: number;
  hypothesesResolved: boolean;
  candidatesRecovered: number;
  fitResidualRelative: number | null;
  reconstructionComplete: boolean;
  reasons: string[];
};

export type BoundaryConditionedActiveDiscoveryModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  baseIntercept: number;
  baseWeights: number[];
  terms: BcadTerm[];
  expression: string;
  reconstructionComplete: boolean;
  diagnostics: BcadDiagnostics;
};

export type BoundaryConditionedActiveDiscoveryConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  termCount: number;
  oracleBudget: number;
  randomSeed?: number;
  finiteDifferenceStep?: number;
  maxGradientProbes?: number;
  acquisitionPoolSize?: number;
  acquisitionStrategy?: BcadAcquisitionStrategy;
  rankStrategy?: BcadRankStrategy;
  competingHypothesesForAcquisition?: boolean;
  minimumGradientJump?: number;
  minimumCandidateConfidence?: number;
  maximumRelativeFitResidual?: number;
  ridge?: number;
};

type Observation = {
  point: number[];
  value: number;
  gradient: number[];
  intercept: number;
};

type BoundaryHypothesis = {
  normal: number[];
  offset: number;
  positiveGradientJump: number[];
  confidence: number;
  support: number;
  discoveredAt: number;
  validationConfidence: number;
  validationSupport: number;
  validationNegativeSupport: number;
  validationPositiveSupport: number;
  validationAnchor: number[] | null;
};

type EigenSystem = {
  values: number[];
  vectors: number[][];
};

type AcquisitionPoint = {
  point: number[];
  source: 'global' | 'active-subspace' | 'boundary';
};

function dot(first: number[], second: number[]): number {
  let sum = 0;
  for (let index = 0; index < first.length; index++) sum += first[index] * second[index];
  return sum;
}

function norm(vector: number[]): number {
  return Math.sqrt(Math.max(0, dot(vector, vector)));
}

function normalize(vector: number[]): number[] {
  const length = norm(vector);
  return length > 1e-14 ? vector.map(value => value / length) : vector.map(() => 0);
}

function mean(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randomPoint(random: () => number, ranges: Array<[number, number]>): number[] {
  return ranges.map(([min, max]) => min + (max - min) * random());
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

function affineIntercept(observation: Observation): number {
  return observation.value - dot(observation.gradient, observation.point);
}

function covarianceEigenSystem(gradients: number[][]): EigenSystem {
  const dimension = gradients[0]?.length ?? 0;
  if (dimension === 0 || gradients.length < 2) {
    return { values: new Array(dimension).fill(0), vectors: [] };
  }
  const center = Array.from({ length: dimension }, (_, axis) =>
    mean(gradients.map(gradient => gradient[axis])));
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

  const vectors: number[][] = Array.from({ length: dimension }, (_, row) =>
    Array.from({ length: dimension }, (_, col) => row === col ? 1 : 0));
  for (let sweep = 0; sweep < 12; sweep++) {
    let largestOffDiagonal = 0;
    for (let p = 0; p < dimension; p++) {
      for (let q = p + 1; q < dimension; q++) {
        const apq = matrix[p][q];
        largestOffDiagonal = Math.max(largestOffDiagonal, Math.abs(apq));
        if (Math.abs(apq) <= 1e-13) continue;
        const angle = 0.5 * Math.atan2(2 * apq, matrix[q][q] - matrix[p][p]);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const app = matrix[p][p];
        const aqq = matrix[q][q];
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
          const vip = vectors[row][p];
          const viq = vectors[row][q];
          vectors[row][p] = cosine * vip - sine * viq;
          vectors[row][q] = sine * vip + cosine * viq;
        }
      }
    }
    if (largestOffDiagonal < 1e-11) break;
  }
  const ordered = Array.from({ length: dimension }, (_, index) => ({
    value: Math.max(0, matrix[index][index]),
    vector: normalize(vectors.map(row => row[index]))
  })).sort((a, b) => b.value - a.value);
  return { values: ordered.map(item => item.value), vectors: ordered.map(item => item.vector) };
}

function estimateRank(eigenSystem: EigenSystem, dimension: number): number {
  if (!eigenSystem.values.length || eigenSystem.values[0] <= 1e-12) return 0;
  const cutoff = Math.max(1e-12, eigenSystem.values[0] * 0.025);
  return Math.min(dimension, eigenSystem.values.filter(value => value >= cutoff).length);
}

function normalizedSpectrum(eigenSystem: EigenSystem, dimension: number): number[] {
  const values = eigenSystem.values.slice(0, dimension);
  const total = values.reduce((sum, value) => sum + value, 0);
  return total <= 1e-12
    ? new Array(dimension).fill(0)
    : Array.from({ length: dimension }, (_, index) => (values[index] ?? 0) / total);
}

function computeRankStability(spectra: number[][]): number {
  if (spectra.length < 2) return 0;
  const recent = spectra.slice(-3);
  const deltas: number[] = [];
  for (let index = 1; index < recent.length; index++) {
    deltas.push(recent[index].reduce((sum, value, axis) =>
      sum + Math.abs(value - recent[index - 1][axis]), 0));
  }
  return Math.max(0, 1 - mean(deltas) / 0.5);
}

function orthogonalize(vector: number[], basis: number[][]): number[] {
  const result = [...vector];
  for (const direction of basis) {
    const projection = dot(result, direction);
    for (let axis = 0; axis < result.length; axis++) result[axis] -= projection * direction[axis];
  }
  return result;
}

function localObservations(
  observations: Observation[],
  normal: number[],
  offset: number,
  limit: number,
  localRadius: number
): Observation[] {
  const ranked = observations
    .map(observation => ({
      observation,
      distance: Math.abs(dot(normal, observation.point) + offset)
    }))
    .sort((first, second) => first.distance - second.distance);
  const withinBand = ranked.filter(entry => entry.distance <= localRadius);
  const selected = withinBand.length >= 4 ? withinBand.slice(0, limit) : ranked.slice(0, Math.min(limit, Math.max(4, withinBand.length)));
  return selected.map(entry => entry.observation);
}

function meanGradient(observations: Observation[], dimension: number): number[] {
  return Array.from({ length: dimension }, (_, axis) =>
    mean(observations.map(observation => observation.gradient[axis])));
}

function makeBoundaryHypothesis(
  observations: Observation[],
  first: Observation,
  second: Observation,
  age: number,
  minimumGradientJump: number,
  localRadius: number
): BoundaryHypothesis | null {
  const rawJump = first.gradient.map((value, axis) => value - second.gradient[axis]);
  if (norm(rawJump) < minimumGradientJump) return null;
  let plane = canonicalize(rawJump, first.intercept - second.intercept);
  let nearby = localObservations(
    observations,
    plane.normal,
    plane.offset,
    Math.max(8, 3 * plane.normal.length),
    localRadius
  );
  let negative = nearby.filter(observation => dot(plane.normal, observation.point) + plane.offset < 0);
  let positive = nearby.filter(observation => dot(plane.normal, observation.point) + plane.offset >= 0);

  if (negative.length >= 2 && positive.length >= 2) {
    const negativeGradient = meanGradient(negative, plane.normal.length);
    const positiveGradient = meanGradient(positive, plane.normal.length);
    const jump = positiveGradient.map((value, axis) => value - negativeGradient[axis]);
    const interceptJump = mean(positive.map(affineIntercept)) - mean(negative.map(affineIntercept));
    if (norm(jump) >= minimumGradientJump) {
      plane = canonicalize(jump, interceptJump);
      nearby = localObservations(
        observations,
        plane.normal,
        plane.offset,
        Math.max(8, 3 * plane.normal.length),
        localRadius
      );
      negative = nearby.filter(observation => dot(plane.normal, observation.point) + plane.offset < 0);
      positive = nearby.filter(observation => dot(plane.normal, observation.point) + plane.offset >= 0);
    }
  }

  if (negative.length === 0 || positive.length === 0) return null;
  const dimension = plane.normal.length;
  const negativeGradient = meanGradient(negative, dimension);
  const positiveGradient = meanGradient(positive, dimension);
  const positiveGradientJump = positiveGradient.map((value, axis) => value - negativeGradient[axis]);
  const jumpNorm = norm(positiveGradientJump);
  if (jumpNorm < minimumGradientJump) return null;
  const alignment = Math.abs(dot(normalize(positiveGradientJump), plane.normal));
  const center = meanGradient(nearby, dimension);
  let totalEnergy = 0;
  let withinEnergy = 0;
  for (const observation of nearby) {
    totalEnergy += observation.gradient.reduce((sum, value, axis) =>
      sum + (value - center[axis]) ** 2, 0);
  }
  for (const [items, clusterCenter] of [[negative, negativeGradient], [positive, positiveGradient]] as const) {
    for (const observation of items) {
      withinEnergy += observation.gradient.reduce((sum, value, axis) =>
        sum + (value - clusterCenter[axis]) ** 2, 0);
    }
  }
  const purity = totalEnergy <= 1e-12 ? 0 : Math.max(0, 1 - withinEnergy / totalEnergy);
  const sideBalance = Math.min(negative.length, positive.length) / Math.max(negative.length, positive.length);
  const sideSupport = Math.min(1, Math.min(negative.length, positive.length) / 2);
  const confidence = alignment * purity * Math.sqrt(sideBalance) * sideSupport;
  return {
    normal: plane.normal,
    offset: plane.offset,
    positiveGradientJump,
    confidence: Math.min(1, Math.max(0, confidence)),
    support: Math.min(negative.length, positive.length),
    discoveredAt: age,
    validationConfidence: 0,
    validationSupport: 0,
    validationNegativeSupport: 0,
    validationPositiveSupport: 0,
    validationAnchor: null
  };
}

function updateBoundaryHypotheses(
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  age: number,
  minimumGradientJump: number,
  maximumHypotheses: number,
  localRadius: number,
  finiteDifferenceStep: number
): BoundaryHypothesis[] {
  if (observations.length < 2) return hypotheses;
  const latest = observations[observations.length - 1];
  const nearestTransitions = observations.slice(0, -1)
    .map(observation => ({
      observation,
      jump: norm(latest.gradient.map((value, axis) => value - observation.gradient[axis]))
    }))
    .filter(candidate => candidate.jump >= minimumGradientJump)
    .sort((first, second) => second.jump - first.jump)
    .slice(0, 8);
  const candidates = [...hypotheses];
  for (const transition of nearestTransitions) {
    const candidate = makeBoundaryHypothesis(
      observations,
      latest,
      transition.observation,
      age,
      minimumGradientJump,
      localRadius
    );
    if (!candidate || candidate.confidence <= 0.05) continue;
    const duplicate = candidates.find(existing =>
      Math.abs(dot(existing.normal, candidate.normal)) >= 0.995
      && Math.abs(existing.offset - candidate.offset) <= 0.025);
    if (!duplicate) {
      candidates.push(candidate);
    } else if (candidate.confidence * candidate.support > duplicate.confidence * duplicate.support) {
      duplicate.positiveGradientJump = candidate.positiveGradientJump;
      duplicate.support = candidate.support;
      duplicate.confidence = candidate.confidence;
    } else {
      duplicate.support = Math.max(duplicate.support, candidate.support);
      duplicate.confidence = Math.max(duplicate.confidence, candidate.confidence);
    }
  }
  const ranked = candidates
    .sort((first, second) =>
      second.confidence * Math.sqrt(second.support) - first.confidence * Math.sqrt(first.support))
    .slice(0, maximumHypotheses);
  return ranked.map(hypothesis => {
    const stencilMargin = finiteDifferenceStep
      * hypothesis.normal.reduce((sum, value) => sum + Math.abs(value), 0) * 1.1;
    const validationSamples = observations.slice(hypothesis.discoveredAt).filter(observation => {
      const distance = Math.abs(dot(hypothesis.normal, observation.point) + hypothesis.offset);
      return distance <= localRadius && distance > stencilMargin;
    });
    const negative = validationSamples.filter(observation =>
      dot(hypothesis.normal, observation.point) + hypothesis.offset < 0);
    const positive = validationSamples.filter(observation =>
      dot(hypothesis.normal, observation.point) + hypothesis.offset >= 0);
    const validationSupport = Math.min(negative.length, positive.length);
    if (validationSupport < 1) {
      return {
        ...hypothesis,
        validationConfidence: 0,
        validationSupport,
        validationNegativeSupport: negative.length,
        validationPositiveSupport: positive.length
      };
    }
    const negativeGradient = meanGradient(negative, hypothesis.normal.length);
    const positiveGradient = meanGradient(positive, hypothesis.normal.length);
    const validationJump = positiveGradient.map((value, axis) => value - negativeGradient[axis]);
    const alignment = Math.abs(dot(normalize(validationJump), hypothesis.normal));
    const center = meanGradient(validationSamples, hypothesis.normal.length);
    let totalEnergy = 0;
    let withinEnergy = 0;
    for (const observation of validationSamples) {
      totalEnergy += observation.gradient.reduce((sum, value, axis) =>
        sum + (value - center[axis]) ** 2, 0);
    }
    for (const [items, clusterCenter] of [[negative, negativeGradient], [positive, positiveGradient]] as const) {
      for (const observation of items) {
        withinEnergy += observation.gradient.reduce((sum, value, axis) =>
          sum + (value - clusterCenter[axis]) ** 2, 0);
      }
    }
    const purity = totalEnergy <= 1e-12 ? 0 : Math.max(0, 1 - withinEnergy / totalEnergy);
    const sideBalance = Math.min(negative.length, positive.length) / Math.max(negative.length, positive.length);
    const supportFactor = Math.min(1, validationSupport);
    return {
      ...hypothesis,
      validationConfidence: Math.min(1, alignment * purity * Math.sqrt(sideBalance) * supportFactor),
      validationSupport,
      validationNegativeSupport: negative.length,
      validationPositiveSupport: positive.length
    };
  });
}

function hypothesisUncertainty(hypotheses: BoundaryHypothesis[], termCount: number): number {
  const supported = hypotheses.filter(hypothesis =>
    hypothesis.validationConfidence >= 0.55 && hypothesis.validationSupport >= 1);
  const unresolved = hypotheses.filter(hypothesis =>
    hypothesis.confidence >= 0.35 && hypothesis.validationSupport < 1).length;
  const countMismatch = Math.min(1, Math.abs(supported.length - termCount) / Math.max(1, termCount));
  const confidencePenalty = supported.length
    ? mean(supported.map(hypothesis => 1 - hypothesis.validationConfidence))
    : 1;
  const unresolvedPenalty = Math.min(1, unresolved / Math.max(1, termCount));
  return Math.min(1, 0.5 * countMismatch + 0.25 * confidencePenalty + 0.25 * unresolvedPenalty);
}

function projectToBoundary(
  anchor: number[],
  hypothesis: BoundaryHypothesis,
  ranges: Array<[number, number]>
): number[] | null {
  let minimumOffset = -Infinity;
  let maximumOffset = Infinity;
  for (let axis = 0; axis < anchor.length; axis++) {
    const component = hypothesis.normal[axis];
    if (Math.abs(component) <= 1e-12) continue;
    const first = (ranges[axis][0] - anchor[axis]) / component;
    const second = (ranges[axis][1] - anchor[axis]) / component;
    minimumOffset = Math.max(minimumOffset, Math.min(first, second));
    maximumOffset = Math.min(maximumOffset, Math.max(first, second));
  }
  if (minimumOffset > maximumOffset) return null;
  const desiredOffset = -(dot(hypothesis.normal, anchor) + hypothesis.offset);
  const alongNormal = Math.max(minimumOffset, Math.min(maximumOffset, desiredOffset));
  const projected = anchor.map((value, axis) => value + alongNormal * hypothesis.normal[axis]);
  return projected.every((value, axis) =>
    value >= ranges[axis][0] - 1e-10 && value <= ranges[axis][1] + 1e-10)
    ? projected
    : null;
}

function boundaryValidationPoint(
  hypothesis: BoundaryHypothesis,
  side: 'negative' | 'positive',
  ranges: Array<[number, number]>,
  random: () => number,
  finiteDifferenceStep: number
): number[] | null {
  const normalWidth = finiteDifferenceStep
    * hypothesis.normal.reduce((sum, value) => sum + Math.abs(value), 0);
  const offset = Math.max(2.5 * normalWidth, finiteDifferenceStep * 1.5);
  const sign = side === 'negative' ? -1 : 1;
  for (let attempt = 0; attempt < (hypothesis.validationAnchor ? 1 : 8); attempt++) {
    const foot = hypothesis.validationAnchor
      ?? projectToBoundary(randomPoint(random, ranges), hypothesis, ranges);
    if (!foot) continue;
    const bothSidesFit = [-1, 1].every(direction => foot.every((value, axis) => {
      const pointValue = value + direction * offset * hypothesis.normal[axis];
      return pointValue >= ranges[axis][0] + finiteDifferenceStep
        && pointValue <= ranges[axis][1] - finiteDifferenceStep;
    }));
    if (!bothSidesFit) {
      hypothesis.validationAnchor = null;
      continue;
    }
    hypothesis.validationAnchor = foot;
    return foot.map((value, axis) => value + sign * offset * hypothesis.normal[axis]);
  }
  return null;
}

function clampPoint(point: number[], ranges: Array<[number, number]>, margin: number): number[] {
  return point.map((value, axis) =>
    Math.max(ranges[axis][0] + margin, Math.min(ranges[axis][1] - margin, value)));
}

function buildAcquisitionPool(
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  eigenSystem: EigenSystem,
  effectiveRank: number,
  ranges: Array<[number, number]>,
  random: () => number,
  poolSize: number,
  finiteDifferenceStep: number
): AcquisitionPoint[] {
  const dimension = ranges.length;
  const margin = Math.min(...ranges.map(([min, max]) => Math.max(finiteDifferenceStep * 1.1, (max - min) * 1e-5)));
  const candidates: AcquisitionPoint[] = [];
  const add = (point: number[], source: AcquisitionPoint['source']): void => {
    candidates.push({ point: clampPoint(point, ranges, margin), source });
  };
  const globalCount = Math.max(4, Math.floor(poolSize / 4));
  for (let index = 0; index < globalCount; index++) add(randomPoint(random, ranges), 'global');

  const basis = eigenSystem.vectors.slice(0, effectiveRank);
  for (let index = 0; index < poolSize - globalCount; index++) {
    if (hypotheses.length > 0 && index % 3 !== 2) {
      const hypothesis = hypotheses[index % Math.min(hypotheses.length, 8)];
      const anchor = randomPoint(random, ranges);
      let tangent = Array.from({ length: dimension }, () => 2 * random() - 1);
      if (basis.length > 0 && effectiveRank < dimension) {
        const active = basis[Math.floor(random() * basis.length)];
        tangent = active.map((value, axis) => value - dot(active, hypothesis.normal) * hypothesis.normal[axis]);
      } else {
        tangent = tangent.map((value, axis) => value - dot(tangent, hypothesis.normal) * hypothesis.normal[axis]);
      }
      tangent = normalize(tangent);
      const tangentScale = (0.025 + 0.09 * random()) * Math.sqrt(dimension);
      const tangentAnchor = anchor.map((value, axis) => value + tangentScale * tangent[axis]);
      const foot = projectToBoundary(tangentAnchor, hypothesis, ranges);
      if (!foot) {
        add(anchor, 'global');
        continue;
      }
      const crossingWidth = Math.max(
        finiteDifferenceStep * Math.max(1, hypothesis.normal.reduce((sum, value) => sum + Math.abs(value), 0)),
        1e-5
      );
      const offsetChoices = [-4, -2, -1, 0, 1, 2, 4];
      const normalOffset = offsetChoices[index % offsetChoices.length] * crossingWidth;
      add(foot.map((value, axis) => value + normalOffset * hypothesis.normal[axis]), 'boundary');
    } else if (basis.length > 0 && effectiveRank < dimension) {
      const direction = basis[Math.floor(random() * basis.length)];
      const anchor = observations[Math.floor(random() * observations.length)].point;
      const amount = (2 * random() - 1) * Math.sqrt(dimension);
      add(anchor.map((value, axis) => value + amount * direction[axis]), 'active-subspace');
    } else {
      add(randomPoint(random, ranges), 'global');
    }
  }
  return candidates;
}

function predictAffine(observation: Observation, point: number[]): number {
  return observation.value + dot(
    observation.gradient,
    point.map((value, axis) => value - observation.point[axis])
  );
}

function nearestObservation(observations: Observation[], point: number[]): Observation {
  return observations.reduce((nearest, observation) =>
    norm(point.map((value, axis) => value - observation.point[axis]))
      < norm(point.map((value, axis) => value - nearest.point[axis]))
      ? observation
      : nearest);
}

function gradientPredictionForHypothesis(
  observation: Observation,
  point: number[],
  hypothesis: BoundaryHypothesis
): number[] {
  const previousSide = dot(hypothesis.normal, observation.point) + hypothesis.offset >= 0 ? 1 : 0;
  const nextSide = dot(hypothesis.normal, point) + hypothesis.offset >= 0 ? 1 : 0;
  return observation.gradient.map((value, axis) =>
    value + (nextSide - previousSide) * hypothesis.positiveGradientJump[axis]);
}

function candidateScore(
  candidate: AcquisitionPoint,
  strategy: BcadAcquisitionStrategy,
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  ranges: Array<[number, number]>,
  finiteDifferenceStep: number
): number {
  if (observations.length === 0) return 0;
  const nearest = nearestObservation(observations, candidate.point);
  const domainDiameter = Math.sqrt(ranges.reduce((sum, [min, max]) => sum + (max - min) ** 2, 0));
  const nearestDistance = norm(candidate.point.map((value, axis) => value - nearest.point[axis]));
  const novelty = nearestDistance / Math.max(1e-12, domainDiameter);
  const queryCost = 2 * ranges.length + 1;

  if (strategy === 'prediction-disagreement') {
    const predicted = observations.slice(-Math.min(24, observations.length))
      .map(observation => predictAffine(observation, candidate.point));
    const center = mean(predicted);
    const outputMean = mean(observations.map(observation => observation.value));
    const outputScale = Math.sqrt(mean(observations.map(observation => (observation.value - outputMean) ** 2)));
    const disagreement = Math.sqrt(mean(predicted.map(value => (value - center) ** 2)))
      / Math.max(1e-8, outputScale);
    return (disagreement + 0.12 * novelty) / queryCost;
  }

  if (hypotheses.length === 0) return novelty / queryCost;
  const predictions = hypotheses.map(hypothesis => ({
    gradient: gradientPredictionForHypothesis(nearest, candidate.point, hypothesis),
    confidence: Math.max(0.05, hypothesis.validationConfidence || hypothesis.confidence * 0.25),
    jump: norm(hypothesis.positiveGradientJump),
    distance: Math.abs(dot(hypothesis.normal, candidate.point) + hypothesis.offset),
    normal: hypothesis.normal
  }));
  const totalWeight = predictions.reduce((sum, prediction) => sum + prediction.confidence, 0);
  const predictedMean = predictions[0].gradient.map((_, axis) =>
    predictions.reduce((sum, prediction) =>
      sum + prediction.confidence * prediction.gradient[axis], 0) / Math.max(1e-12, totalWeight));
  const disagreement = Math.sqrt(mean(predictions.map(prediction =>
    prediction.gradient.reduce((sum, value, axis) =>
      sum + prediction.confidence * (value - predictedMean[axis]) ** 2, 0) / Math.max(1e-12, totalWeight))));
  const gradientScale = Math.max(1e-8, mean(predictions.map(prediction => prediction.jump)));
  const crossingPotential = Math.max(...predictions.map(prediction => {
    const width = Math.max(
      finiteDifferenceStep * Math.max(1, prediction.normal.reduce((sum, value) => sum + Math.abs(value), 0)),
      1e-5
    );
    return prediction.confidence * prediction.jump
      * Math.max(0, 1 - prediction.distance / (2 * width));
  })) / gradientScale;
  const boundaryInformation = disagreement / gradientScale + crossingPotential;
  if (strategy === 'gradient-jump') {
    return (crossingPotential + 0.08 * novelty) / queryCost;
  }
  return (boundaryInformation + 0.08 * novelty) / queryCost;
}

function selectAcquisition(
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  eigenSystem: EigenSystem,
  effectiveRank: number,
  ranges: Array<[number, number]>,
  random: () => number,
  poolSize: number,
  finiteDifferenceStep: number,
  strategy: BcadAcquisitionStrategy
): AcquisitionPoint {
  const candidates = buildAcquisitionPool(
    observations,
    hypotheses,
    eigenSystem,
    effectiveRank,
    ranges,
    random,
    poolSize,
    finiteDifferenceStep
  );
  let best = candidates[0];
  let bestScore = -Infinity;
  for (const candidate of candidates) {
    const score = candidateScore(candidate, strategy, observations, hypotheses, ranges, finiteDifferenceStep);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return best;
}

function evaluateGradient(
  point: number[],
  step: number,
  ranges: Array<[number, number]>,
  query: (input: number[]) => number
): Observation {
  const value = query(point);
  const gradient = new Array(point.length);
  for (let axis = 0; axis < point.length; axis++) {
    const left = [...point];
    const right = [...point];
    left[axis] = Math.max(ranges[axis][0], left[axis] - step);
    right[axis] = Math.min(ranges[axis][1], right[axis] + step);
    const width = right[axis] - left[axis];
    if (width <= 0) throw new Error('BCAD finite-difference probe has no width inside the configured range');
    gradient[axis] = (query(right) - query(left)) / width;
  }
  const observation = { point, value, gradient, intercept: 0 };
  observation.intercept = affineIntercept(observation);
  return observation;
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
    if (Math.abs(divisor) < 1e-12) throw new Error('BCAD coefficient system is singular');
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
  hypotheses: BoundaryHypothesis[],
  ridge: number
): number[] {
  const dimension = observations[0]?.point.length ?? 0;
  const width = 1 + dimension + hypotheses.length;
  const matrix = Array.from({ length: width }, () => new Array(width).fill(0));
  const vector = new Array(width).fill(0);
  for (const observation of observations) {
    const features = [
      1,
      ...observation.point,
      ...hypotheses.map(hypothesis =>
        Math.max(0, dot(hypothesis.normal, observation.point) + hypothesis.offset))
    ];
    for (let row = 0; row < width; row++) {
      vector[row] += features[row] * observation.value;
      for (let col = 0; col < width; col++) matrix[row][col] += features[row] * features[col];
    }
  }
  for (let index = 1; index < width; index++) matrix[index][index] += ridge;
  return solveLinearSystem(matrix, vector);
}

function prediction(
  coefficients: number[],
  point: number[],
  hypotheses: BoundaryHypothesis[]
): number {
  return coefficients[0]
    + dot(coefficients.slice(1, point.length + 1), point)
    + hypotheses.reduce((sum, hypothesis, index) =>
      sum + coefficients[point.length + 1 + index]
        * Math.max(0, dot(hypothesis.normal, point) + hypothesis.offset), 0);
}

function relativeFitResidual(
  observations: Observation[],
  coefficients: number[],
  hypotheses: BoundaryHypothesis[]
): number {
  const residual = Math.sqrt(mean(observations.map(observation =>
    (observation.value - prediction(coefficients, observation.point, hypotheses)) ** 2)));
  const targetMean = mean(observations.map(observation => observation.value));
  const targetScale = Math.sqrt(mean(observations.map(observation =>
    (observation.value - targetMean) ** 2)));
  return residual / Math.max(1e-8, targetScale);
}

function buildExpression(model: Omit<BoundaryConditionedActiveDiscoveryModel, 'expression'>): string {
  return [
    model.baseIntercept.toPrecision(8),
    ...model.baseWeights.map((weight, axis) => `${weight.toPrecision(8)} * x${axis}`),
    ...model.terms.map(term =>
      `${term.readout.toPrecision(8)} * max(0, ${term.normal.map((value, axis) =>
        `${value.toPrecision(8)} * x${axis}`).join(' + ')} + ${term.offset.toPrecision(8)})`)
  ].join(' + ');
}

export function decompileBlackBoxReluNdBoundaryConditionedActiveDiscovery(
  oracle: (input: number[]) => number,
  config: BoundaryConditionedActiveDiscoveryConfig
): BoundaryConditionedActiveDiscoveryModel {
  const dimension = Math.floor(config.dimension);
  if (!Number.isSafeInteger(dimension) || dimension < 2) throw new Error('BCAD requires dimension >= 2');
  if (config.ranges.length !== dimension || config.ranges.some(([min, max]) => !(max > min))) {
    throw new Error('BCAD requires one valid range per dimension');
  }
  if (!Number.isSafeInteger(config.termCount) || config.termCount < 1) throw new Error('BCAD requires termCount >= 1');
  if (!Number.isSafeInteger(config.oracleBudget) || config.oracleBudget < 0) {
    throw new Error('BCAD requires a non-negative safe-integer oracleBudget');
  }
  const acquisitionStrategy = config.acquisitionStrategy ?? 'boundary-uncertainty';
  const rankStrategy = config.rankStrategy ?? 'delayed-reduction';
  const random = makeRandom(config.randomSeed ?? 20261029);
  const step = config.finiteDifferenceStep
    ?? Math.min(...config.ranges.map(([min, max]) => (max - min) * 1e-4));
  if (!(step > 0) || config.ranges.some(([min, max]) => 2 * step >= max - min)) {
    throw new Error('BCAD finite-difference step must fit inside every input range');
  }
  const costPerGradientProbe = 2 * dimension + 1;
  const budgetProbeLimit = Math.floor(config.oracleBudget / costPerGradientProbe);
  const maxProbes = Math.min(
    budgetProbeLimit,
    Math.max(0, Math.floor(config.maxGradientProbes ?? Math.min(96, budgetProbeLimit)))
  );
  const poolSize = Math.max(8, Math.floor(config.acquisitionPoolSize ?? 48));
  const minimumGradientJump = Math.max(1e-12, config.minimumGradientJump ?? 1e-6);
  const localRadius = 0.025 * Math.sqrt(mean(config.ranges.map(([min, max]) => (max - min) ** 2)));
  const minimumCandidateConfidence = Math.min(1, Math.max(0, config.minimumCandidateConfidence ?? 0.55));
  const competingHypothesesForAcquisition = config.competingHypothesesForAcquisition ?? true;
  const maximumHypotheses = Math.max(4, 4 * config.termCount);
  const observations: Observation[] = [];
  let hypotheses: BoundaryHypothesis[] = [];
  let totalOracleCalls = 0;
  let stoppedReason: BcadDiagnostics['stoppedReason'] = null;
  let budgetExhausted = false;
  let eigenSystem: EigenSystem = { values: new Array(dimension).fill(0), vectors: [] };
  let estimatedActiveRank = 0;
  let fixedInitialRank: number | null = null;
  let effectiveQueryRank = dimension;
  let rankStability = 0;
  const rankHistory: number[] = [];
  const spectrumHistory: number[][] = [];
  const acquisitionQueries: BcadDiagnostics['acquisitionQueries'] = {
    'boundary-uncertainty': 0,
    'prediction-disagreement': 0,
    'gradient-jump': 0
  };
  let globalExplorationQueries = 0;
  let activeSubspaceQueries = 0;
  let boundaryConditionedQueries = 0;
  let boundaryValidationQueries = 0;
  let gradientJumpQueries = 0;
  const reasons: string[] = [];

  const query = (point: number[]): number => {
    if (totalOracleCalls >= config.oracleBudget) {
      budgetExhausted = true;
      stoppedReason = 'oracle-budget';
      throw new Error('BCAD oracle budget exhausted before a query');
    }
    totalOracleCalls++;
    const value = oracle(point);
    if (!Number.isFinite(value)) throw new Error('BCAD oracle returned a non-finite value');
    return value;
  };
  const updateRank = (probeCount: number, force = false): void => {
    if (probeCount < 2 || (!force && probeCount % 4 !== 0 && probeCount !== maxProbes)) return;
    eigenSystem = covarianceEigenSystem(observations.map(observation => observation.gradient));
    estimatedActiveRank = estimateRank(eigenSystem, dimension);
    rankHistory.push(estimatedActiveRank);
    spectrumHistory.push(normalizedSpectrum(eigenSystem, dimension));
    rankStability = computeRankStability(spectrumHistory);
    if (fixedInitialRank === null && probeCount >= Math.min(8, dimension + 1)) {
      fixedInitialRank = estimatedActiveRank;
    }
    if (rankStrategy === 'full-dimensional') {
      effectiveQueryRank = dimension;
    } else if (rankStrategy === 'fixed-initial') {
      effectiveQueryRank = fixedInitialRank ?? dimension;
    } else if (rankStrategy === 'stable-adaptive') {
      const lastRanks = rankHistory.slice(-3);
      const rankIsStable = lastRanks.length === 3 && lastRanks.every(rank => rank === lastRanks[0]);
      effectiveQueryRank = rankIsStable && rankStability >= 0.75
        ? Math.max(1, estimatedActiveRank)
        : dimension;
    } else {
      const uncertainty = hypothesisUncertainty(hypotheses, config.termCount);
      const supportedCount = hypotheses.filter(candidate =>
        candidate.validationConfidence >= minimumCandidateConfidence && candidate.validationSupport >= 1).length;
      const mayReduce = supportedCount >= config.termCount && uncertainty <= 0.25;
      effectiveQueryRank = mayReduce && estimatedActiveRank > 0
        ? estimatedActiveRank
        : dimension;
    }
    if (effectiveQueryRank <= 0 || effectiveQueryRank > dimension) effectiveQueryRank = dimension;
  };

  for (let probeIndex = 0; probeIndex < maxProbes; probeIndex++) {
    if (totalOracleCalls + costPerGradientProbe > config.oracleBudget) {
      budgetExhausted = true;
      stoppedReason = 'oracle-budget';
      break;
    }
    let acquisition: AcquisitionPoint;
    if (observations.length < 2) {
      acquisition = {
        point: observations.length === 0
          ? config.ranges.map(([min, max]) => (min + max) / 2)
          : randomPoint(random, config.ranges),
        source: 'global'
      };
    } else {
      const validationTargets = hypotheses
        .filter(candidate => candidate.confidence >= 0.35 && candidate.validationSupport < 1)
        .sort((first, second) =>
          second.confidence * (2 - second.validationSupport)
          - first.confidence * (2 - first.validationSupport));
      const validationTarget = validationTargets[0];
      const validationPoint = validationTarget
        ? boundaryValidationPoint(
          validationTarget,
          validationTarget.validationNegativeSupport <= validationTarget.validationPositiveSupport
            ? 'negative'
            : 'positive',
          config.ranges,
          random,
          step
        )
        : null;
      const supportedCount = hypotheses.filter(candidate =>
        candidate.validationConfidence >= minimumCandidateConfidence && candidate.validationSupport >= 1).length;
      const preferGlobalExploration = supportedCount < config.termCount && probeIndex % 4 === 3;
      acquisition = validationPoint
        ? { point: validationPoint, source: 'boundary' }
        : preferGlobalExploration
          ? { point: randomPoint(random, config.ranges), source: 'global' }
        : selectAcquisition(
          observations,
          competingHypothesesForAcquisition ? hypotheses : hypotheses.slice(0, 1),
          eigenSystem,
          effectiveQueryRank,
          config.ranges,
          random,
          poolSize,
          step,
          acquisitionStrategy
        );
      if (validationPoint) boundaryValidationQueries++;
      acquisitionQueries[acquisitionStrategy]++;
    }
    if (acquisition.source === 'global') globalExplorationQueries++;
    if (acquisition.source === 'active-subspace') activeSubspaceQueries++;
    if (acquisition.source === 'boundary') {
      boundaryConditionedQueries++;
      if (acquisitionStrategy === 'gradient-jump') gradientJumpQueries++;
    }

    observations.push(evaluateGradient(acquisition.point, step, config.ranges, query));
    hypotheses = updateBoundaryHypotheses(
      observations,
      hypotheses,
      observations.length,
      minimumGradientJump,
      maximumHypotheses,
      localRadius,
      step
    );
    updateRank(observations.length);

    const supported = hypotheses.filter(candidate =>
      candidate.validationConfidence >= minimumCandidateConfidence && candidate.validationSupport >= 1);
    const uncertainty = hypothesisUncertainty(hypotheses, config.termCount);
    const rankEnough = rankStrategy === 'full-dimensional'
      || (effectiveQueryRank < dimension && rankStability >= 0.75)
      || (estimatedActiveRank === dimension && rankStability >= 0.75);
    const fitWidth = 1 + dimension + supported.length;
    const fitReady = observations.length >= fitWidth + Math.max(4, Math.ceil(fitWidth * 0.5));
    if (supported.length === config.termCount && uncertainty <= 0.12 && rankEnough && fitReady) {
      try {
        const provisionalFit = fitCoefficients(observations, supported, config.ridge ?? 1e-9);
        if (relativeFitResidual(observations, provisionalFit, supported)
          <= (config.maximumRelativeFitResidual ?? 0.02)) break;
      } catch (error) {
        if (!(error instanceof Error) || error.message !== 'BCAD coefficient system is singular') throw error;
      }
    }
  }

  updateRank(observations.length, true);
  if (stoppedReason === null && totalOracleCalls + costPerGradientProbe > config.oracleBudget) {
    budgetExhausted = true;
    stoppedReason = 'oracle-budget';
  } else if (stoppedReason === null && observations.length >= maxProbes) {
    stoppedReason = 'probe-limit';
  }
  const finalHypotheses = hypotheses
    .filter(candidate =>
      candidate.validationConfidence >= minimumCandidateConfidence && candidate.validationSupport >= 1)
    .slice(0, config.termCount);
  const candidateUncertainty = hypothesisUncertainty(hypotheses, config.termCount);
  const hypothesesResolved = finalHypotheses.length === config.termCount && candidateUncertainty <= 0.25;
  const width = 1 + dimension + finalHypotheses.length;
  const enoughFitSamples = observations.length >= width + Math.max(4, Math.ceil(width * 0.5));
  let coefficients = new Array(width).fill(0);
  let fitResidualRelative: number | null = null;
  if (enoughFitSamples) {
    try {
      coefficients = fitCoefficients(observations, finalHypotheses, config.ridge ?? 1e-9);
      fitResidualRelative = relativeFitResidual(observations, coefficients, finalHypotheses);
    } catch (error) {
      if (!(error instanceof Error)) throw error;
      reasons.push(`coefficient_fit_failed: ${error.message}`);
    }
  } else {
    reasons.push('insufficient_overdetermined_observations_for_coefficient_fit');
  }
  if (finalHypotheses.length < config.termCount) {
    reasons.push(`only_${finalHypotheses.length}_of_${config.termCount}_boundary_hypotheses_supported`);
  }
  if (!hypothesesResolved) reasons.push('competing_boundary_hypotheses_remain_unresolved');
  if (stoppedReason === 'oracle-budget') reasons.push('oracle_budget_exhausted_before_complete_gradient_probe');
  if (rankStability < 0.75) reasons.push('active_subspace_spectrum_not_stable');
  if (fitResidualRelative !== null && fitResidualRelative > (config.maximumRelativeFitResidual ?? 0.02)) {
    reasons.push('training_fit_residual_exceeded_confidence_gate');
  }
  const reconstructionComplete = hypothesesResolved
    && enoughFitSamples
    && fitResidualRelative !== null
    && fitResidualRelative <= (config.maximumRelativeFitResidual ?? 0.02)
    && stoppedReason !== 'oracle-budget';
  const terms: BcadTerm[] = finalHypotheses.map((hypothesis, index) => ({
    normal: hypothesis.normal,
    offset: hypothesis.offset,
    readout: enoughFitSamples && fitResidualRelative !== null ? coefficients[dimension + 1 + index] : 0,
    support: hypothesis.support,
    confidence: hypothesis.confidence
  }));
  const diagnostics: BcadDiagnostics = {
    totalOracleCalls,
    oracleBudget: config.oracleBudget,
    budgetExhausted,
    stoppedReason,
    gradientProbes: observations.length,
    acquisitionStrategy,
    rankStrategy,
    acquisitionQueries,
    competingHypothesesForAcquisition,
    globalExplorationQueries,
    activeSubspaceQueries,
    boundaryConditionedQueries,
    boundaryValidationQueries,
    gradientJumpQueries,
    estimatedActiveRank,
    effectiveQueryRank,
    eigenvalues: eigenSystem.values,
    rankStability,
    rankMisestimated: null,
    boundaryHypotheses: hypotheses.length,
    maximumBoundaryHypotheses: maximumHypotheses,
    boundaryHypothesisConfidences: hypotheses.map(hypothesis => hypothesis.confidence),
    boundaryHypothesisSupport: hypotheses.map(hypothesis => hypothesis.support),
    boundaryHypothesisNormals: hypotheses.map(hypothesis => hypothesis.normal),
    boundaryHypothesisOffsets: hypotheses.map(hypothesis => hypothesis.offset),
    boundaryValidationConfidences: hypotheses.map(hypothesis => hypothesis.validationConfidence),
    boundaryValidationSupport: hypotheses.map(hypothesis => hypothesis.validationSupport),
    boundaryValidationNegativeSupport: hypotheses.map(hypothesis => hypothesis.validationNegativeSupport),
    boundaryValidationPositiveSupport: hypotheses.map(hypothesis => hypothesis.validationPositiveSupport),
    candidateUncertainty,
    hypothesesResolved,
    candidatesRecovered: terms.length,
    fitResidualRelative,
    reconstructionComplete,
    reasons
  };
  const modelBase = {
    dimension,
    ranges: config.ranges,
    baseIntercept: enoughFitSamples && fitResidualRelative !== null ? coefficients[0] : 0,
    baseWeights: enoughFitSamples && fitResidualRelative !== null
      ? coefficients.slice(1, dimension + 1)
      : new Array(dimension).fill(0),
    terms,
    reconstructionComplete,
    diagnostics
  };
  return { ...modelBase, expression: buildExpression(modelBase) };
}

export function predictBlackBoxReluNdBoundaryConditionedActiveDiscovery(
  model: BoundaryConditionedActiveDiscoveryModel,
  input: number[]
): number {
  if (input.length !== model.dimension) throw new Error('BCAD prediction dimension mismatch');
  let value = model.baseIntercept + dot(model.baseWeights, input);
  for (const term of model.terms) value += term.readout * Math.max(0, dot(term.normal, input) + term.offset);
  return Number.isFinite(value) ? value : 0;
}
