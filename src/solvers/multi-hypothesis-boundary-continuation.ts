export type MbcdQueryPolicy = 'contradiction-seeking' | 'prediction-disagreement';
export type MbcdRankPolicy = 'retain-alternatives' | 'single-best';
export type MbcdValidationPolicy = 'independent' | 'support-only';

export type MbcdRankHypothesis = {
  rank: number;
  probability: number;
  residualEnergyRatio: number;
  plausible: boolean;
};

export type MbcdBoundaryTerm = {
  normal: number[];
  offset: number;
  readout: number;
  confidence: number;
  support: number;
  boundaryDetected: boolean;
  normalIdentified: boolean;
  offsetIdentified: boolean;
  unique: boolean;
  normalUncertaintyDegrees: number;
  offsetUncertainty: number;
  validationNegativeSupport: number;
  validationPositiveSupport: number;
};

export type MbcdDiagnostics = {
  totalOracleCalls: number;
  oracleBudget: number;
  budgetExhausted: boolean;
  stoppedReason: 'oracle-budget' | 'probe-limit' | 'reconstruction-complete' | null;
  gradientProbes: number;
  oracleCallsPerGradientProbe: number;
  contradictionQueries: number;
  predictionDisagreementQueries: number;
  acquisitionQueries: number;
  boundaryValidationQueries: number;
  queryPolicy: MbcdQueryPolicy;
  rankPolicy: MbcdRankPolicy;
  validationPolicy: MbcdValidationPolicy;
  rankHypotheses: MbcdRankHypothesis[];
  plausibleActiveRanks: number[];
  estimatedActiveRank: number | null;
  rankUncertainty: number;
  rankResolved: boolean;
  rankStability: number;
  boundaryHypotheses: number;
  boundaryDetectionCount: number;
  normalIdentifiedCount: number;
  offsetIdentifiedCount: number;
  uniqueHypothesisCount: number;
  acceptedBoundaryCount: number;
  unresolvedHypothesisCount: number;
  unresolvedHypotheses: Array<{
    normal: number[];
    offset: number;
    confidence: number;
    detectionStatus: 'undetected' | 'detected';
    normalIdentified: boolean;
    offsetIdentified: boolean;
    unique: boolean;
  }>;
  fitResidualRelative: number | null;
  reconstructionComplete: boolean;
  reasons: string[];
};

export type MultiHypothesisBoundaryContinuationModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  baseIntercept: number;
  baseWeights: number[];
  terms: MbcdBoundaryTerm[];
  expression: string;
  reconstructionComplete: boolean;
  diagnostics: MbcdDiagnostics;
};

export type MultiHypothesisBoundaryContinuationConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  termCount: number;
  oracleBudget: number;
  randomSeed?: number;
  finiteDifferenceStep?: number;
  maxGradientProbes?: number;
  maximumBoundaryHypotheses?: number;
  minimumGradientJump?: number;
  maximumRelativeFitResidual?: number;
  ridge?: number;
  queryPolicy?: MbcdQueryPolicy;
  rankPolicy?: MbcdRankPolicy;
  validationPolicy?: MbcdValidationPolicy;
};

type Observation = {
  point: number[];
  value: number;
  gradient: number[];
  intercept: number;
  index: number;
};

type BoundaryHypothesis = {
  normal: number[];
  offset: number;
  gradientJump: number[];
  confidence: number;
  support: number;
  createdAt: number;
  directionSamples: number[][];
  offsetSamples: number[];
  formationIndices: Set<number>;
  validationNegativeSupport: number;
  validationPositiveSupport: number;
  validationJump: number[];
  validationAlignment: number;
  normalUncertaintyDegrees: number;
  offsetUncertainty: number;
  boundaryDetected: boolean;
  normalIdentified: boolean;
  offsetIdentified: boolean;
  unique: boolean;
};

type EigenSystem = { values: number[]; vectors: number[][] };

const ANGLE_TOLERANCE_DEGREES = 12;
const PLANE_DUPLICATE_ANGLE_COSINE = Math.cos(8 * Math.PI / 180);
const PLANE_DUPLICATE_OFFSET = 0.04;
const MAX_ACQUISITION_CANDIDATES = 32;

function dot(first: number[], second: number[]): number {
  let result = 0;
  for (let index = 0; index < first.length; index++) result += first[index] * second[index];
  return result;
}

function vectorNorm(vector: number[]): number {
  return Math.sqrt(Math.max(0, dot(vector, vector)));
}

function normalize(vector: number[]): number[] {
  const length = vectorNorm(vector);
  return length > 1e-14 ? vector.map(value => value / length) : vector.map(() => 0);
}

function mean(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function meanVector(vectors: number[][], dimension: number): number[] {
  return Array.from({ length: dimension }, (_, axis) => mean(vectors.map(vector => vector[axis])));
}

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = state * 16807 % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function canonicalize(normalInput: number[], offsetInput: number): { normal: number[]; offset: number } {
  const length = vectorNorm(normalInput);
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

function randomPoint(random: () => number, ranges: Array<[number, number]>): number[] {
  return ranges.map(([minimum, maximum]) => minimum + (maximum - minimum) * random());
}

function clampPoint(point: number[], ranges: Array<[number, number]>): number[] {
  return point.map((coordinate, axis) =>
    Math.min(ranges[axis][1], Math.max(ranges[axis][0], coordinate)));
}

function covarianceEigenSystem(gradients: number[][]): EigenSystem {
  const dimension = gradients[0]?.length ?? 0;
  if (!dimension || gradients.length < 2) {
    return { values: new Array(dimension).fill(0), vectors: [] };
  }
  const center = meanVector(gradients, dimension);
  const matrix = Array.from({ length: dimension }, () => new Array(dimension).fill(0));
  for (const gradient of gradients) {
    for (let row = 0; row < dimension; row++) {
      const centeredRow = gradient[row] - center[row];
      for (let column = row; column < dimension; column++) {
        matrix[row][column] += centeredRow * (gradient[column] - center[column]) / gradients.length;
      }
    }
  }
  for (let row = 0; row < dimension; row++) {
    for (let column = 0; column < row; column++) matrix[row][column] = matrix[column][row];
  }
  const vectors: number[][] = Array.from({ length: dimension }, (_, row) =>
    Array.from({ length: dimension }, (_, column) => row === column ? 1 : 0));
  for (let sweep = 0; sweep < 10; sweep++) {
    let largest = 0;
    for (let p = 0; p < dimension; p++) {
      for (let q = p + 1; q < dimension; q++) {
        const cross = matrix[p][q];
        largest = Math.max(largest, Math.abs(cross));
        if (Math.abs(cross) < 1e-12) continue;
        const angle = 0.5 * Math.atan2(2 * cross, matrix[q][q] - matrix[p][p]);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const pp = matrix[p][p];
        const qq = matrix[q][q];
        matrix[p][p] = cosine * cosine * pp - 2 * sine * cosine * cross + sine * sine * qq;
        matrix[q][q] = sine * sine * pp + 2 * sine * cosine * cross + cosine * cosine * qq;
        matrix[p][q] = 0;
        matrix[q][p] = 0;
        for (let index = 0; index < dimension; index++) {
          if (index === p || index === q) continue;
          const ip = matrix[index][p];
          const iq = matrix[index][q];
          matrix[index][p] = cosine * ip - sine * iq;
          matrix[p][index] = matrix[index][p];
          matrix[index][q] = sine * ip + cosine * iq;
          matrix[q][index] = matrix[index][q];
        }
        for (let row = 0; row < dimension; row++) {
          const vp = vectors[row][p];
          const vq = vectors[row][q];
          vectors[row][p] = cosine * vp - sine * vq;
          vectors[row][q] = sine * vp + cosine * vq;
        }
      }
    }
    if (largest < 1e-11) break;
  }
  const ordered = Array.from({ length: dimension }, (_, index) => ({
    value: Math.max(0, matrix[index][index]),
    vector: normalize(vectors.map(row => row[index]))
  })).sort((first, second) => second.value - first.value);
  return { values: ordered.map(entry => entry.value), vectors: ordered.map(entry => entry.vector) };
}

function normalizedSpectrum(eigenSystem: EigenSystem): number[] {
  const total = eigenSystem.values.reduce((sum, value) => sum + value, 0);
  return total <= 1e-14
    ? eigenSystem.values.map(() => 0)
    : eigenSystem.values.map(value => value / total);
}

function rankHypotheses(
  eigenSystem: EigenSystem,
  maximumRank: number,
  previousSpectrum: number[][],
  rankPolicy: MbcdRankPolicy
): { hypotheses: MbcdRankHypothesis[]; plausibleRanks: number[]; estimate: number | null; uncertainty: number; resolved: boolean; stability: number } {
  const spectrum = normalizedSpectrum(eigenSystem);
  const candidates = Array.from({ length: maximumRank + 1 }, (_, rank) => {
    const tail = eigenSystem.values.slice(rank).reduce((sum, value) => sum + value, 0);
    const total = eigenSystem.values.reduce((sum, value) => sum + value, 0);
    const residualEnergyRatio = total <= 1e-14 ? 0 : tail / total;
    return { rank, residualEnergyRatio, score: residualEnergyRatio + rank * 0.025 };
  });
  const best = candidates.reduce((chosen, candidate) =>
    candidate.score < chosen.score ? candidate : chosen, candidates[0]);
  const temperature = 0.01;
  const unnormalized = candidates.map(candidate => Math.exp(-(candidate.score - best.score) / temperature));
  const totalProbability = unnormalized.reduce((sum, value) => sum + value, 0);
  const probabilities = unnormalized.map(value => value / totalProbability);
  const allHypotheses = candidates.map((candidate, index) => ({
    rank: candidate.rank,
    probability: probabilities[index],
    residualEnergyRatio: candidate.residualEnergyRatio,
    plausible: candidate.score <= best.score + 0.04 && probabilities[index] >= 0.04
  }));
  const top = allHypotheses.reduce((chosen, candidate) =>
    candidate.probability > chosen.probability ? candidate : chosen, allHypotheses[0]);
  const plausible = allHypotheses.filter(candidate => candidate.plausible);
  const retained = rankPolicy === 'retain-alternatives'
    ? plausible
    : [top];
  const entropy = -allHypotheses.reduce((sum, candidate) =>
    sum + (candidate.probability > 0 ? candidate.probability * Math.log(candidate.probability) : 0), 0);
  const uncertainty = candidates.length <= 1 ? 0 : entropy / Math.log(candidates.length);
  const recent = previousSpectrum.slice(-4);
  let stability = 0;
  if (recent.length >= 2 && spectrum.length) {
    const distances: number[] = [];
    for (let index = 1; index < recent.length; index++) {
      distances.push(spectrum.reduce((sum, value, axis) =>
        sum + Math.abs(value - recent[index - 1][axis]), 0));
    }
    stability = Math.max(0, 1 - mean(distances) / 0.5);
  }
  return {
    hypotheses: retained,
    plausibleRanks: retained.map(candidate => candidate.rank),
    estimate: top.rank,
    uncertainty,
    resolved: top.probability >= 0.85 && stability >= 0.75,
    stability,
  };
}

function normalDispersionDegrees(samples: number[][], reference: number[]): number {
  if (samples.length < 2) return 180;
  return mean(samples.map(sample => {
    const similarity = Math.min(1, Math.max(-1, Math.abs(dot(normalize(sample), reference))));
    return Math.acos(similarity) * 180 / Math.PI;
  }));
}

function updateCandidateEvidence(
  candidate: BoundaryHypothesis,
  observations: Observation[],
  minimumGradientJump: number,
  finiteDifferenceStep: number,
  validationPolicy: MbcdValidationPolicy,
  localRadius: number
): void {
  const stencilMargin = finiteDifferenceStep
    * candidate.normal.reduce((sum, value) => sum + Math.abs(value), 0) * 1.5;
  const eligible = observations.filter(observation =>
    (validationPolicy === 'support-only' || observation.index >= candidate.createdAt)
    && !candidate.formationIndices.has(observation.index)
    && Math.abs(dot(candidate.normal, observation.point) + candidate.offset) <= localRadius
    && Math.abs(dot(candidate.normal, observation.point) + candidate.offset) > stencilMargin);
  const negative = eligible.filter(observation =>
    dot(candidate.normal, observation.point) + candidate.offset < 0);
  const positive = eligible.filter(observation =>
    dot(candidate.normal, observation.point) + candidate.offset > 0);
  candidate.validationNegativeSupport = negative.length;
  candidate.validationPositiveSupport = positive.length;
  const negativeGradient = meanVector(negative.map(observation => observation.gradient), candidate.normal.length);
  const positiveGradient = meanVector(positive.map(observation => observation.gradient), candidate.normal.length);
  candidate.validationJump = positiveGradient.map((value, axis) => value - negativeGradient[axis]);
  const jumpNorm = vectorNorm(candidate.validationJump);
  candidate.validationAlignment = jumpNorm <= 1e-14
    ? 0
    : Math.abs(dot(normalize(candidate.validationJump), candidate.normal));
  candidate.boundaryDetected = negative.length >= 1
    && positive.length >= 1
    && jumpNorm >= minimumGradientJump
    && candidate.validationAlignment >= 0.8;
  candidate.normalUncertaintyDegrees = normalDispersionDegrees(candidate.directionSamples, candidate.normal);
  candidate.offsetUncertainty = candidate.offsetSamples.length < 2
    ? 1e6
    : Math.max(...candidate.offsetSamples) - Math.min(...candidate.offsetSamples);
  candidate.normalIdentified = candidate.boundaryDetected
    && negative.length >= 2
    && positive.length >= 2
    && candidate.directionSamples.length >= 2
    && candidate.normalUncertaintyDegrees <= ANGLE_TOLERANCE_DEGREES
    && candidate.validationAlignment >= 0.95;
  candidate.offsetIdentified = candidate.boundaryDetected
    && negative.length >= 2
    && positive.length >= 2
    && candidate.offsetSamples.length >= 2
    && candidate.offsetUncertainty <= 0.1;
  const sideBalance = Math.min(negative.length, positive.length)
    / Math.max(1, Math.max(negative.length, positive.length));
  candidate.support = Math.min(negative.length, positive.length);
  candidate.confidence = Math.min(1, Math.max(0,
    candidate.validationAlignment * Math.sqrt(sideBalance)
    * Math.min(1, candidate.support / 3)
    * Math.min(1, candidate.directionSamples.length / 3)));
}

function candidateFromPair(
  first: Observation,
  second: Observation,
  createdAt: number,
  minimumGradientJump: number
): BoundaryHypothesis | null {
  const jump = second.gradient.map((value, axis) => value - first.gradient[axis]);
  const jumpNorm = vectorNorm(jump);
  if (jumpNorm < minimumGradientJump) return null;
  const plane = canonicalize(jump, second.intercept - first.intercept);
  return {
    normal: plane.normal,
    offset: plane.offset,
    gradientJump: jump,
    confidence: 0,
    support: 0,
    createdAt,
    directionSamples: [plane.normal],
    offsetSamples: [plane.offset],
    formationIndices: new Set([first.index, second.index]),
    validationNegativeSupport: 0,
    validationPositiveSupport: 0,
    validationJump: new Array(jump.length).fill(0),
    validationAlignment: 0,
    normalUncertaintyDegrees: 180,
    offsetUncertainty: 1e6,
    boundaryDetected: false,
    normalIdentified: false,
    offsetIdentified: false,
    unique: false
  };
}

function updateBoundaryHypotheses(
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  maximumHypotheses: number,
  minimumGradientJump: number,
  finiteDifferenceStep: number,
  validationPolicy: MbcdValidationPolicy,
  localRadius: number
): BoundaryHypothesis[] {
  if (observations.length < 2) return hypotheses;
  const latest = observations[observations.length - 1];
  const alternatives = observations.slice(0, -1)
    .map(observation => ({
      observation,
      jump: vectorNorm(latest.gradient.map((value, axis) => value - observation.gradient[axis]))
    }))
    .filter(candidate => candidate.jump >= minimumGradientJump)
    .sort((first, second) => second.jump - first.jump)
    .slice(0, 12);
  for (const alternative of alternatives) {
    const candidate = candidateFromPair(
      alternative.observation,
      latest,
      observations.length,
      minimumGradientJump
    );
    if (!candidate) continue;
    const duplicate = hypotheses.find(existing =>
      Math.abs(dot(existing.normal, candidate.normal)) >= PLANE_DUPLICATE_ANGLE_COSINE
      && Math.abs(existing.offset - candidate.offset) <= PLANE_DUPLICATE_OFFSET);
    if (duplicate) {
      duplicate.directionSamples.push(candidate.normal);
      duplicate.offsetSamples.push(candidate.offset);
      for (const index of candidate.formationIndices) duplicate.formationIndices.add(index);
      duplicate.gradientJump = candidate.gradientJump;
    } else {
      hypotheses.push(candidate);
    }
  }
  hypotheses.sort((first, second) =>
    second.confidence * Math.sqrt(Math.max(1, second.support))
    - first.confidence * Math.sqrt(Math.max(1, first.support)));
  hypotheses.splice(maximumHypotheses);
  for (const hypothesis of hypotheses) {
    updateCandidateEvidence(
      hypothesis,
      observations,
      minimumGradientJump,
      finiteDifferenceStep,
      validationPolicy,
      localRadius
    );
  }
  const formationIndexMap = new Map<number, BoundaryHypothesis[]>();
  for (const hypothesis of hypotheses) {
    for (const index of hypothesis.formationIndices) {
      const entries = formationIndexMap.get(index) ?? [];
      entries.push(hypothesis);
      formationIndexMap.set(index, entries);
    }
  }
  for (const hypothesis of hypotheses) {
    const alternatives = new Set<BoundaryHypothesis>();
    for (const index of hypothesis.formationIndices) {
      for (const other of formationIndexMap.get(index) ?? []) {
        if (other !== hypothesis) alternatives.add(other);
      }
    }
    const unresolvedAlternative = [...alternatives].some(other =>
      (
        (Math.abs(dot(other.normal, hypothesis.normal)) >= 0.9
          && Math.abs(other.offset - hypothesis.offset) <= 0.1
          && (!other.boundaryDetected || !other.normalIdentified || !other.offsetIdentified))
        || (
          [...hypothesis.formationIndices].some(index => other.formationIndices.has(index))
          && (Math.abs(dot(other.normal, hypothesis.normal)) < PLANE_DUPLICATE_ANGLE_COSINE
            || Math.abs(other.offset - hypothesis.offset) > PLANE_DUPLICATE_OFFSET)
          && !(
            hypothesis.boundaryDetected && hypothesis.normalIdentified && hypothesis.offsetIdentified
            && other.boundaryDetected && other.normalIdentified && other.offsetIdentified
          )
        )
      ));
    hypothesis.unique = !unresolvedAlternative;
  }
  return hypotheses;
}

function featureVector(point: number[], hypotheses: BoundaryHypothesis[]): number[] {
  return [
    1,
    ...point,
    ...hypotheses.map(hypothesis =>
      Math.max(0, dot(hypothesis.normal, point) + hypothesis.offset))
  ];
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] | null {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let index = column; index <= size; index++) augmented[column][index] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let index = column; index <= size; index++) {
        augmented[row][index] -= factor * augmented[column][index];
      }
    }
  }
  return augmented.map(row => row[size]);
}

function fitCoefficients(
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  ridge: number
): number[] | null {
  const width = 1 + (observations[0]?.point.length ?? 0) + hypotheses.length;
  if (observations.length < width) return null;
  const gram = Array.from({ length: width }, () => new Array(width).fill(0));
  const target = new Array(width).fill(0);
  for (const observation of observations) {
    const features = featureVector(observation.point, hypotheses);
    for (let row = 0; row < width; row++) {
      target[row] += features[row] * observation.value;
      for (let column = 0; column < width; column++) {
        gram[row][column] += features[row] * features[column];
      }
    }
  }
  for (let index = 1; index < width; index++) gram[index][index] += ridge;
  return solveLinearSystem(gram, target);
}

function evaluateFit(
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  coefficients: number[]
): number {
  const residuals = observations.map(observation =>
    dot(featureVector(observation.point, hypotheses), coefficients) - observation.value);
  const residual = Math.sqrt(mean(residuals.map(value => value * value)));
  const scale = Math.sqrt(mean(observations.map(observation => observation.value ** 2)));
  return residual / Math.max(1e-8, scale);
}

function candidatePredictionDisagreement(point: number[], hypotheses: BoundaryHypothesis[]): number {
  if (hypotheses.length < 2) return 0;
  const predictions = hypotheses.map(hypothesis => {
    const side = dot(hypothesis.normal, point) + hypothesis.offset >= 0;
    return side ? hypothesis.validationJump : new Array(point.length).fill(0);
  });
  const center = meanVector(predictions, point.length);
  return mean(predictions.map(prediction =>
    vectorNorm(prediction.map((value, axis) => value - center[axis]))));
}

function selectProbe(
  random: () => number,
  ranges: Array<[number, number]>,
  observations: Observation[],
  hypotheses: BoundaryHypothesis[],
  queryPolicy: MbcdQueryPolicy,
  rankUncertainty: number,
  remainingBudget: number,
  probeCost: number
): { point: number[]; contradictionScore: number } {
  const pool = Array.from({ length: MAX_ACQUISITION_CANDIDATES }, () => randomPoint(random, ranges));
  const acquisitionHypotheses = hypotheses.slice(0, 16);
  const margin = Math.max(1e-3, Math.min(0.08, probeCost > 0 ? 0.5 * Math.sqrt(probeCost) / 100 : 0.02));
  for (const hypothesis of acquisitionHypotheses.slice(0, 8)) {
    for (const side of [-1, 1]) {
      const point = randomPoint(random, ranges);
      const signedDistance = dot(hypothesis.normal, point) + hypothesis.offset;
      pool.push(clampPoint(
        point.map((value, axis) => value + (side * margin - signedDistance) * hypothesis.normal[axis]),
        ranges
      ));
    }
  }
  const disagreementScale = Math.max(1e-8, Math.sqrt(ranges.reduce((sum, [low, high]) =>
    sum + (high - low) ** 2, 0)));
  const costFactor = Math.min(1, remainingBudget / Math.max(1, probeCost));
  let bestPoint = pool[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  let bestDisagreement = 0;
  for (const point of pool) {
    let boundaryDisagreement = 0;
    if (queryPolicy === 'contradiction-seeking') {
      let negativeWeight = 0;
      let positiveWeight = 0;
      for (const hypothesis of acquisitionHypotheses) {
        const weight = Math.sqrt(Math.max(0, hypothesis.confidence));
        if (dot(hypothesis.normal, point) + hypothesis.offset >= 0) positiveWeight += weight;
        else negativeWeight += weight;
      }
      boundaryDisagreement = negativeWeight * positiveWeight;
    } else {
      boundaryDisagreement = candidatePredictionDisagreement(point, acquisitionHypotheses) / disagreementScale;
    }
    const boundaryProximity = acquisitionHypotheses.length
      ? Math.max(...acquisitionHypotheses.map(hypothesis =>
        Math.max(0, 1 - Math.abs(dot(hypothesis.normal, point) + hypothesis.offset) / 0.25)))
      : 0;
    const nearestDistance = observations.length
      ? Math.min(...observations.map(observation =>
        vectorNorm(point.map((value, axis) => value - observation.point[axis]))))
      : disagreementScale;
    const novelty = Math.min(1, nearestDistance / disagreementScale);
    const score = costFactor
      * (2 * boundaryDisagreement + boundaryProximity + 0.25 * novelty + 0.15 * rankUncertainty);
    if (score > bestScore) {
      bestScore = score;
      bestPoint = point;
      bestDisagreement = boundaryDisagreement;
    }
  }
  return { point: bestPoint, contradictionScore: bestDisagreement };
}

function buildExpression(
  model: Pick<MultiHypothesisBoundaryContinuationModel, 'baseIntercept' | 'baseWeights' | 'terms'>
): string {
  const terms: string[] = [];
  if (model.baseIntercept !== 0) terms.push(model.baseIntercept.toFixed(5));
  model.baseWeights.forEach((weight, axis) => {
    if (weight !== 0) terms.push(`${weight.toFixed(5)}*x${axis}`);
  });
  model.terms.forEach(term => {
    if (term.readout !== 0) {
      const linear = term.normal.map((value, axis) =>
        value === 0 ? '' : `${value.toFixed(4)}*x${axis}`).filter(Boolean).join(' + ');
      terms.push(`${term.readout.toFixed(5)}*relu(${linear} ${term.offset >= 0 ? '+' : '-'} ${Math.abs(term.offset).toFixed(5)})`);
    }
  });
  return terms.length ? terms.join(' + ') : '0';
}

function validateConfig(config: MultiHypothesisBoundaryContinuationConfig): void {
  if (!Number.isSafeInteger(config.dimension) || config.dimension < 1) {
    throw new Error('MBCD dimension must be a positive safe integer');
  }
  if (!Number.isSafeInteger(config.termCount) || config.termCount < 1) {
    throw new Error('MBCD termCount must be a positive safe integer');
  }
  if (!Number.isSafeInteger(config.oracleBudget) || config.oracleBudget < 0) {
    throw new Error('MBCD oracleBudget must be a non-negative safe integer');
  }
  if (config.ranges.length !== config.dimension || config.ranges.some(([low, high]) =>
    !Number.isFinite(low) || !Number.isFinite(high) || low >= high)) {
    throw new Error('MBCD ranges must contain one finite increasing interval per input dimension');
  }
  if (config.maxGradientProbes !== undefined
    && (!Number.isSafeInteger(config.maxGradientProbes) || config.maxGradientProbes < 1)) {
    throw new Error('MBCD maxGradientProbes must be a positive safe integer');
  }
  if (config.maximumBoundaryHypotheses !== undefined
    && (!Number.isSafeInteger(config.maximumBoundaryHypotheses) || config.maximumBoundaryHypotheses < 1)) {
    throw new Error('MBCD maximumBoundaryHypotheses must be a positive safe integer');
  }
  if (config.randomSeed !== undefined && !Number.isSafeInteger(config.randomSeed)) {
    throw new Error('MBCD randomSeed must be a safe integer');
  }
  if (config.finiteDifferenceStep !== undefined
    && (!Number.isFinite(config.finiteDifferenceStep) || config.finiteDifferenceStep <= 0)) {
    throw new Error('MBCD finiteDifferenceStep must be positive and finite');
  }
  if (config.minimumGradientJump !== undefined
    && (!Number.isFinite(config.minimumGradientJump) || config.minimumGradientJump < 0)) {
    throw new Error('MBCD minimumGradientJump must be non-negative and finite');
  }
  if (config.maximumRelativeFitResidual !== undefined
    && (!Number.isFinite(config.maximumRelativeFitResidual) || config.maximumRelativeFitResidual <= 0)) {
    throw new Error('MBCD maximumRelativeFitResidual must be positive and finite');
  }
  if (config.ridge !== undefined && (!Number.isFinite(config.ridge) || config.ridge < 0)) {
    throw new Error('MBCD ridge must be non-negative and finite');
  }
  if (!['contradiction-seeking', 'prediction-disagreement'].includes(config.queryPolicy ?? 'contradiction-seeking')) {
    throw new Error('MBCD queryPolicy is not supported');
  }
  if (!['retain-alternatives', 'single-best'].includes(config.rankPolicy ?? 'retain-alternatives')) {
    throw new Error('MBCD rankPolicy is not supported');
  }
  if (!['independent', 'support-only'].includes(config.validationPolicy ?? 'independent')) {
    throw new Error('MBCD validationPolicy is not supported');
  }
}

export function decompileBlackBoxReluNdMultiHypothesisBoundaryContinuation(
  oracle: (input: number[]) => number,
  config: MultiHypothesisBoundaryContinuationConfig
): MultiHypothesisBoundaryContinuationModel {
  validateConfig(config);
  const dimension = config.dimension;
  const ranges = config.ranges.map(range => [...range] as [number, number]);
  const step = config.finiteDifferenceStep ?? 1e-3;
  if (!Number.isFinite(step) || step <= 0) throw new Error('MBCD finiteDifferenceStep must be positive and finite');
  const queryPolicy = config.queryPolicy ?? 'contradiction-seeking';
  const rankPolicy = config.rankPolicy ?? 'retain-alternatives';
  const validationPolicy = config.validationPolicy ?? 'independent';
  const random = makeRandom(config.randomSeed ?? 1);
  const oracleCallsPerGradientProbe = 2 * dimension + 1;
  const maxGradientProbes = config.maxGradientProbes
    ?? Math.min(64, Math.max(1, Math.floor(config.oracleBudget / oracleCallsPerGradientProbe)));
  const maximumHypotheses = config.maximumBoundaryHypotheses
    ?? Math.max(8, Math.min(32, 4 * config.termCount));
  const minimumGradientJump = config.minimumGradientJump ?? 0.005;
  const maximumRelativeFitResidual = config.maximumRelativeFitResidual ?? 0.05;
  const localRadius = 0.08 * Math.sqrt(ranges.reduce((sum, [low, high]) =>
    sum + (high - low) ** 2, 0));
  const observations: Observation[] = [];
  let hypotheses: BoundaryHypothesis[] = [];
  let totalOracleCalls = 0;
  let budgetExhausted = false;
  let stoppedReason: MbcdDiagnostics['stoppedReason'] = null;
  let contradictionQueries = 0;
  let predictionDisagreementQueries = 0;
  let acquisitionQueries = 0;
  let boundaryValidationQueries = 0;
  let eigenSystem: EigenSystem = { values: new Array(dimension).fill(0), vectors: [] };
  let rankEvidence: ReturnType<typeof rankHypotheses> = {
    hypotheses: [],
    plausibleRanks: [],
    estimate: null,
    uncertainty: 1,
    resolved: false,
    stability: 0
  };
  const previousSpectra: number[][] = [];
  const queryOracle = (input: number[]): number => {
    if (totalOracleCalls >= config.oracleBudget) {
      throw new Error('MBCD internal oracle budget guard rejected an unbudgeted call');
    }
    totalOracleCalls++;
    const result = oracle([...input]);
    if (!Number.isFinite(result)) throw new Error('MBCD oracle returned a non-finite value');
    return result;
  };
  const refreshRankEvidence = (): void => {
    eigenSystem = covarianceEigenSystem(observations.map(entry => entry.gradient));
    const spectrum = normalizedSpectrum(eigenSystem);
    if (spectrum.length) previousSpectra.push(spectrum);
    rankEvidence = rankHypotheses(eigenSystem, dimension, previousSpectra, rankPolicy);
  };
  const observeGradient = (centerInput: number[]): Observation => {
    const center = clampPoint(centerInput, ranges);
    const value = queryOracle(center);
    const gradient = new Array(dimension).fill(0);
    for (let axis = 0; axis < dimension; axis++) {
      const plus = [...center];
      const minus = [...center];
      plus[axis] = Math.min(ranges[axis][1], center[axis] + step);
      minus[axis] = Math.max(ranges[axis][0], center[axis] - step);
      const denominator = plus[axis] - minus[axis];
      if (denominator <= 0) throw new Error(`MBCD finite-difference interval is empty on axis ${axis}`);
      gradient[axis] = (queryOracle(plus) - queryOracle(minus)) / denominator;
      if (!Number.isFinite(gradient[axis])) throw new Error(`MBCD gradient is non-finite on axis ${axis}`);
    }
    const observation: Observation = {
      point: center,
      value,
      gradient,
      intercept: value - dot(gradient, center),
      index: observations.length
    };
    if (!Number.isFinite(observation.intercept)) throw new Error('MBCD affine intercept is non-finite');
    if (hypotheses.some(hypothesis => {
      const signedDistance = Math.abs(dot(hypothesis.normal, observation.point) + hypothesis.offset);
      const margin = step * hypothesis.normal.reduce((sum, value) => sum + Math.abs(value), 0) * 1.5;
      return observation.index >= hypothesis.createdAt
        && !hypothesis.formationIndices.has(observation.index)
        && signedDistance > margin
        && signedDistance <= localRadius;
    })) {
      boundaryValidationQueries++;
    }
    observations.push(observation);
    if (observations.length % 4 === 0) refreshRankEvidence();
    hypotheses = updateBoundaryHypotheses(
      observations,
      hypotheses,
      maximumHypotheses,
      minimumGradientJump,
      step,
      validationPolicy,
      localRadius
    );
    return observation;
  };

  for (let probe = 0; probe < maxGradientProbes; probe++) {
    if (config.oracleBudget - totalOracleCalls < oracleCallsPerGradientProbe) {
      budgetExhausted = true;
      stoppedReason = 'oracle-budget';
      break;
    }
    let center: number[];
    if (observations.length < 4 || hypotheses.length === 0) {
      center = randomPoint(random, ranges);
    } else {
      const selection = selectProbe(
        random,
        ranges,
        observations,
        hypotheses,
        queryPolicy,
        rankEvidence.uncertainty,
        config.oracleBudget - totalOracleCalls,
        oracleCallsPerGradientProbe
      );
      center = selection.point;
      acquisitionQueries++;
      if (queryPolicy === 'contradiction-seeking' && selection.contradictionScore > 0) {
        contradictionQueries++;
      } else if (queryPolicy === 'prediction-disagreement' && selection.contradictionScore > 1e-8) {
        predictionDisagreementQueries++;
      }
    }
    observeGradient(center);
    if (observations.length >= dimension + config.termCount + 5) {
      const accepted = hypotheses.filter(candidate =>
        candidate.boundaryDetected && candidate.normalIdentified
        && candidate.offsetIdentified && candidate.unique);
      const readyRank = rankEvidence.resolved && rankEvidence.plausibleRanks.length === 1;
      if (accepted.length >= config.termCount && readyRank) {
        const fitCandidates = accepted.slice(0, config.termCount);
        const coefficients = fitCoefficients(observations, fitCandidates, config.ridge ?? 1e-8);
        if (coefficients) {
          const residual = evaluateFit(observations, fitCandidates, coefficients);
          if (residual <= maximumRelativeFitResidual) {
            stoppedReason = 'reconstruction-complete';
            break;
          }
        }
      }
    }
  }
  const cannotAffordNextProbe = config.oracleBudget - totalOracleCalls < oracleCallsPerGradientProbe;
  if (cannotAffordNextProbe) budgetExhausted = true;
  if (stoppedReason === null) {
    if (cannotAffordNextProbe) stoppedReason = 'oracle-budget';
    else if (observations.length >= maxGradientProbes) stoppedReason = 'probe-limit';
  }
  if (observations.length && observations.length % 4 !== 0) refreshRankEvidence();

  const rankedHypotheses = [...hypotheses].sort((first, second) =>
    Number(second.boundaryDetected) - Number(first.boundaryDetected)
    || Number(second.normalIdentified && second.offsetIdentified) - Number(first.normalIdentified && first.offsetIdentified)
    || second.confidence * Math.sqrt(Math.max(1, second.support))
      - first.confidence * Math.sqrt(Math.max(1, first.support)));
  const selectedHypotheses = rankedHypotheses.slice(0, config.termCount);
  const coefficients = fitCoefficients(observations, selectedHypotheses, config.ridge ?? 1e-8);
  const fitResidualRelative = coefficients
    ? evaluateFit(observations, selectedHypotheses, coefficients)
    : null;
  const acceptedBoundaryCount = hypotheses.filter(candidate =>
    candidate.boundaryDetected && candidate.normalIdentified
    && candidate.offsetIdentified && candidate.unique).length;
  const unresolvedHypotheses = hypotheses
    .filter(candidate => candidate.confidence > 0.05
      && !(candidate.boundaryDetected && candidate.normalIdentified
        && candidate.offsetIdentified && candidate.unique))
    .map(candidate => ({
      normal: candidate.normal,
      offset: candidate.offset,
      confidence: candidate.confidence,
      detectionStatus: candidate.boundaryDetected ? 'detected' as const : 'undetected' as const,
      normalIdentified: candidate.normalIdentified,
      offsetIdentified: candidate.offsetIdentified,
      unique: candidate.unique
    }));
  const enoughFitSamples = coefficients !== null;
  const reconstructionComplete = acceptedBoundaryCount >= config.termCount
    && selectedHypotheses.length === config.termCount
    && unresolvedHypotheses.length === 0
    && rankEvidence.resolved
    && rankEvidence.plausibleRanks.length === 1
    && enoughFitSamples
    && fitResidualRelative !== null
    && fitResidualRelative <= maximumRelativeFitResidual
    && stoppedReason !== 'oracle-budget';
  const reasons: string[] = [];
  if (acceptedBoundaryCount < config.termCount) {
    reasons.push(`only_${acceptedBoundaryCount}_of_${config.termCount}_boundaries_fully_identified`);
  }
  if (!rankEvidence.resolved || rankEvidence.plausibleRanks.length !== 1) {
    reasons.push('competing_active_rank_hypotheses_remain');
  }
  if (unresolvedHypotheses.length) reasons.push('competing_boundary_hypotheses_remain_unresolved');
  if (!enoughFitSamples) reasons.push('insufficient_observations_for_overdetermined_fit');
  if (fitResidualRelative !== null && fitResidualRelative > maximumRelativeFitResidual) {
    reasons.push('fit_residual_exceeded_completion_threshold');
  }
  if (stoppedReason === 'oracle-budget') reasons.push('oracle_budget_exhausted_before_complete_identification');
  const terms: MbcdBoundaryTerm[] = selectedHypotheses.map((hypothesis, index) => ({
    normal: hypothesis.normal,
    offset: hypothesis.offset,
    readout: coefficients ? coefficients[dimension + 1 + index] : 0,
    confidence: hypothesis.confidence,
    support: hypothesis.support,
    boundaryDetected: hypothesis.boundaryDetected,
    normalIdentified: hypothesis.normalIdentified,
    offsetIdentified: hypothesis.offsetIdentified,
    unique: hypothesis.unique,
    normalUncertaintyDegrees: hypothesis.normalUncertaintyDegrees,
    offsetUncertainty: hypothesis.offsetUncertainty,
    validationNegativeSupport: hypothesis.validationNegativeSupport,
    validationPositiveSupport: hypothesis.validationPositiveSupport
  }));
  const baseModel = {
    dimension,
    ranges,
    baseIntercept: coefficients?.[0] ?? 0,
    baseWeights: coefficients?.slice(1, dimension + 1) ?? new Array(dimension).fill(0),
    terms,
    reconstructionComplete
  };
  const diagnostics: MbcdDiagnostics = {
    totalOracleCalls,
    oracleBudget: config.oracleBudget,
    budgetExhausted,
    stoppedReason,
    gradientProbes: observations.length,
    oracleCallsPerGradientProbe,
    contradictionQueries,
    predictionDisagreementQueries,
    acquisitionQueries,
    boundaryValidationQueries,
    queryPolicy,
    rankPolicy,
    validationPolicy,
    rankHypotheses: rankEvidence.hypotheses,
    plausibleActiveRanks: rankEvidence.plausibleRanks,
    estimatedActiveRank: rankEvidence.estimate,
    rankUncertainty: rankEvidence.uncertainty,
    rankResolved: rankEvidence.resolved && rankEvidence.plausibleRanks.length === 1,
    rankStability: rankEvidence.stability,
    boundaryHypotheses: hypotheses.length,
    boundaryDetectionCount: hypotheses.filter(candidate => candidate.boundaryDetected).length,
    normalIdentifiedCount: hypotheses.filter(candidate => candidate.normalIdentified).length,
    offsetIdentifiedCount: hypotheses.filter(candidate => candidate.offsetIdentified).length,
    uniqueHypothesisCount: hypotheses.filter(candidate => candidate.unique).length,
    acceptedBoundaryCount,
    unresolvedHypothesisCount: unresolvedHypotheses.length,
    unresolvedHypotheses,
    fitResidualRelative,
    reconstructionComplete,
    reasons
  };
  return {
    ...baseModel,
    expression: buildExpression(baseModel),
    diagnostics
  };
}

export function predictBlackBoxReluNdMultiHypothesisBoundaryContinuation(
  model: MultiHypothesisBoundaryContinuationModel,
  input: number[]
): number {
  if (input.length !== model.dimension) throw new Error('MBCD prediction dimension mismatch');
  let value = model.baseIntercept + dot(model.baseWeights, input);
  for (const term of model.terms) {
    value += term.readout * Math.max(0, dot(term.normal, input) + term.offset);
  }
  return Number.isFinite(value) ? value : 0;
}
