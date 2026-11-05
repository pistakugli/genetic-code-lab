import { decompileBlackBoxRelu1D } from './blackbox-relu-decompiler';
import { decompileNoisyBlackBoxRelu1D } from './noisy-blackbox-relu-decompiler';

export type BlackBoxReluNdTerm = {
  normal: number[];
  offset: number;
  readout: number;
  support: number;
  residual: number;
};

export type AdaptiveStageADiagnostics = {
  methodSelected: 'star-probe' | 'ransac' | 'hybrid' | 'adaptive';
  starProbesAttempted: number;
  starProbesAccepted: number;
  starProbesExecuted: number;
  maxStarProbes: number | null;
  starProbeOracleCalls: number;
  starProbesRejectedContamination: number;
  starProbesRejectedLowAmplitude: number;
  starProbesRejectedLowSupport: number;
  starProbesRejectedBudget: number;
  ransacFallbackTriggered: boolean;
  ransacFallbackPlanesFound: number;
  totalOracleCalls: number;
  budgetConsumed: number;
  budgetLimit: number | null;
  budgetExhausted: boolean;
  stoppedPhase: OracleBudgetPhase | null;
  oracleCallsByPhase: Record<OracleBudgetPhase, number>;
  candidatesRecovered: number;
  reconstructionComplete: boolean;
  reasons: string[];
};

type OracleBudgetPhase = 'kink-collection' | 'star-probe' | 'fit-sampling';

class OracleBudgetExceeded extends Error {
  constructor() {
    super('oracle budget exhausted');
    this.name = 'OracleBudgetExceeded';
  }
}

export type BlackBoxReluNdModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  lineCount: number;
  samplesPerLine: number;
  baseIntercept: number;
  baseWeights: number[];
  terms: BlackBoxReluNdTerm[];
  expression: string;
  kinkPointCount: number;
  adaptiveDiagnostics?: AdaptiveStageADiagnostics;
  reconstructionComplete?: boolean;
};

export type BlackBoxReluNdLineProbeConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  termCount: number;
  stageAMethod?: 'ransac' | 'star-probe' | 'hybrid' | 'adaptive';
  starProbeDelta?: number;
  starProbeEpsilon?: number;
  starProbeMinJump?: number;
  starProbeAffineTolerance?: number;
  /** Maximum primary kink candidates tested by Star-Probe. Each candidate is
   * one probe; its gradient stencil and any normal-consistency stencils are
   * included in that probe's oracle-call cost. Undefined is unlimited. */
  adaptiveMaxStarProbes?: number;
  adaptiveMinStarSupport?: number;
  adaptiveProjectionTolerance?: number;
  /** Minimum ratio of inlier kinks whose local star-probe normal is consistent
   * with the candidate normal.  Only applied in 'star-probe', 'hybrid', and
   * 'adaptive' modes.  Default 0.55.  Set to 0 to disable. */
  adaptiveNormalConsistencyRatio?: number;
  /** Maximum number of inlier kinks to probe when verifying normal consistency.
   * Default 5.  Each probe costs (4*d + 2) oracle calls. */
  adaptiveNormalConsistencyMaxProbe?: number;
  oracleBudget?: number;
  lineCount?: number;
  samplesPerLine?: number;
  querySamples?: number;
  aggregation?: 'median' | 'trimmed-mean' | 'mean';
  trimFraction?: number;
  minSegmentPoints?: number;
  minAbsLineDeltaSlope?: number;
  randomSeed?: number;
  ransacIterations?: number;
  inlierTolerance?: number;
  minSupport?: number;
  slopeTolerance?: number;
  kinkMergeTolerance?: number;
  ridge?: number;
};

type KinkPoint = {
  point: number[];
  deltaSlope: number;
  direction?: number[];
};

type PlaneCandidate = {
  normal: number[];
  offset: number;
  inliers: KinkPoint[];
  residual: number;
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
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = norm(values);
  if (length <= 1e-14) throw new Error('cannot normalize near-zero vector');
  return values.map(value => value / length);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function aggregate(values: number[], mode: 'median' | 'trimmed-mean' | 'mean', trimFraction: number): number {
  if (mode === 'mean') return mean(values);
  const sorted = [...values].sort((a, b) => a - b);
  if (mode === 'median') return sorted[Math.floor(sorted.length / 2)];
  const trim = Math.min(Math.floor(sorted.length / 2), Math.floor(sorted.length * trimFraction));
  return mean(sorted.slice(trim, sorted.length - trim));
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function randomDirection(random: () => number, dimension: number): number[] {
  const values = Array.from({ length: dimension }, () => randomBetween(random, -1, 1));
  if (norm(values) <= 1e-12) values[0] = 1;
  return normalize(values);
}

function randomPoint(random: () => number, ranges: Array<[number, number]>): number[] {
  return ranges.map(([min, max]) => randomBetween(random, min, max));
}

function lineBounds(origin: number[], direction: number[], ranges: Array<[number, number]>): [number, number] {
  let minT = -Infinity;
  let maxT = Infinity;
  for (let i = 0; i < origin.length; i++) {
    const d = direction[i];
    if (Math.abs(d) < 1e-14) continue;
    const a = (ranges[i][0] - origin[i]) / d;
    const b = (ranges[i][1] - origin[i]) / d;
    minT = Math.max(minT, Math.min(a, b));
    maxT = Math.min(maxT, Math.max(a, b));
  }
  if (!Number.isFinite(minT) || !Number.isFinite(maxT) || !(maxT > minT)) {
    throw new Error('failed to derive valid random-line bounds');
  }
  const margin = (maxT - minT) * 0.04;
  return [minT + margin, maxT - margin];
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
    if (Math.abs(divisor) < 1e-13) throw new Error('line-probe linear system is singular');
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

function nullVector(rows: number[][], dimension: number): number[] | null {
  for (let free = 0; free < dimension; free++) {
    const used = Array.from({ length: dimension }, (_, index) => index).filter(index => index !== free);
    const matrix = rows.map(row => used.map(index => row[index]));
    const rhs = rows.map(row => -row[free]);
    try {
      const solved = solveSystem(matrix, rhs);
      const vector = new Array(dimension).fill(0);
      vector[free] = 1;
      for (let i = 0; i < used.length; i++) vector[used[i]] = solved[i];
      return normalize(vector);
    } catch {
      // Try a different free variable; degenerate point samples are common in RANSAC.
    }
  }
  return null;
}

function fitPlaneFromPoints(points: number[][]): { normal: number[]; offset: number } | null {
  const dimension = points[0]?.length ?? 0;
  if (points.length < dimension) return null;
  const anchor = points[0];
  const rows = points.slice(1, dimension).map(point => point.map((value, index) => value - anchor[index]));
  const normal = nullVector(rows, dimension);
  if (!normal) return null;
  let offset = -dot(normal, anchor);
  const signIndex = normal.findIndex(value => Math.abs(value) > 1e-10);
  if (signIndex >= 0 && normal[signIndex] < 0) {
    for (let i = 0; i < normal.length; i++) normal[i] = -normal[i];
    offset = -offset;
  }
  return { normal, offset };
}

function planeDistance(plane: { normal: number[]; offset: number }, point: number[]): number {
  return Math.abs(dot(plane.normal, point) + plane.offset);
}

function sampleDistinct(random: () => number, max: number, count: number): number[] {
  const picked = new Set<number>();
  while (picked.size < count && picked.size < max) picked.add(Math.floor(random() * max));
  return [...picked];
}

function findPlane(
  points: KinkPoint[],
  dimension: number,
  random: () => number,
  iterations: number,
  tolerance: number,
  minSupport: number
): PlaneCandidate | null {
  let best: PlaneCandidate | null = null;
  if (points.length < dimension) return null;

  for (let i = 0; i < iterations; i++) {
    const indexes = sampleDistinct(random, points.length, dimension);
    if (indexes.length < dimension) continue;
    const plane = fitPlaneFromPoints(indexes.map(index => points[index].point));
    if (!plane) continue;
    const inliers = points.filter(point => planeDistance(plane, point.point) <= tolerance);
    if (inliers.length < minSupport) continue;
    const residual = mean(inliers.map(point => planeDistance(plane, point.point)));
    if (
      !best
      || inliers.length > best.inliers.length
      || (inliers.length === best.inliers.length && residual < best.residual)
    ) {
      best = { ...plane, inliers, residual };
    }
  }
  return best ? refinePlaneFromInliers(best, points, dimension, tolerance, minSupport) : null;
}

function fitPlaneFromInliers(points: KinkPoint[], dimension: number, referenceNormal: number[]): { normal: number[]; offset: number } | null {
  const dependent = referenceNormal
    .map((value, index) => ({ index, value: Math.abs(value) }))
    .sort((a, b) => b.value - a.value)[0]?.index ?? 0;
  const independent = Array.from({ length: dimension }, (_, index) => index).filter(index => index !== dependent);
  const width = dimension;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const point of points) {
    const row = [1, ...independent.map(index => point.point[index])];
    const target = point.point[dependent];
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * target;
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += 1e-10;
  try {
    const coefficients = solveSystem(xtx, xty);
    const normal = new Array(dimension).fill(0);
    normal[dependent] = 1;
    for (let i = 0; i < independent.length; i++) normal[independent[i]] = -coefficients[1 + i];
    let normalized = normalize(normal);
    let offset = -coefficients[0] / norm(normal);
    if (dot(normalized, referenceNormal) < 0) {
      normalized = normalized.map(value => -value);
      offset = -offset;
    }
    return { normal: normalized, offset };
  } catch {
    return null;
  }
}

function refinePlaneFromInliers(
  candidate: PlaneCandidate,
  allPoints: KinkPoint[],
  dimension: number,
  tolerance: number,
  minSupport: number
): PlaneCandidate {
  const refined = fitPlaneFromInliers(candidate.inliers, dimension, candidate.normal);
  if (!refined) return candidate;
  const inliers = allPoints.filter(point => planeDistance(refined, point.point) <= tolerance);
  if (inliers.length < minSupport) return candidate;
  const residual = mean(inliers.map(point => planeDistance(refined, point.point)));
  if (inliers.length < candidate.inliers.length && residual > candidate.residual) return candidate;
  return { ...refined, inliers, residual };
}

function collectKinkPoints(
  oracle: (input: number[]) => number,
  config: Required<Pick<BlackBoxReluNdLineProbeConfig, 'dimension' | 'ranges' | 'termCount' | 'lineCount' | 'samplesPerLine' | 'querySamples' | 'aggregation' | 'trimFraction' | 'minSegmentPoints' | 'minAbsLineDeltaSlope' | 'slopeTolerance' | 'kinkMergeTolerance'>>,
  random: () => number
): KinkPoint[] {
  const points: KinkPoint[] = [];
  for (let i = 0; i < config.lineCount; i++) {
    const origin = randomPoint(random, config.ranges);
    const direction = randomDirection(random, config.dimension);
    const [min, max] = lineBounds(origin, direction, config.ranges);
    if (!(max > min)) continue;
    const lineOracle = (t: number): number => oracle(origin.map((value, index) => value + t * direction[index]));
    let lineModel: ReturnType<typeof decompileBlackBoxRelu1D>;
    try {
      lineModel = config.querySamples > 1
        ? decompileNoisyBlackBoxRelu1D(lineOracle, {
          min,
          max,
          maxTerms: config.termCount,
          samples: config.samplesPerLine,
          querySamples: config.querySamples,
          aggregation: config.aggregation,
          trimFraction: config.trimFraction,
          minSegmentPoints: config.minSegmentPoints
        })
        : decompileBlackBoxRelu1D(lineOracle, {
          min,
          max,
          samples: config.samplesPerLine,
          slopeTolerance: config.slopeTolerance,
          kinkMergeTolerance: config.kinkMergeTolerance,
          minPlateauIntervals: 4
        });
    } catch (error) {
      if (!(error instanceof OracleBudgetExceeded)) throw error;
      break;
    }
    for (const term of lineModel.terms) {
      if (term.kink > min && term.kink < max && Math.abs(term.deltaSlope) >= config.minAbsLineDeltaSlope) {
        points.push({
          point: origin.map((value, index) => value + term.kink * direction[index]),
          deltaSlope: term.deltaSlope,
          direction
        });
      }
    }
  }
  return points;
}

export type StarProbeCandidate = {
  normal: number[];
  offset: number;
  kink: KinkPoint;
  jumpNorm: number;
  affineConsistent: boolean;
};

function estimateGradientAtPoint(
  oracle: (input: number[]) => number,
  center: number[],
  epsilon: number,
  affineTolerance: number
): { gradient: number[]; affineConsistent: boolean } {
  const dimension = center.length;
  const gradient = new Array(dimension).fill(0);
  const centerVal = oracle(center);
  let affineConsistent = true;

  for (let j = 0; j < dimension; j++) {
    const plus = [...center];
    const minus = [...center];
    plus[j] += epsilon;
    minus[j] -= epsilon;
    const yPlus = oracle(plus);
    const yMinus = oracle(minus);

    gradient[j] = (yPlus - yMinus) / (2 * epsilon);

    const curvature = Math.abs(yPlus + yMinus - 2 * centerVal);
    if (curvature > affineTolerance) {
      affineConsistent = false;
    }
  }

  return { gradient, affineConsistent };
}

export type StarProbeRejectionReason =
  | 'affine_curvature_violation'
  | 'sub_threshold_jump_norm'
  | 'projection_inconsistency'
  | 'zero_direction';

export type StarProbeOutcome = {
  candidate: StarProbeCandidate | null;
  rejectionReason?: StarProbeRejectionReason;
};

export type StarProbeDiagnosticsAccumulator = {
  attempted: number;
  accepted: number;
  rejectedCurvature: number;
  rejectedLowJump: number;
  rejectedProjection: number;
  rejectedLowSupport: number;
  rejectedBudget: number;
  reasons: string[];
};

export function probeKinkNormalStarDetailed(
  oracle: (input: number[]) => number,
  kink: KinkPoint,
  direction: number[],
  delta: number,
  epsilon: number,
  minJump: number,
  affineTolerance: number,
  projectionTolerance: number = 0.3
): StarProbeOutcome {
  const dirLength = norm(direction);
  if (dirLength < 1e-12) return { candidate: null, rejectionReason: 'zero_direction' };
  const unitDir = direction.map(v => v / dirLength);

  const xPlus = kink.point.map((v, i) => v + delta * unitDir[i]);
  const xMinus = kink.point.map((v, i) => v - delta * unitDir[i]);

  const plusRes = estimateGradientAtPoint(oracle, xPlus, epsilon, affineTolerance);
  const minusRes = estimateGradientAtPoint(oracle, xMinus, epsilon, affineTolerance);

  if (!plusRes.affineConsistent || !minusRes.affineConsistent) {
    return { candidate: null, rejectionReason: 'affine_curvature_violation' };
  }

  const deltaG = plusRes.gradient.map((v, i) => v - minusRes.gradient[i]);
  const jumpNorm = norm(deltaG);
  if (jumpNorm < minJump) {
    return { candidate: null, rejectionReason: 'sub_threshold_jump_norm' };
  }

  // Directional projection consistency check: Delta g projected along the probe direction
  // must match the 1D slope jump across the kink, ruling out multi-hyperplane contaminated jumps.
  const expectedLineJump = kink.deltaSlope / dirLength;
  const projectedJump = dot(deltaG, unitDir);
  const projError = Math.abs(Math.abs(projectedJump) - Math.abs(expectedLineJump));
  if (projError / Math.max(1e-4, Math.abs(expectedLineJump)) > projectionTolerance) {
    return { candidate: null, rejectionReason: 'projection_inconsistency' };
  }

  let normal = normalize(deltaG);
  let offset = -dot(normal, kink.point);

  const signIndex = normal.findIndex(v => Math.abs(v) > 1e-10);
  if (signIndex >= 0 && normal[signIndex] < 0) {
    normal = normal.map(v => -v);
    offset = -offset;
  }

  return {
    candidate: {
      normal,
      offset,
      kink,
      jumpNorm,
      affineConsistent: true
    }
  };
}

export function probeKinkNormalStar(
  oracle: (input: number[]) => number,
  kink: KinkPoint,
  direction: number[],
  delta: number,
  epsilon: number,
  minJump: number,
  affineTolerance: number,
  projectionTolerance: number = 0.3
): StarProbeCandidate | null {
  return probeKinkNormalStarDetailed(
    oracle,
    kink,
    direction,
    delta,
    epsilon,
    minJump,
    affineTolerance,
    projectionTolerance
  ).candidate;
}

/**
 * Cross-validate a candidate plane's normal by probing a sample of its inlier kink points.
 * For each probed inlier, we compute the local star-probe normal and check whether it is
 * consistent with the candidate normal (cosine similarity >= normalThreshold).
 *
 * Returns the fraction of probed inliers whose local normal is consistent.
 * A true hyperplane will have a high fraction (typically >= 0.6).
 * A contaminated vertex plane (where multiple hyperplanes meet) will have a
 * substantially lower fraction because different inlier kinks belong to different
 * true hyperplanes and give inconsistent local normals.
 */
function checkNormalConsistency(
  oracle: (input: number[]) => number,
  candidatePlane: { normal: number[]; offset: number },
  kinkPoints: KinkPoint[],
  delta: number,
  epsilon: number,
  minJump: number,
  affineTolerance: number,
  inlierTolerance: number,
  maxProbe: number,
  normalThreshold: number
): number {
  const inliers = kinkPoints.filter(kp => planeDistance(candidatePlane, kp.point) <= inlierTolerance);
  if (inliers.length === 0) return 0;

  const probeSet = inliers.slice(0, maxProbe);
  let consistent = 0;
  for (const kink of probeSet) {
    const direction = kink.direction ?? kink.point;
    const { candidate } = probeKinkNormalStarDetailed(
      oracle, kink, direction, delta, epsilon, minJump, affineTolerance, 1.0 // skip proj check here
    );
    if (!candidate) continue;
    const cosSim = Math.abs(dot(candidate.normal, candidatePlane.normal));
    if (cosSim >= normalThreshold) consistent++;
  }
  return consistent / probeSet.length;
}

function findPlanesViaStarProbe(
  oracle: (input: number[]) => number,
  kinkPoints: KinkPoint[],
  dimension: number,
  targetCount: number,
  delta: number,
  epsilon: number,
  minJump: number,
  affineTolerance: number,
  inlierTolerance: number,
  minStarSupport: number = 2,
  projectionTolerance: number = 0.3,
  diagnostics?: StarProbeDiagnosticsAccumulator,
  checkBudget?: () => boolean,
  normalConsistencyRatio: number = 0,
  normalConsistencyMaxProbe: number = 5,
  normalConsistencyThreshold: number = 0.85,
  maxStarProbes?: number
): PlaneCandidate[] {
  const planes: PlaneCandidate[] = [];

  for (const kink of kinkPoints) {
    if (planes.length >= targetCount) break;
    if (maxStarProbes !== undefined && diagnostics && diagnostics.attempted >= maxStarProbes) {
      diagnostics.reasons.push('adaptive_max_star_probes_reached');
      break;
    }
    if (checkBudget && !checkBudget()) {
      if (diagnostics) {
        diagnostics.rejectedBudget++;
        diagnostics.reasons.push('oracle_budget_cap_reached_stopping_star_probes');
      }
      break;
    }

    if (diagnostics) diagnostics.attempted++;
    try {
      const direction = kink.direction ?? kink.point;
      const { candidate: star, rejectionReason } = probeKinkNormalStarDetailed(
        oracle,
        kink,
        direction,
        delta,
        epsilon,
        minJump,
        affineTolerance,
        projectionTolerance
      );

      if (!star) {
        if (diagnostics && rejectionReason) {
          if (rejectionReason === 'affine_curvature_violation') diagnostics.rejectedCurvature++;
          else if (rejectionReason === 'sub_threshold_jump_norm') diagnostics.rejectedLowJump++;
          else if (rejectionReason === 'projection_inconsistency') diagnostics.rejectedProjection++;
        }
        continue;
      }

      const alreadyFound = planes.some(existing => {
        const cosSim = Math.abs(dot(star.normal, existing.normal));
        if (cosSim > 0.985) {
          const sign = dot(star.normal, existing.normal) > 0 ? 1 : -1;
          return Math.abs(sign * star.offset - existing.offset) < 0.15;
        }
        return false;
      });

      if (alreadyFound) continue;

      const candidatePlane = { normal: star.normal, offset: star.offset };
      const inliers = kinkPoints.filter(p => planeDistance(candidatePlane, p.point) <= inlierTolerance);

      if (inliers.length < minStarSupport && kinkPoints.length >= minStarSupport * 2) {
        if (diagnostics) diagnostics.rejectedLowSupport++;
        continue;
      }

      if (normalConsistencyRatio > 0 && inliers.length >= 2) {
        const ratio = checkNormalConsistency(
          oracle, candidatePlane, kinkPoints,
          delta, epsilon, minJump, affineTolerance,
          inlierTolerance, normalConsistencyMaxProbe, normalConsistencyThreshold
        );
        if (ratio < normalConsistencyRatio) {
          if (diagnostics) {
            diagnostics.rejectedProjection++;
            diagnostics.reasons.push(`normal_consistency_check_failed_ratio=${ratio.toFixed(3)}`);
          }
          continue;
        }
      }

      if (diagnostics) diagnostics.accepted++;

      const residual = inliers.length > 0
        ? mean(inliers.map(p => planeDistance(candidatePlane, p.point)))
        : planeDistance(candidatePlane, kink.point);

      planes.push({
        normal: star.normal,
        offset: star.offset,
        inliers: inliers.length > 0 ? inliers : [kink],
        residual
      });
    } catch (error) {
      if (!(error instanceof OracleBudgetExceeded)) throw error;
      if (diagnostics) {
        diagnostics.rejectedBudget++;
        diagnostics.reasons.push('oracle_budget_exhausted_during_star_probe');
      }
      break;
    }
  }

  return planes;
}

function fitCoefficients(
  samples: Array<{ input: number[]; target: number }>,
  terms: Array<{ normal: number[]; offset: number }>,
  ridge: number
): { coefficients: number[]; mse: number } {
  const dimension = samples[0].input.length;
  const width = 1 + dimension + terms.length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const sample of samples) {
    const row = [
      1,
      ...sample.input,
      ...terms.map(term => Math.max(0, dot(term.normal, sample.input) + term.offset))
    ];
    for (let i = 0; i < width; i++) {
      xty[i] += row[i] * sample.target;
      for (let j = 0; j < width; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += ridge;
  const coefficients = solveSystem(xtx, xty);
  let sse = 0;
  for (const sample of samples) {
    const predicted = predictFromCoefficients(sample.input, coefficients, terms);
    const error = predicted - sample.target;
    sse += error * error;
  }
  return { coefficients, mse: sse / Math.max(1, samples.length) };
}

function predictFromCoefficients(
  input: number[],
  coefficients: number[],
  terms: Array<{ normal: number[]; offset: number }>
): number {
  let value = coefficients[0];
  for (let i = 0; i < input.length; i++) value += coefficients[1 + i] * input[i];
  for (let i = 0; i < terms.length; i++) {
    value += coefficients[1 + input.length + i] * Math.max(0, dot(terms[i].normal, input) + terms[i].offset);
  }
  return Number.isFinite(value) ? value : 0;
}

function makeFitSamples(
  oracle: (input: number[]) => number,
  ranges: Array<[number, number]>,
  random: () => number,
  count: number,
  querySamples: number,
  aggregationMode: 'median' | 'trimmed-mean' | 'mean',
  trimFraction: number
): { samples: Array<{ input: number[]; target: number }>; complete: boolean } {
  const samples: Array<{ input: number[]; target: number }> = [];
  for (let i = 0; i < count; i++) {
    const input = randomPoint(random, ranges);
    const values: number[] = [];
    try {
      for (let j = 0; j < querySamples; j++) values.push(oracle(input));
    } catch (error) {
      if (!(error instanceof OracleBudgetExceeded)) throw error;
      return { samples, complete: false };
    }
    samples.push({ input, target: aggregate(values, aggregationMode, trimFraction) });
  }
  return { samples, complete: true };
}

function buildExpression(model: Omit<BlackBoxReluNdModel, 'expression'>): string {
  const variables = Array.from({ length: model.dimension }, (_, index) => `x${index}`);
  const parts = [
    formatNumber(model.baseIntercept),
    ...model.baseWeights.map((weight, index) => `${formatNumber(weight)} * ${variables[index]}`)
  ];
  for (const term of model.terms) {
    const inside = [
      ...term.normal.map((weight, index) => `${formatNumber(weight)} * ${variables[index]}`),
      formatNumber(term.offset)
    ].join(' + ');
    parts.push(`${formatNumber(term.readout)} * max(0, ${inside})`);
  }
  return parts.join(' + ');
}

export function decompileBlackBoxReluNdLineProbe(
  oracle: (input: number[]) => number,
  config: BlackBoxReluNdLineProbeConfig
): BlackBoxReluNdModel {
  const dimension = Math.max(2, Math.floor(config.dimension));
  if (config.ranges.length !== dimension) throw new Error('ND line-probe decompiler requires one range per dimension');
  if (config.ranges.some(([min, max]) => !(max > min))) throw new Error('ND line-probe decompiler requires valid ranges');
  if (config.termCount < 1) throw new Error('ND line-probe decompiler requires at least one term');
  if (config.oracleBudget !== undefined
    && (!Number.isSafeInteger(config.oracleBudget) || config.oracleBudget < 0)) {
    throw new Error('ND line-probe decompiler requires oracleBudget to be a non-negative safe integer');
  }
  if (config.adaptiveMaxStarProbes !== undefined
    && (!Number.isSafeInteger(config.adaptiveMaxStarProbes) || config.adaptiveMaxStarProbes < 0)) {
    throw new Error('ND line-probe decompiler requires adaptiveMaxStarProbes to be a non-negative safe integer');
  }

  const budget = config.oracleBudget;
  let totalOracleCalls = 0;
  let currentPhase: OracleBudgetPhase = 'kink-collection';
  let stoppedPhase: OracleBudgetPhase | null = null;
  let starProbeOracleCalls = 0;
  const oracleCallsByPhase: Record<OracleBudgetPhase, number> = {
    'kink-collection': 0,
    'star-probe': 0,
    'fit-sampling': 0
  };
  const markBudgetExhausted = (phase: OracleBudgetPhase): void => {
    stoppedPhase ??= phase;
  };
  const countedOracle = (input: number[]): number => {
    if (budget !== undefined && totalOracleCalls >= budget) {
      markBudgetExhausted(currentPhase);
      throw new OracleBudgetExceeded();
    }
    totalOracleCalls++;
    oracleCallsByPhase[currentPhase]++;
    if (currentPhase === 'star-probe') starProbeOracleCalls++;
    return oracle(input);
  };

  const lineCount = Math.max(8, Math.floor(config.lineCount ?? 180));
  const samplesPerLine = Math.max(96, Math.floor(config.samplesPerLine ?? 801));
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregationMode = config.aggregation ?? 'median';
  const trimFraction = Math.max(0, Math.min(0.45, config.trimFraction ?? 0.2));
  const minSegmentPoints = Math.max(4, Math.floor(config.minSegmentPoints ?? 18));
  const minAbsLineDeltaSlope = Math.max(0, config.minAbsLineDeltaSlope ?? 0);
  const random = makeRandom(config.randomSeed ?? 260528);
  let kinkPoints: KinkPoint[] = [];
  try {
    kinkPoints = collectKinkPoints(
      countedOracle,
      {
        dimension,
        ranges: config.ranges,
        termCount: Math.floor(config.termCount),
        lineCount,
        samplesPerLine,
        querySamples,
        aggregation: aggregationMode,
        trimFraction,
        minSegmentPoints,
        minAbsLineDeltaSlope,
        slopeTolerance: config.slopeTolerance ?? 1e-7,
        kinkMergeTolerance: config.kinkMergeTolerance ?? 1e-5
      },
      random
    );
  } catch (error) {
    if (!(error instanceof OracleBudgetExceeded)) throw error;
  }

  const stageAMethod = config.stageAMethod ?? 'ransac';
  const isAdaptive = stageAMethod === 'adaptive';
  const starProbeDelta = config.starProbeDelta ?? 0.015;
  const starProbeEpsilon = config.starProbeEpsilon ?? 1e-4;
  const starProbeMinJump = config.starProbeMinJump ?? 0.05;
  const starProbeAffineTolerance = config.starProbeAffineTolerance ?? 1e-4;
  const inlierTolerance = config.inlierTolerance ?? 1e-5;
  const ransacIterations = Math.max(64, Math.floor(config.ransacIterations ?? 1800));
  const minSupport = Math.max(dimension + 1, Math.floor(config.minSupport ?? Math.max(12, lineCount * 0.35)));
  const minStarSupport = config.adaptiveMinStarSupport ?? Math.max(2, Math.min(dimension, Math.floor(minSupport * 0.25)));
  const projectionTolerance = config.adaptiveProjectionTolerance ?? 0.3;
  // Normal consistency check parameters (only active in adaptive mode by default;
  // star-probe and hybrid can opt-in via adaptiveNormalConsistencyRatio config).
  const normalConsistencyRatio = config.adaptiveNormalConsistencyRatio ?? (isAdaptive ? 0.55 : 0);
  const normalConsistencyMaxProbe = Math.max(1, config.adaptiveNormalConsistencyMaxProbe ?? 5);
  const maxStarProbes = config.adaptiveMaxStarProbes;

  const starDiagnostics: StarProbeDiagnosticsAccumulator = {
    attempted: 0,
    accepted: 0,
    rejectedCurvature: 0,
    rejectedLowJump: 0,
    rejectedProjection: 0,
    rejectedLowSupport: 0,
    rejectedBudget: 0,
    reasons: []
  };

  const checkBudget = (): boolean => {
    if (budget === undefined || totalOracleCalls < budget) return true;
    markBudgetExhausted(currentPhase);
    return false;
  };

  let planes: PlaneCandidate[] = [];

  if (stageAMethod === 'star-probe' || stageAMethod === 'hybrid' || isAdaptive) {
    currentPhase = 'star-probe';
    planes = findPlanesViaStarProbe(
      countedOracle,
      kinkPoints,
      dimension,
      Math.floor(config.termCount),
      starProbeDelta,
      starProbeEpsilon,
      starProbeMinJump,
      starProbeAffineTolerance,
      inlierTolerance,
      minStarSupport,
      projectionTolerance,
      starDiagnostics,
      checkBudget,
      normalConsistencyRatio,
      normalConsistencyMaxProbe,
      0.85,
      maxStarProbes
    );
  }

  let ransacFallbackTriggered = false;
  let ransacFallbackPlanesFound = 0;

  if (planes.length < Math.floor(config.termCount) && (stageAMethod === 'ransac' || stageAMethod === 'hybrid' || isAdaptive)) {
    if (isAdaptive) {
      ransacFallbackTriggered = true;
      if (starDiagnostics.rejectedLowJump > 0) {
        starDiagnostics.reasons.push('sub_threshold_amplitude_detected_fallback_to_ransac');
      }
      if (starDiagnostics.rejectedCurvature > 0 || starDiagnostics.rejectedProjection > 0) {
        starDiagnostics.reasons.push('boundary_intersection_contamination_detected_fallback_to_ransac');
      }
      if (starDiagnostics.rejectedLowSupport > 0) {
        starDiagnostics.reasons.push('insufficient_inlier_support_fallback_to_ransac');
      }
      if (starDiagnostics.rejectedBudget > 0) {
        starDiagnostics.reasons.push('budget_limit_triggered_fallback_to_ransac');
      }
      if (planes.length === 0) {
        starDiagnostics.reasons.push('no_clean_star_candidates_full_ransac_fallback');
      }
    }

    let remaining = [...kinkPoints];
    for (const discovered of planes) {
      remaining = remaining.filter(point => planeDistance(discovered, point.point) > inlierTolerance);
    }
    const needed = Math.floor(config.termCount) - planes.length;
    // In adaptive mode with a small kink pool (e.g. small-amplitude neurons produce few kinks),
    // the global minSupport may exceed the available pool.  Use an adaptive floor so RANSAC
    // can still find the hyperplane without lowering global quality on large pools.
    const effectiveMinSupport = isAdaptive && remaining.length > 0 && remaining.length < minSupport
      ? Math.max(dimension + 1, Math.floor(remaining.length * 0.5))
      : minSupport;
    for (let i = 0; i < needed; i++) {
      const plane = findPlane(remaining, dimension, random, ransacIterations, inlierTolerance, effectiveMinSupport);
      if (!plane) break;
      planes.push(plane);
      ransacFallbackPlanesFound++;
      remaining = remaining.filter(point => planeDistance(plane, point.point) > inlierTolerance);
    }
  }

  if (isAdaptive && planes.length >= Math.floor(config.termCount) && !ransacFallbackTriggered) {
    starDiagnostics.reasons.push('all_terms_recovered_via_star_probe_with_verified_support');
  }

  const methodSelected: 'star-probe' | 'ransac' | 'hybrid' | 'adaptive' = isAdaptive
    ? (ransacFallbackTriggered ? (planes.length > ransacFallbackPlanesFound ? 'hybrid' : 'ransac') : 'star-probe')
    : stageAMethod;

  const budgetAlreadySpent = budget !== undefined && totalOracleCalls >= budget;
  if (budgetAlreadySpent && stoppedPhase === null) markBudgetExhausted('fit-sampling');

  if (planes.length !== Math.floor(config.termCount) && !budgetAlreadySpent) {
    throw new Error(`ND line-probe decompiler recovered ${planes.length}/${Math.floor(config.termCount)} hyperplanes`);
  }

  let fitSampling = { samples: [] as Array<{ input: number[]; target: number }>, complete: false };
  if (!budgetAlreadySpent) {
    currentPhase = 'fit-sampling';
    fitSampling = makeFitSamples(
      countedOracle,
      config.ranges,
      random,
      Math.max(256, 80 * (dimension + planes.length)),
      querySamples,
      aggregationMode,
      trimFraction
    );
  }

  const geometry = planes.map(plane => ({ normal: plane.normal, offset: plane.offset }));
  const fitWidth = 1 + dimension + geometry.length;
  const fitComplete = fitSampling.complete && fitSampling.samples.length >= fitWidth;
  const fit = fitComplete
    ? fitCoefficients(fitSampling.samples, geometry, config.ridge ?? 1e-10)
    : { coefficients: new Array(fitWidth).fill(0), mse: 0 };
  const reconstructionComplete = planes.length === Math.floor(config.termCount) && fitComplete;
  const terms: BlackBoxReluNdTerm[] = planes.map((plane, index) => ({
    normal: plane.normal,
    offset: plane.offset,
    readout: fit.coefficients[1 + dimension + index],
    support: plane.inliers.length,
    residual: plane.residual
  }));
  const diagnosticsEnabled = isAdaptive || budget !== undefined;
  const budgetExhausted = budget !== undefined && totalOracleCalls >= budget;
  const adaptiveDiagnostics: AdaptiveStageADiagnostics | undefined = diagnosticsEnabled
    ? {
        methodSelected,
        starProbesAttempted: starDiagnostics.attempted,
        starProbesAccepted: starDiagnostics.accepted,
        starProbesExecuted: starDiagnostics.attempted,
        maxStarProbes: maxStarProbes ?? null,
        starProbeOracleCalls,
        starProbesRejectedContamination: starDiagnostics.rejectedCurvature + starDiagnostics.rejectedProjection,
        starProbesRejectedLowAmplitude: starDiagnostics.rejectedLowJump,
        starProbesRejectedLowSupport: starDiagnostics.rejectedLowSupport,
        starProbesRejectedBudget: starDiagnostics.rejectedBudget,
        ransacFallbackTriggered,
        ransacFallbackPlanesFound,
        totalOracleCalls,
        budgetConsumed: totalOracleCalls,
        budgetLimit: budget ?? null,
        budgetExhausted,
        stoppedPhase,
        oracleCallsByPhase,
        candidatesRecovered: planes.length,
        reconstructionComplete,
        reasons: starDiagnostics.reasons
      }
    : undefined;
  const modelBase = {
    dimension,
    ranges: config.ranges,
    lineCount,
    samplesPerLine,
    baseIntercept: fit.coefficients[0],
    baseWeights: fit.coefficients.slice(1, 1 + dimension),
    terms,
    kinkPointCount: kinkPoints.length,
    adaptiveDiagnostics,
    ...(budget !== undefined ? { reconstructionComplete } : {})
  };
  return { ...modelBase, expression: buildExpression(modelBase) };
}

export function predictBlackBoxReluNdLineProbeModel(model: BlackBoxReluNdModel, input: number[]): number {
  if (input.length !== model.dimension) throw new Error('ND line-probe prediction dimension mismatch');
  let value = model.baseIntercept;
  for (let i = 0; i < model.dimension; i++) value += model.baseWeights[i] * input[i];
  for (const term of model.terms) {
    value += term.readout * Math.max(0, dot(term.normal, input) + term.offset);
  }
  return Number.isFinite(value) ? value : 0;
}

export function predictBlackBoxReluNdLineProbeBatch(model: BlackBoxReluNdModel, inputs: number[][]): number[] {
  return inputs.map(input => predictBlackBoxReluNdLineProbeModel(model, input));
}
