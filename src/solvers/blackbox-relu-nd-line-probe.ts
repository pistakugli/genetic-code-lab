import { decompileBlackBoxRelu1D } from './blackbox-relu-decompiler';
import { decompileNoisyBlackBoxRelu1D } from './noisy-blackbox-relu-decompiler';

export type BlackBoxReluNdTerm = {
  normal: number[];
  offset: number;
  readout: number;
  support: number;
  residual: number;
};

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
};

export type BlackBoxReluNdLineProbeConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  termCount: number;
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
    const lineModel = config.querySamples > 1
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
      : decompileBlackBoxRelu1D(
        lineOracle,
        {
          min,
          max,
          samples: config.samplesPerLine,
          slopeTolerance: config.slopeTolerance,
          kinkMergeTolerance: config.kinkMergeTolerance,
          minPlateauIntervals: 4
        }
      );
    for (const term of lineModel.terms) {
      if (term.kink > min && term.kink < max && Math.abs(term.deltaSlope) >= config.minAbsLineDeltaSlope) {
        points.push({
          point: origin.map((value, index) => value + term.kink * direction[index]),
          deltaSlope: term.deltaSlope
        });
      }
    }
  }
  return points;
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
): Array<{ input: number[]; target: number }> {
  return Array.from({ length: count }, () => {
    const input = randomPoint(random, ranges);
    const values = Array.from({ length: querySamples }, () => oracle(input));
    return { input, target: aggregate(values, aggregationMode, trimFraction) };
  });
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

  const lineCount = Math.max(8, Math.floor(config.lineCount ?? 180));
  const samplesPerLine = Math.max(96, Math.floor(config.samplesPerLine ?? 801));
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregationMode = config.aggregation ?? 'median';
  const trimFraction = Math.max(0, Math.min(0.45, config.trimFraction ?? 0.2));
  const minSegmentPoints = Math.max(4, Math.floor(config.minSegmentPoints ?? 18));
  const minAbsLineDeltaSlope = Math.max(0, config.minAbsLineDeltaSlope ?? 0);
  const random = makeRandom(config.randomSeed ?? 260528);
  const kinkPoints = collectKinkPoints(
    oracle,
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

  const planes: PlaneCandidate[] = [];
  let remaining = [...kinkPoints];
  const ransacIterations = Math.max(64, Math.floor(config.ransacIterations ?? 1800));
  const inlierTolerance = config.inlierTolerance ?? 1e-5;
  const minSupport = Math.max(dimension + 1, Math.floor(config.minSupport ?? Math.max(12, lineCount * 0.35)));
  for (let i = 0; i < Math.floor(config.termCount); i++) {
    const plane = findPlane(remaining, dimension, random, ransacIterations, inlierTolerance, minSupport);
    if (!plane) break;
    planes.push(plane);
    remaining = remaining.filter(point => planeDistance(plane, point.point) > inlierTolerance);
  }

  if (planes.length !== Math.floor(config.termCount)) {
    throw new Error(`ND line-probe decompiler recovered ${planes.length}/${Math.floor(config.termCount)} hyperplanes`);
  }

  const fitSamples = makeFitSamples(
    oracle,
    config.ranges,
    random,
    Math.max(256, 80 * (dimension + planes.length)),
    querySamples,
    aggregationMode,
    trimFraction
  );
  const geometry = planes.map(plane => ({ normal: plane.normal, offset: plane.offset }));
  const fit = fitCoefficients(fitSamples, geometry, config.ridge ?? 1e-10);
  const terms: BlackBoxReluNdTerm[] = planes.map((plane, index) => ({
    normal: plane.normal,
    offset: plane.offset,
    readout: fit.coefficients[1 + dimension + index],
    support: plane.inliers.length,
    residual: plane.residual
  }));
  const modelBase = {
    dimension,
    ranges: config.ranges,
    lineCount,
    samplesPerLine,
    baseIntercept: fit.coefficients[0],
    baseWeights: fit.coefficients.slice(1, 1 + dimension),
    terms,
    kinkPointCount: kinkPoints.length
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
