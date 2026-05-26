import {
  BlackBoxReluModel,
  BlackBoxReluTerm,
  predictBlackBoxReluModel
} from './blackbox-relu-decompiler';

export type NoisyBlackBoxReluDecompilerConfig = {
  min: number;
  max: number;
  maxTerms: number;
  samples?: number;
  querySamples?: number;
  aggregation?: 'median' | 'trimmed-mean' | 'mean';
  trimFraction?: number;
  minSegmentPoints?: number;
};

type Point = { x: number; y: number };
type SegmentFit = {
  start: number;
  end: number;
  slope: number;
  intercept: number;
  sse: number;
};

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(8);
  return value.toPrecision(10).replace(/\.?0+$/, '');
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

function buildExpression(model: Omit<BlackBoxReluModel, 'expression'>): string {
  const parts = [
    formatNumber(model.baseIntercept),
    `${formatNumber(model.baseSlope)} * x`
  ];
  for (const term of model.terms) {
    const offset = term.kink >= 0
      ? `x - ${formatNumber(term.kink)}`
      : `x + ${formatNumber(-term.kink)}`;
    parts.push(`${formatNumber(term.deltaSlope)} * max(0, ${offset})`);
  }
  return parts.join(' + ');
}

function fitSegment(points: Point[], start: number, end: number): SegmentFit {
  const n = end - start;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = start; i < end; i++) {
    const { x, y } = points[i];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
  }
  const denom = n * sxx - sx * sx;
  const slope = Math.abs(denom) > 1e-12 ? (n * sxy - sx * sy) / denom : 0;
  const intercept = (sy - slope * sx) / n;
  const sse = Math.max(0, syy + slope * slope * sxx + n * intercept * intercept + 2 * slope * intercept * sx - 2 * slope * sxy - 2 * intercept * sy);
  return { start, end, slope, intercept, sse };
}

function segmentPoints(points: Point[], segmentCount: number, minSegmentPoints: number): SegmentFit[] {
  const n = points.length;
  if (segmentCount * minSegmentPoints > n) throw new Error('not enough samples for requested ReLU terms');
  const fits: Array<Array<SegmentFit | undefined>> = Array.from({ length: n + 1 }, () => new Array(n + 1));
  for (let start = 0; start < n; start++) {
    for (let end = start + minSegmentPoints; end <= n; end++) fits[start][end] = fitSegment(points, start, end);
  }

  const dp = Array.from({ length: segmentCount + 1 }, () => new Array(n + 1).fill(Infinity));
  const prev = Array.from({ length: segmentCount + 1 }, () => new Array(n + 1).fill(-1));
  for (let end = minSegmentPoints; end <= n; end++) {
    dp[1][end] = fits[0][end]?.sse ?? Infinity;
  }
  for (let segments = 2; segments <= segmentCount; segments++) {
    const minEnd = segments * minSegmentPoints;
    for (let end = minEnd; end <= n; end++) {
      for (let split = (segments - 1) * minSegmentPoints; split <= end - minSegmentPoints; split++) {
        const fit = fits[split][end];
        if (!fit) continue;
        const score = dp[segments - 1][split] + fit.sse;
        if (score < dp[segments][end]) {
          dp[segments][end] = score;
          prev[segments][end] = split;
        }
      }
    }
  }

  const result: SegmentFit[] = [];
  let end = n;
  for (let segments = segmentCount; segments >= 1; segments--) {
    const split = segments === 1 ? 0 : prev[segments][end];
    if (split < 0) throw new Error('failed to segment noisy ReLU samples');
    const fit = fits[split][end];
    if (!fit) throw new Error('missing segment fit');
    result.push(fit);
    end = split;
  }
  return result.reverse();
}

export function decompileNoisyBlackBoxRelu1D(
  oracle: (x: number) => number,
  config: NoisyBlackBoxReluDecompilerConfig
): BlackBoxReluModel {
  if (!(config.max > config.min)) throw new Error('noisy black-box ReLU decompiler requires max > min');
  const samples = Math.max(64, Math.floor(config.samples ?? 361));
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 21));
  const aggregation = config.aggregation ?? 'median';
  const trimFraction = Math.max(0, Math.min(0.45, config.trimFraction ?? 0.2));
  const minSegmentPoints = Math.max(4, Math.floor(config.minSegmentPoints ?? 18));
  const points: Point[] = Array.from({ length: samples }, (_, index) => {
    const x = config.min + ((config.max - config.min) * index) / (samples - 1);
    const values = Array.from({ length: querySamples }, () => oracle(x));
    return { x, y: aggregate(values, aggregation, trimFraction) };
  });

  const segments = segmentPoints(points, Math.max(1, Math.floor(config.maxTerms) + 1), minSegmentPoints);
  const terms: BlackBoxReluTerm[] = [];
  for (let i = 0; i < segments.length - 1; i++) {
    const left = segments[i];
    const right = segments[i + 1];
    const deltaSlope = right.slope - left.slope;
    const denom = right.slope - left.slope;
    if (Math.abs(denom) < 1e-10) continue;
    const kink = (left.intercept - right.intercept) / denom;
    terms.push({ kink, deltaSlope });
  }

  const modelBase = {
    inputRange: [config.min, config.max] as [number, number],
    samples,
    baseIntercept: segments[0].intercept,
    baseSlope: segments[0].slope,
    terms,
    plateaus: segments.map(segment => ({
      startX: points[segment.start].x,
      endX: points[segment.end - 1].x,
      slope: segment.slope,
      intercept: segment.intercept,
      intervals: segment.end - segment.start - 1
    }))
  };
  return { ...modelBase, expression: buildExpression(modelBase) };
}

export function predictNoisyBlackBoxReluModel(model: BlackBoxReluModel, x: number): number {
  return predictBlackBoxReluModel(model, x);
}
