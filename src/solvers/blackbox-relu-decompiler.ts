export type BlackBoxReluTerm = {
  kink: number;
  deltaSlope: number;
};

export type BlackBoxReluModel = {
  inputRange: [number, number];
  samples: number;
  baseIntercept: number;
  baseSlope: number;
  terms: BlackBoxReluTerm[];
  expression: string;
  plateaus: Array<{
    startX: number;
    endX: number;
    slope: number;
    intercept: number;
    intervals: number;
  }>;
};

export type BlackBoxReluDecompilerConfig = {
  min: number;
  max: number;
  samples?: number;
  slopeTolerance?: number;
  minPlateauIntervals?: number;
  kinkMergeTolerance?: number;
};

type Point = { x: number; y: number };
type SlopeGroup = { start: number; end: number; sum: number; count: number };

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(8);
  return value.toPrecision(10).replace(/\.?0+$/, '');
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

function groupSlopes(slopes: number[], tolerance: number): SlopeGroup[] {
  const groups: SlopeGroup[] = [];
  let current: SlopeGroup | null = null;
  for (let i = 0; i < slopes.length; i++) {
    const slope = slopes[i];
    if (!current) {
      current = { start: i, end: i, sum: slope, count: 1 };
      continue;
    }
    const currentMean = current.sum / current.count;
    if (Math.abs(slope - currentMean) <= tolerance) {
      current.end = i;
      current.sum += slope;
      current.count++;
    } else {
      groups.push(current);
      current = { start: i, end: i, sum: slope, count: 1 };
    }
  }
  if (current) groups.push(current);
  return groups;
}

function mergeCloseTerms(terms: BlackBoxReluTerm[], tolerance: number): BlackBoxReluTerm[] {
  const sorted = terms
    .filter(term => Number.isFinite(term.kink) && Number.isFinite(term.deltaSlope))
    .sort((a, b) => a.kink - b.kink);
  const merged: BlackBoxReluTerm[] = [];
  for (const term of sorted) {
    const last = merged[merged.length - 1];
    if (last && Math.abs(last.kink - term.kink) <= tolerance) {
      const totalDelta = last.deltaSlope + term.deltaSlope;
      last.kink = Math.abs(totalDelta) > 1e-12
        ? ((last.kink * last.deltaSlope) + (term.kink * term.deltaSlope)) / totalDelta
        : mean([last.kink, term.kink]);
      last.deltaSlope = totalDelta;
    } else {
      merged.push({ ...term });
    }
  }
  return merged.filter(term => Math.abs(term.deltaSlope) > 1e-10);
}

export function predictBlackBoxReluModel(model: BlackBoxReluModel, x: number): number {
  let y = model.baseIntercept + model.baseSlope * x;
  for (const term of model.terms) {
    y += term.deltaSlope * Math.max(0, x - term.kink);
  }
  return Number.isFinite(y) ? y : 0;
}

export function predictBlackBoxReluBatch(model: BlackBoxReluModel, xs: number[]): number[] {
  return xs.map(x => predictBlackBoxReluModel(model, x));
}

export function decompileBlackBoxRelu1D(
  oracle: (x: number) => number,
  config: BlackBoxReluDecompilerConfig
): BlackBoxReluModel {
  if (!(config.max > config.min)) throw new Error('black-box ReLU decompiler requires max > min');
  const samples = Math.max(32, Math.floor(config.samples ?? 1201));
  const step = (config.max - config.min) / (samples - 1);
  const slopeTolerance = config.slopeTolerance ?? 1e-7;
  const minPlateauIntervals = Math.max(3, Math.floor(config.minPlateauIntervals ?? 6));
  const kinkMergeTolerance = config.kinkMergeTolerance ?? step * 2;
  const points: Point[] = Array.from({ length: samples }, (_, index) => {
    const x = config.min + step * index;
    return { x, y: oracle(x) };
  });
  const slopes = points.slice(0, -1).map((point, index) => {
    const next = points[index + 1];
    return (next.y - point.y) / (next.x - point.x);
  });
  const groups = groupSlopes(slopes, slopeTolerance);
  const plateaus = groups
    .filter(group => group.count >= minPlateauIntervals)
    .map(group => {
      const slope = group.sum / group.count;
      const plateauPoints = points.slice(group.start, group.end + 2);
      const intercept = mean(plateauPoints.map(point => point.y - slope * point.x));
      return {
        startX: points[group.start].x,
        endX: points[group.end + 1].x,
        slope,
        intercept,
        intervals: group.count
      };
    });

  if (plateaus.length === 0) {
    const slope = (points[points.length - 1].y - points[0].y) / (points[points.length - 1].x - points[0].x);
    const base = {
      inputRange: [config.min, config.max] as [number, number],
      samples,
      baseIntercept: mean(points.map(point => point.y - slope * point.x)),
      baseSlope: slope,
      terms: [],
      plateaus: []
    };
    return { ...base, expression: buildExpression(base) };
  }

  const terms: BlackBoxReluTerm[] = [];
  for (let i = 0; i < plateaus.length - 1; i++) {
    const left = plateaus[i];
    const right = plateaus[i + 1];
    const slopeDelta = right.slope - left.slope;
    if (Math.abs(slopeDelta) <= slopeTolerance) continue;
    const kink = (left.intercept - right.intercept) / (right.slope - left.slope);
    const gapLeft = left.endX - step;
    const gapRight = right.startX + step;
    if (kink >= gapLeft - step * 3 && kink <= gapRight + step * 3) {
      terms.push({ kink, deltaSlope: slopeDelta });
    }
  }

  const modelBase = {
    inputRange: [config.min, config.max] as [number, number],
    samples,
    baseIntercept: plateaus[0].intercept,
    baseSlope: plateaus[0].slope,
    terms: mergeCloseTerms(terms, kinkMergeTolerance),
    plateaus
  };
  return { ...modelBase, expression: buildExpression(modelBase) };
}
