import {
  BlackBoxRelu2DModel,
  BlackBoxRelu2DTerm,
  predictBlackBoxRelu2DModel
} from './blackbox-relu2d-decompiler';
import { decompileNoisyBlackBoxRelu1D } from './noisy-blackbox-relu-decompiler';

export type NoisyBlackBoxRelu2DDecompilerConfig = {
  xRange: [number, number];
  yRange: [number, number];
  maxTerms: number;
  ySlices?: number[];
  sliceCount?: number;
  samplesPerSlice?: number;
  querySamples?: number;
  aggregation?: 'median' | 'trimmed-mean' | 'mean';
  minSegmentPoints?: number;
  clusterSlopeTolerance?: number;
  kinkLineTolerance?: number;
  minSliceSupport?: number;
};

type SliceTerm = {
  y: number;
  kinkX: number;
  deltaSlopeX: number;
};

type TermCluster = {
  terms: SliceTerm[];
  sumDelta: number;
};

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(8);
  return value.toPrecision(10).replace(/\.?0+$/, '');
}

function makeSlices([min, max]: [number, number], count: number): number[] {
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function fitLine(xs: number[], ys: number[]): { intercept: number; slope: number } {
  const xMean = mean(xs);
  const yMean = mean(ys);
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < xs.length; i++) {
    numerator += (xs[i] - xMean) * (ys[i] - yMean);
    denominator += (xs[i] - xMean) ** 2;
  }
  const slope = denominator > 0 ? numerator / denominator : 0;
  return { intercept: yMean - slope * xMean, slope };
}

function lineError(line: { intercept: number; slope: number }, term: SliceTerm): number {
  return Math.abs(term.kinkX - (line.intercept + line.slope * term.y));
}

function robustFitKinkLine(
  terms: SliceTerm[],
  tolerance: number,
  minSupport: number
): { line: { intercept: number; slope: number }; inliers: SliceTerm[] } | null {
  if (terms.length < Math.max(2, minSupport)) return null;
  let best: { line: { intercept: number; slope: number }; inliers: SliceTerm[]; sse: number } | null = null;
  for (let i = 0; i < terms.length; i++) {
    for (let j = i + 1; j < terms.length; j++) {
      if (Math.abs(terms[i].y - terms[j].y) < 1e-12) continue;
      const slope = (terms[j].kinkX - terms[i].kinkX) / (terms[j].y - terms[i].y);
      const intercept = terms[i].kinkX - slope * terms[i].y;
      const line = { intercept, slope };
      const inliers = terms.filter(term => lineError(line, term) <= tolerance);
      if (inliers.length < minSupport) continue;
      const refined = fitLine(inliers.map(term => term.y), inliers.map(term => term.kinkX));
      const refinedInliers = terms.filter(term => lineError(refined, term) <= tolerance);
      if (refinedInliers.length < minSupport) continue;
      const sse = refinedInliers.reduce((sum, term) => {
        const error = lineError(refined, term);
        return sum + error * error;
      }, 0);
      if (
        !best
        || refinedInliers.length > best.inliers.length
        || (refinedInliers.length === best.inliers.length && sse < best.sse)
      ) {
        best = { line: refined, inliers: refinedInliers, sse };
      }
    }
  }
  if (best) return { line: best.line, inliers: best.inliers };
  const fallback = fitLine(terms.map(term => term.y), terms.map(term => term.kinkX));
  const inliers = terms.filter(term => lineError(fallback, term) <= tolerance);
  return inliers.length >= minSupport ? { line: fallback, inliers } : null;
}

function clusterTerms(terms: SliceTerm[], tolerance: number): TermCluster[] {
  const clusters: TermCluster[] = [];
  for (const term of terms.sort((a, b) => a.deltaSlopeX - b.deltaSlopeX)) {
    const cluster = clusters.find(item => Math.abs(term.deltaSlopeX - item.sumDelta / item.terms.length) <= tolerance);
    if (cluster) {
      cluster.terms.push(term);
      cluster.sumDelta += term.deltaSlopeX;
    } else {
      clusters.push({ terms: [term], sumDelta: term.deltaSlopeX });
    }
  }
  return clusters;
}

function buildExpression(model: Omit<BlackBoxRelu2DModel, 'expression'>): string {
  const parts = [
    formatNumber(model.baseIntercept),
    `${formatNumber(model.baseSlopeX)} * x`,
    `${formatNumber(model.baseSlopeY)} * y`
  ];
  for (const term of model.terms) {
    const yPart = term.alphaY >= 0
      ? `x - ${formatNumber(term.alphaY)} * y`
      : `x + ${formatNumber(-term.alphaY)} * y`;
    const offset = term.beta >= 0
      ? `${yPart} - ${formatNumber(term.beta)}`
      : `${yPart} + ${formatNumber(-term.beta)}`;
    parts.push(`${formatNumber(term.deltaSlopeX)} * max(0, ${offset})`);
  }
  return parts.join(' + ');
}

export function decompileNoisyBlackBoxRelu2D(
  oracle: (x: number, y: number) => number,
  config: NoisyBlackBoxRelu2DDecompilerConfig
): BlackBoxRelu2DModel {
  const [xMin, xMax] = config.xRange;
  const [yMin, yMax] = config.yRange;
  if (!(xMax > xMin) || !(yMax > yMin)) throw new Error('noisy 2D ReLU decompiler requires valid x/y ranges');
  const ySlices = config.ySlices ?? makeSlices(config.yRange, Math.max(5, Math.floor(config.sliceCount ?? 13)));
  const samplesPerSlice = Math.max(80, Math.floor(config.samplesPerSlice ?? 421));
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 21));
  const minSegmentPoints = Math.max(6, Math.floor(config.minSegmentPoints ?? 20));
  const clusterSlopeTolerance = config.clusterSlopeTolerance ?? 0.08;
  const kinkLineTolerance = config.kinkLineTolerance ?? 0.09;
  const minSliceSupport = Math.max(2, Math.floor(config.minSliceSupport ?? Math.ceil(ySlices.length * 0.65)));

  const sliceTerms: SliceTerm[] = [];
  const baseIntercepts: Array<{ y: number; intercept: number }> = [];
  const baseSlopesX: number[] = [];
  for (const y of ySlices) {
    const sliceModel = decompileNoisyBlackBoxRelu1D(
      x => oracle(x, y),
      {
        min: xMin,
        max: xMax,
        maxTerms: config.maxTerms,
        samples: samplesPerSlice,
        querySamples,
        aggregation: config.aggregation ?? 'median',
        minSegmentPoints
      }
    );
    baseIntercepts.push({ y, intercept: sliceModel.baseIntercept });
    baseSlopesX.push(sliceModel.baseSlope);
    for (const term of sliceModel.terms) {
      if (term.kink >= xMin && term.kink <= xMax) {
        sliceTerms.push({ y, kinkX: term.kink, deltaSlopeX: term.deltaSlope });
      }
    }
  }

  const baseLine = fitLine(baseIntercepts.map(item => item.y), baseIntercepts.map(item => item.intercept));
  const terms: BlackBoxRelu2DTerm[] = clusterTerms(sliceTerms, clusterSlopeTolerance)
    .filter(cluster => cluster.terms.length >= minSliceSupport)
    .flatMap(cluster => {
      const fit = robustFitKinkLine(cluster.terms, kinkLineTolerance, minSliceSupport);
      if (!fit) return [];
      return {
        alphaY: fit.line.slope,
        beta: fit.line.intercept,
        deltaSlopeX: mean(fit.inliers.map(term => term.deltaSlopeX)),
        support: fit.inliers.length
      };
    })
    .sort((a, b) => a.beta - b.beta);

  const modelBase = {
    xRange: config.xRange,
    yRange: config.yRange,
    samplesPerSlice,
    baseIntercept: baseLine.intercept,
    baseSlopeX: mean(baseSlopesX),
    baseSlopeY: baseLine.slope,
    terms
  };
  return {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
}

export function predictNoisyBlackBoxRelu2DModel(model: BlackBoxRelu2DModel, x: number, y: number): number {
  return predictBlackBoxRelu2DModel(model, x, y);
}

export function predictNoisyBlackBoxRelu2DBatch(model: BlackBoxRelu2DModel, inputs: Array<[number, number]>): number[] {
  return inputs.map(([x, y]) => predictNoisyBlackBoxRelu2DModel(model, x, y));
}
