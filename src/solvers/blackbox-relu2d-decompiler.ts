import { decompileBlackBoxRelu1D } from './blackbox-relu-decompiler';

export type BlackBoxRelu2DTerm = {
  alphaY: number;
  beta: number;
  deltaSlopeX: number;
  support: number;
};

export type BlackBoxRelu2DModel = {
  xRange: [number, number];
  yRange: [number, number];
  samplesPerSlice: number;
  baseIntercept: number;
  baseSlopeX: number;
  baseSlopeY: number;
  terms: BlackBoxRelu2DTerm[];
  expression: string;
};

export type BlackBoxRelu2DDecompilerConfig = {
  xRange: [number, number];
  yRange: [number, number];
  ySlices?: number[];
  samplesPerSlice?: number;
  slopeTolerance?: number;
  clusterSlopeTolerance?: number;
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

export function predictBlackBoxRelu2DModel(model: BlackBoxRelu2DModel, x: number, y: number): number {
  let value = model.baseIntercept + model.baseSlopeX * x + model.baseSlopeY * y;
  for (const term of model.terms) {
    value += term.deltaSlopeX * Math.max(0, x - term.alphaY * y - term.beta);
  }
  return Number.isFinite(value) ? value : 0;
}

export function predictBlackBoxRelu2DBatch(model: BlackBoxRelu2DModel, inputs: Array<[number, number]>): number[] {
  return inputs.map(([x, y]) => predictBlackBoxRelu2DModel(model, x, y));
}

export function decompileBlackBoxRelu2D(
  oracle: (x: number, y: number) => number,
  config: BlackBoxRelu2DDecompilerConfig
): BlackBoxRelu2DModel {
  const [xMin, xMax] = config.xRange;
  const [yMin, yMax] = config.yRange;
  if (!(xMax > xMin) || !(yMax > yMin)) throw new Error('2D ReLU decompiler requires valid x/y ranges');
  const ySlices = config.ySlices ?? makeSlices(config.yRange, 13);
  const samplesPerSlice = Math.max(64, Math.floor(config.samplesPerSlice ?? 1201));
  const slopeTolerance = config.slopeTolerance ?? 1e-8;
  const clusterSlopeTolerance = config.clusterSlopeTolerance ?? 1e-7;
  const minSliceSupport = Math.max(2, Math.floor(config.minSliceSupport ?? Math.ceil(ySlices.length * 0.75)));

  const sliceTerms: SliceTerm[] = [];
  const baseIntercepts: Array<{ y: number; intercept: number }> = [];
  const baseSlopesX: number[] = [];
  for (const y of ySlices) {
    const sliceModel = decompileBlackBoxRelu1D(
      x => oracle(x, y),
      {
        min: xMin,
        max: xMax,
        samples: samplesPerSlice,
        slopeTolerance,
        minPlateauIntervals: 6
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
  const terms = clusterTerms(sliceTerms, clusterSlopeTolerance)
    .filter(cluster => cluster.terms.length >= minSliceSupport)
    .map(cluster => {
      const line = fitLine(cluster.terms.map(term => term.y), cluster.terms.map(term => term.kinkX));
      return {
        alphaY: line.slope,
        beta: line.intercept,
        deltaSlopeX: cluster.sumDelta / cluster.terms.length,
        support: cluster.terms.length
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
