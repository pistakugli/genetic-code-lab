import {
  BlackBoxRelu3DModel,
  BlackBoxRelu3DTerm,
  predictBlackBoxRelu3DModel
} from './blackbox-relu3d-decompiler';
import { decompileNoisyBlackBoxRelu2D } from './noisy-blackbox-relu2d-decompiler';

export type NoisyBlackBoxRelu3DDecompilerConfig = {
  xRange: [number, number];
  yRange: [number, number];
  zRange: [number, number];
  maxTerms: number;
  zSlices?: number[];
  sliceCount?: number;
  ySliceCount?: number;
  samplesPerSlice?: number;
  querySamples?: number;
  aggregation?: 'median' | 'trimmed-mean' | 'mean';
  minSegmentPoints?: number;
  clusterSlopeTolerance?: number;
  kinkLineTolerance?: number;
  hyperplaneTolerance?: number;
  minSliceSupport?: number;
};

type SliceTerm = {
  z: number;
  alphaY: number;
  betaAtZ: number;
  deltaSlopeX: number;
};

type TermCluster = {
  terms: SliceTerm[];
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
  return Math.abs(term.betaAtZ - (line.intercept + line.slope * term.z));
}

function robustFitBetaLine(
  terms: SliceTerm[],
  tolerance: number,
  minSupport: number
): { line: { intercept: number; slope: number }; inliers: SliceTerm[] } | null {
  if (terms.length < Math.max(2, minSupport)) return null;
  let best: { line: { intercept: number; slope: number }; inliers: SliceTerm[]; sse: number } | null = null;
  for (let i = 0; i < terms.length; i++) {
    for (let j = i + 1; j < terms.length; j++) {
      if (Math.abs(terms[i].z - terms[j].z) < 1e-12) continue;
      const slope = (terms[j].betaAtZ - terms[i].betaAtZ) / (terms[j].z - terms[i].z);
      const intercept = terms[i].betaAtZ - slope * terms[i].z;
      const candidate = { intercept, slope };
      const inliers = terms.filter(term => lineError(candidate, term) <= tolerance);
      if (inliers.length < minSupport) continue;
      const refined = fitLine(inliers.map(term => term.z), inliers.map(term => term.betaAtZ));
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
  const fallback = fitLine(terms.map(term => term.z), terms.map(term => term.betaAtZ));
  const inliers = terms.filter(term => lineError(fallback, term) <= tolerance);
  return inliers.length >= minSupport ? { line: fallback, inliers } : null;
}

function clusterTerms(terms: SliceTerm[], tolerance: number): TermCluster[] {
  const clusters: TermCluster[] = [];
  for (const term of terms.sort((a, b) => a.deltaSlopeX - b.deltaSlopeX)) {
    const cluster = clusters.find(item => {
      const centerDelta = mean(item.terms.map(t => t.deltaSlopeX));
      const centerAlphaY = mean(item.terms.map(t => t.alphaY));
      return Math.abs(term.deltaSlopeX - centerDelta) <= tolerance
        && Math.abs(term.alphaY - centerAlphaY) <= tolerance;
    });
    if (cluster) cluster.terms.push(term);
    else clusters.push({ terms: [term] });
  }
  return clusters;
}

function buildExpression(model: Omit<BlackBoxRelu3DModel, 'expression'>): string {
  const parts = [
    formatNumber(model.baseIntercept),
    `${formatNumber(model.baseSlopeX)} * x`,
    `${formatNumber(model.baseSlopeY)} * y`,
    `${formatNumber(model.baseSlopeZ)} * z`
  ];
  for (const term of model.terms) {
    const yPart = term.alphaY >= 0
      ? `x - ${formatNumber(term.alphaY)} * y`
      : `x + ${formatNumber(-term.alphaY)} * y`;
    const zPart = term.alphaZ >= 0
      ? `${yPart} - ${formatNumber(term.alphaZ)} * z`
      : `${yPart} + ${formatNumber(-term.alphaZ)} * z`;
    const offset = term.beta >= 0
      ? `${zPart} - ${formatNumber(term.beta)}`
      : `${zPart} + ${formatNumber(-term.beta)}`;
    parts.push(`${formatNumber(term.deltaSlopeX)} * max(0, ${offset})`);
  }
  return parts.join(' + ');
}

export function decompileNoisyBlackBoxRelu3D(
  oracle: (x: number, y: number, z: number) => number,
  config: NoisyBlackBoxRelu3DDecompilerConfig
): BlackBoxRelu3DModel {
  const [xMin, xMax] = config.xRange;
  const [yMin, yMax] = config.yRange;
  const [zMin, zMax] = config.zRange;
  if (!(xMax > xMin) || !(yMax > yMin) || !(zMax > zMin)) throw new Error('noisy 3D ReLU decompiler requires valid x/y/z ranges');
  const zSlices = config.zSlices ?? makeSlices(config.zRange, Math.max(5, Math.floor(config.sliceCount ?? 7)));
  const samplesPerSlice = Math.max(80, Math.floor(config.samplesPerSlice ?? 321));
  const clusterSlopeTolerance = config.clusterSlopeTolerance ?? 0.10;
  const hyperplaneTolerance = config.hyperplaneTolerance ?? 0.12;
  const minSliceSupport = Math.max(2, Math.floor(config.minSliceSupport ?? Math.ceil(zSlices.length * 0.65)));

  const sliceTerms: SliceTerm[] = [];
  const baseIntercepts: Array<{ z: number; intercept: number }> = [];
  const baseSlopesX: number[] = [];
  const baseSlopesY: number[] = [];
  for (const z of zSlices) {
    const sliceModel = decompileNoisyBlackBoxRelu2D(
      (x, y) => oracle(x, y, z),
      {
        xRange: config.xRange,
        yRange: config.yRange,
        maxTerms: config.maxTerms,
        sliceCount: config.ySliceCount ?? 11,
        samplesPerSlice,
        querySamples: config.querySamples,
        aggregation: config.aggregation,
        minSegmentPoints: config.minSegmentPoints,
        clusterSlopeTolerance,
        kinkLineTolerance: config.kinkLineTolerance ?? 0.10
      }
    );
    baseIntercepts.push({ z, intercept: sliceModel.baseIntercept });
    baseSlopesX.push(sliceModel.baseSlopeX);
    baseSlopesY.push(sliceModel.baseSlopeY);
    for (const term of sliceModel.terms) {
      sliceTerms.push({
        z,
        alphaY: term.alphaY,
        betaAtZ: term.beta,
        deltaSlopeX: term.deltaSlopeX
      });
    }
  }

  const baseZ = fitLine(baseIntercepts.map(item => item.z), baseIntercepts.map(item => item.intercept));
  const terms: BlackBoxRelu3DTerm[] = clusterTerms(sliceTerms, clusterSlopeTolerance)
    .filter(cluster => cluster.terms.length >= minSliceSupport)
    .flatMap(cluster => {
      const fit = robustFitBetaLine(cluster.terms, hyperplaneTolerance, minSliceSupport);
      if (!fit) return [];
      return {
        alphaY: mean(fit.inliers.map(term => term.alphaY)),
        alphaZ: fit.line.slope,
        beta: fit.line.intercept,
        deltaSlopeX: mean(fit.inliers.map(term => term.deltaSlopeX)),
        support: fit.inliers.length
      };
    })
    .sort((a, b) => a.beta - b.beta);

  const modelBase = {
    xRange: config.xRange,
    yRange: config.yRange,
    zRange: config.zRange,
    zSlices,
    samplesPerSlice,
    baseIntercept: baseZ.intercept,
    baseSlopeX: mean(baseSlopesX),
    baseSlopeY: mean(baseSlopesY),
    baseSlopeZ: baseZ.slope,
    terms
  };
  return {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
}

export function predictNoisyBlackBoxRelu3DModel(model: BlackBoxRelu3DModel, x: number, y: number, z: number): number {
  return predictBlackBoxRelu3DModel(model, x, y, z);
}

export function predictNoisyBlackBoxRelu3DBatch(model: BlackBoxRelu3DModel, inputs: Array<[number, number, number]>): number[] {
  return inputs.map(([x, y, z]) => predictNoisyBlackBoxRelu3DModel(model, x, y, z));
}
