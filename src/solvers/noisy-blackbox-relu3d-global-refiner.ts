import {
  BlackBoxRelu3DModel,
  BlackBoxRelu3DTerm,
  predictBlackBoxRelu3DModel
} from './blackbox-relu3d-decompiler';

export type NoisyBlackBoxRelu3DGlobalRefinerConfig = {
  xRange: [number, number];
  yRange: [number, number];
  zRange: [number, number];
  termCount: number;
  initialModel?: BlackBoxRelu3DModel;
  sampleCounts?: [number, number, number];
  querySamples?: number;
  aggregation?: 'median' | 'trimmed-mean' | 'mean';
  restarts?: number;
  coordinatePasses?: number;
  randomSeed?: number;
  ridge?: number;
  alphaYRange?: [number, number];
  alphaZRange?: [number, number];
  betaRange?: [number, number];
  stepSchedule?: number[];
};

export type NoisyBlackBoxRelu3DGlobalRefinerResult = {
  model: BlackBoxRelu3DModel;
  trainingMse: number;
  sampleCount: number;
  restarts: number;
  objectiveEvaluations: number;
};

type Sample = {
  x: number;
  y: number;
  z: number;
  target: number;
};

type GeometryTerm = {
  alphaY: number;
  alphaZ: number;
  beta: number;
};

type Candidate = {
  terms: GeometryTerm[];
  coefficients: number[];
  mse: number;
};

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(8);
  return value.toPrecision(10).replace(/\.?0+$/, '');
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

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function randomBetween(random: () => number, min: number, max: number): number {
  return min + (max - min) * random();
}

function clamp(value: number, [min, max]: [number, number]): number {
  return Math.max(min, Math.min(max, value));
}

function linspace([min, max]: [number, number], count: number): number[] {
  if (count <= 1) return [(min + max) / 2];
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function aggregate(values: number[], mode: 'median' | 'trimmed-mean' | 'mean'): number {
  if (values.length === 0) return 0;
  if (mode === 'mean') return mean(values);
  const sorted = [...values].sort((a, b) => a - b);
  if (mode === 'median') {
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }
  const trim = Math.floor(sorted.length * 0.2);
  return mean(sorted.slice(trim, sorted.length - trim || sorted.length));
}

function collectSamples(
  oracle: (x: number, y: number, z: number) => number,
  config: NoisyBlackBoxRelu3DGlobalRefinerConfig
): Sample[] {
  const [xCount, yCount, zCount] = config.sampleCounts ?? [9, 7, 7];
  const xs = linspace(config.xRange, Math.max(3, Math.floor(xCount)));
  const ys = linspace(config.yRange, Math.max(3, Math.floor(yCount)));
  const zs = linspace(config.zRange, Math.max(3, Math.floor(zCount)));
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 15));
  const aggregation = config.aggregation ?? 'median';
  const samples: Sample[] = [];
  for (const x of xs) {
    for (const y of ys) {
      for (const z of zs) {
        const values = Array.from({ length: querySamples }, () => oracle(x, y, z));
        samples.push({ x, y, z, target: aggregate(values, aggregation) });
      }
    }
  }
  return samples;
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
    if (Math.abs(divisor) < 1e-14) throw new Error('global ReLU refiner linear system is singular');
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

function fitCoefficients(samples: Sample[], terms: GeometryTerm[], ridge: number): { coefficients: number[]; mse: number } {
  const width = 4 + terms.length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const sample of samples) {
    const row = [
      1,
      sample.x,
      sample.y,
      sample.z,
      ...terms.map(term => Math.max(0, sample.x - term.alphaY * sample.y - term.alphaZ * sample.z - term.beta))
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
    let predicted = coefficients[0] + coefficients[1] * sample.x + coefficients[2] * sample.y + coefficients[3] * sample.z;
    for (let i = 0; i < terms.length; i++) {
      predicted += coefficients[4 + i] * Math.max(0, sample.x - terms[i].alphaY * sample.y - terms[i].alphaZ * sample.z - terms[i].beta);
    }
    const error = predicted - sample.target;
    sse += error * error;
  }
  return { coefficients, mse: sse / Math.max(1, samples.length) };
}

function candidateFromTerms(samples: Sample[], terms: GeometryTerm[], ridge: number): Candidate {
  const fit = fitCoefficients(samples, terms, ridge);
  return {
    terms: terms.map(term => ({ ...term })),
    coefficients: fit.coefficients,
    mse: fit.mse
  };
}

function makeInitialStarts(
  config: NoisyBlackBoxRelu3DGlobalRefinerConfig,
  random: () => number,
  restarts: number
): GeometryTerm[][] {
  const termCount = Math.max(1, Math.floor(config.termCount));
  const alphaYRange = config.alphaYRange ?? [-0.65, 0.65];
  const alphaZRange = config.alphaZRange ?? [-0.60, 0.60];
  const betaRange = config.betaRange ?? [config.xRange[0] * 0.75, config.xRange[1] * 0.75];
  const starts: GeometryTerm[][] = [];
  const seedTerms = (config.initialModel?.terms ?? []).slice(0, termCount).map(term => ({
    alphaY: clamp(term.alphaY, alphaYRange),
    alphaZ: clamp(term.alphaZ, alphaZRange),
    beta: clamp(term.beta, betaRange)
  }));
  while (seedTerms.length < termCount) {
    seedTerms.push({
      alphaY: randomBetween(random, alphaYRange[0], alphaYRange[1]),
      alphaZ: randomBetween(random, alphaZRange[0], alphaZRange[1]),
      beta: randomBetween(random, betaRange[0], betaRange[1])
    });
  }
  starts.push(seedTerms);
  for (let i = 0; i < Math.max(0, restarts - 1); i++) {
    const terms = seedTerms.map(term => ({
      alphaY: clamp(term.alphaY + randomBetween(random, -0.45, 0.45), alphaYRange),
      alphaZ: clamp(term.alphaZ + randomBetween(random, -0.40, 0.40), alphaZRange),
      beta: clamp(term.beta + randomBetween(random, -0.85, 0.85), betaRange)
    }));
    if (i % 3 !== 0 || !config.initialModel) {
      for (let j = 0; j < termCount; j++) {
        if (random() < 0.7) {
          terms[j] = {
            alphaY: randomBetween(random, alphaYRange[0], alphaYRange[1]),
            alphaZ: randomBetween(random, alphaZRange[0], alphaZRange[1]),
            beta: randomBetween(random, betaRange[0], betaRange[1])
          };
        }
      }
    }
    starts.push(terms);
  }
  return starts;
}

function refineCandidate(
  samples: Sample[],
  start: GeometryTerm[],
  config: NoisyBlackBoxRelu3DGlobalRefinerConfig
): { candidate: Candidate; evaluations: number } {
  const alphaYRange = config.alphaYRange ?? [-0.65, 0.65];
  const alphaZRange = config.alphaZRange ?? [-0.60, 0.60];
  const betaRange = config.betaRange ?? [config.xRange[0] * 0.75, config.xRange[1] * 0.75];
  const ridge = config.ridge ?? 1e-8;
  const stepSchedule = config.stepSchedule ?? [0.60, 0.30, 0.15, 0.075, 0.035, 0.016];
  const coordinatePasses = Math.max(1, Math.floor(config.coordinatePasses ?? 2));
  let best = candidateFromTerms(samples, start, ridge);
  let evaluations = 1;

  for (const step of stepSchedule) {
    for (let pass = 0; pass < coordinatePasses; pass++) {
      let improved = false;
      for (let termIndex = 0; termIndex < best.terms.length; termIndex++) {
        const fields: Array<keyof GeometryTerm> = ['alphaY', 'alphaZ', 'beta'];
        for (const field of fields) {
          const range = field === 'alphaY' ? alphaYRange : field === 'alphaZ' ? alphaZRange : betaRange;
          for (const direction of [-1, 1]) {
            const terms = best.terms.map(term => ({ ...term }));
            terms[termIndex][field] = clamp(terms[termIndex][field] + direction * step, range);
            const candidate = candidateFromTerms(samples, terms, ridge);
            evaluations += 1;
            if (candidate.mse < best.mse) {
              best = candidate;
              improved = true;
            }
          }
        }
      }
      if (!improved) break;
    }
  }
  return { candidate: best, evaluations };
}

export function refineNoisyBlackBoxRelu3DGlobal(
  oracle: (x: number, y: number, z: number) => number,
  config: NoisyBlackBoxRelu3DGlobalRefinerConfig
): NoisyBlackBoxRelu3DGlobalRefinerResult {
  if (!(config.xRange[1] > config.xRange[0]) || !(config.yRange[1] > config.yRange[0]) || !(config.zRange[1] > config.zRange[0])) {
    throw new Error('global noisy 3D ReLU refiner requires valid x/y/z ranges');
  }
  if (config.termCount < 1) throw new Error('global noisy 3D ReLU refiner requires at least one term');

  const samples = collectSamples(oracle, config);
  const restarts = Math.max(1, Math.floor(config.restarts ?? 96));
  const random = makeRandom(config.randomSeed ?? 260527);
  const starts = makeInitialStarts(config, random, restarts);
  let best: Candidate | null = null;
  let objectiveEvaluations = 0;
  for (const start of starts) {
    const refined = refineCandidate(samples, start, config);
    objectiveEvaluations += refined.evaluations;
    if (!best || refined.candidate.mse < best.mse) best = refined.candidate;
  }
  if (!best) throw new Error('global noisy 3D ReLU refiner produced no candidate');

  const terms: BlackBoxRelu3DTerm[] = best.terms.map((term, index) => ({
    alphaY: term.alphaY,
    alphaZ: term.alphaZ,
    beta: term.beta,
    deltaSlopeX: best.coefficients[4 + index],
    support: samples.length
  })).sort((a, b) => a.beta - b.beta);
  const modelBase = {
    xRange: config.xRange,
    yRange: config.yRange,
    zRange: config.zRange,
    zSlices: config.initialModel?.zSlices ?? linspace(config.zRange, config.sampleCounts?.[2] ?? 7),
    samplesPerSlice: samples.length,
    baseIntercept: best.coefficients[0],
    baseSlopeX: best.coefficients[1],
    baseSlopeY: best.coefficients[2],
    baseSlopeZ: best.coefficients[3],
    terms
  };
  const model = {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
  return {
    model,
    trainingMse: best.mse,
    sampleCount: samples.length,
    restarts,
    objectiveEvaluations
  };
}

export function predictNoisyBlackBoxRelu3DGlobalModel(model: BlackBoxRelu3DModel, x: number, y: number, z: number): number {
  return predictBlackBoxRelu3DModel(model, x, y, z);
}
