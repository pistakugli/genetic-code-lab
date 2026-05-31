export type ActiveSubspaceFfnTrainingSample = {
  coords: number[];
  target: number;
};

export type ActiveSubspaceFfnDecompilerModel = {
  dimension: number;
  ranges: Array<[number, number]>;
  activeRank: number;
  gradientProbeCount: number;
  trainingSampleCount: number;
  finiteDifferenceStep: number;
  neighborCount: number;
  ridge: number;
  localModel: 'local-quadratic-knn';
  recoveryKind: 'active-subspace-ffn-surrogate';
  activeDirections: number[][];
  gradientEigenvalues: number[];
  totalGradientEnergy: number;
  selectedGradientEnergyRatio: number;
  meanGradientResidualRatio: number;
  samples: ActiveSubspaceFfnTrainingSample[];
  queryBudget: {
    rawOracleCalls: number;
    gradientRawOracleCalls: number;
    trainingRawOracleCalls: number;
    finiteDifferenceCallsPerProbe: number;
  };
  expression: string;
};

export type ActiveSubspaceFfnDecompilerConfig = {
  dimension: number;
  ranges: Array<[number, number]>;
  activeRank: number;
  gradientProbeCount?: number;
  trainingSampleCount?: number;
  finiteDifferenceStep?: number;
  neighborCount?: number;
  randomSeed?: number;
  ridge?: number;
};

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

function randomPoint(random: () => number, ranges: Array<[number, number]>, margin: number): number[] {
  return ranges.map(([min, max]) => {
    const boundedMargin = Math.min(margin, (max - min) * 0.2);
    return randomBetween(random, min + boundedMargin, max - boundedMargin);
  });
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += a[index] * b[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function subtractProjection(vector: number[], basis: number[][]): number[] {
  const output = [...vector];
  for (const direction of basis) {
    const coefficient = dot(output, direction);
    for (let index = 0; index < output.length; index++) output[index] -= coefficient * direction[index];
  }
  return output;
}

function canonicalizeDirection(direction: number[]): number[] {
  let maxIndex = 0;
  for (let index = 1; index < direction.length; index++) {
    if (Math.abs(direction[index]) > Math.abs(direction[maxIndex])) maxIndex = index;
  }
  return direction[maxIndex] < 0 ? direction.map(value => -value) : direction;
}

function covarianceVector(gradients: number[][], vector: number[]): number[] {
  const output = new Array(vector.length).fill(0);
  for (const gradient of gradients) {
    const coefficient = dot(gradient, vector);
    for (let index = 0; index < output.length; index++) output[index] += gradient[index] * coefficient;
  }
  return output.map(value => value / Math.max(1, gradients.length));
}

function estimateCoordinateGradient(
  oracle: (input: number[]) => number,
  input: number[],
  step: number
): number[] {
  const gradient: number[] = [];
  for (let axis = 0; axis < input.length; axis++) {
    const left = [...input];
    const right = [...input];
    left[axis] -= step;
    right[axis] += step;
    gradient.push((oracle(right) - oracle(left)) / (2 * step));
  }
  return gradient;
}

function recoverActiveDirections(
  gradients: number[][],
  rank: number,
  random: () => number
): { directions: number[][]; eigenvalues: number[]; totalEnergy: number } {
  const dimension = gradients[0]?.length ?? 0;
  const totalEnergy = gradients.reduce((sum, gradient) => sum + dot(gradient, gradient), 0) / Math.max(1, gradients.length);
  const directions: number[][] = [];
  const eigenvalues: number[] = [];
  for (let component = 0; component < rank; component++) {
    let vector = normalize(subtractProjection(
      Array.from({ length: dimension }, () => randomBetween(random, -1, 1)),
      directions
    ));
    for (let iteration = 0; iteration < 80; iteration++) {
      const multiplied = covarianceVector(gradients, vector);
      const orthogonal = subtractProjection(multiplied, directions);
      const length = norm(orthogonal);
      if (length < 1e-14) break;
      vector = orthogonal.map(value => value / length);
    }
    vector = canonicalizeDirection(normalize(subtractProjection(vector, directions)));
    const eigenvalue = Math.max(0, dot(vector, covarianceVector(gradients, vector)));
    directions.push(vector);
    eigenvalues.push(eigenvalue);
  }
  return { directions, eigenvalues, totalEnergy };
}

function projectInput(directions: number[][], input: number[]): number[] {
  return directions.map(direction => dot(direction, input));
}

function squaredDistance(a: number[], b: number[]): number {
  let distance = 0;
  for (let index = 0; index < a.length; index++) {
    const delta = a[index] - b[index];
    distance += delta * delta;
  }
  return distance;
}

function insertNeighbor(
  neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }>,
  candidate: { distance: number; sample: ActiveSubspaceFfnTrainingSample },
  maxCount: number
): void {
  let index = 0;
  while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
  if (index >= maxCount) return;
  neighbors.splice(index, 0, candidate);
  if (neighbors.length > maxCount) neighbors.pop();
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
    if (Math.abs(divisor) < 1e-12) throw new Error('active-subspace local quadratic system is singular');
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

function meanTarget(samples: ActiveSubspaceFfnTrainingSample[]): number {
  return samples.reduce((sum, sample) => sum + sample.target, 0) / Math.max(1, samples.length);
}

function buildExpression(model: Omit<ActiveSubspaceFfnDecompilerModel, 'expression'>): string {
  return [
    'active_subspace_ffn(',
    `dimension=${model.dimension}`,
    `rank=${model.activeRank}`,
    `gradient_probes=${model.gradientProbeCount}`,
    `samples=${model.trainingSampleCount}`,
    `energy=${model.selectedGradientEnergyRatio.toPrecision(6)}`,
    `queries=${model.queryBudget.rawOracleCalls}`,
    ')'
  ].join(' ');
}

export function decompileActiveSubspaceFfn(
  oracle: (input: number[]) => number,
  config: ActiveSubspaceFfnDecompilerConfig
): ActiveSubspaceFfnDecompilerModel {
  const dimension = Math.max(2, Math.floor(config.dimension));
  if (config.ranges.length !== dimension) throw new Error('active-subspace FFN decompiler requires one range per dimension');
  if (config.ranges.some(([min, max]) => !(max > min))) throw new Error('active-subspace FFN decompiler requires valid ranges');
  const activeRank = Math.max(1, Math.min(dimension, Math.floor(config.activeRank)));
  const gradientProbeCount = Math.max(activeRank + 2, Math.floor(config.gradientProbeCount ?? 16));
  const trainingSampleCount = Math.max(activeRank * 16, Math.floor(config.trainingSampleCount ?? 2400));
  const minRange = Math.min(...config.ranges.map(([min, max]) => max - min));
  const finiteDifferenceStep = Math.max(1e-6, Math.min(config.finiteDifferenceStep ?? minRange * 1e-4, minRange * 0.02));
  const neighborCount = Math.max(activeRank + 2, Math.floor(config.neighborCount ?? Math.max(32, activeRank * 24)));
  const ridge = config.ridge ?? 1e-8;
  const random = makeRandom(config.randomSeed ?? 260601);
  let rawOracleCalls = 0;
  const countedOracle = (input: number[]): number => {
    rawOracleCalls++;
    return oracle(input);
  };

  const gradients: number[][] = [];
  const gradientCallsStart = rawOracleCalls;
  for (let index = 0; index < gradientProbeCount; index++) {
    const point = randomPoint(random, config.ranges, finiteDifferenceStep * 2);
    gradients.push(estimateCoordinateGradient(countedOracle, point, finiteDifferenceStep));
  }
  const gradientRawOracleCalls = rawOracleCalls - gradientCallsStart;

  const { directions, eigenvalues, totalEnergy } = recoverActiveDirections(gradients, activeRank, random);
  const selectedEnergy = eigenvalues.reduce((sum, value) => sum + value, 0);
  const selectedGradientEnergyRatio = selectedEnergy / Math.max(1e-12, totalEnergy);
  let residualRatioSum = 0;
  for (const gradient of gradients) {
    const projection = directions.reduce((sum, direction) => sum + dot(gradient, direction) ** 2, 0);
    residualRatioSum += Math.sqrt(Math.max(0, dot(gradient, gradient) - projection)) / Math.max(1e-12, norm(gradient));
  }
  const meanGradientResidualRatio = residualRatioSum / Math.max(1, gradients.length);

  const trainingCallsStart = rawOracleCalls;
  const samples: ActiveSubspaceFfnTrainingSample[] = [];
  for (let index = 0; index < trainingSampleCount; index++) {
    const input = randomPoint(random, config.ranges, 0);
    samples.push({
      coords: projectInput(directions, input),
      target: countedOracle(input)
    });
  }
  const trainingRawOracleCalls = rawOracleCalls - trainingCallsStart;

  const modelBase = {
    dimension,
    ranges: config.ranges,
    activeRank,
    gradientProbeCount,
    trainingSampleCount,
    finiteDifferenceStep,
    neighborCount,
    ridge,
    localModel: 'local-quadratic-knn' as const,
    recoveryKind: 'active-subspace-ffn-surrogate' as const,
    activeDirections: directions,
    gradientEigenvalues: eigenvalues,
    totalGradientEnergy: totalEnergy,
    selectedGradientEnergyRatio,
    meanGradientResidualRatio,
    samples,
    queryBudget: {
      rawOracleCalls,
      gradientRawOracleCalls,
      trainingRawOracleCalls,
      finiteDifferenceCallsPerProbe: 2 * dimension
    }
  };
  return {
    ...modelBase,
    expression: buildExpression(modelBase)
  };
}

export function predictActiveSubspaceFfnModel(model: ActiveSubspaceFfnDecompilerModel, input: number[]): number {
  if (input.length !== model.dimension) throw new Error('active-subspace FFN prediction dimension mismatch');
  const coords = projectInput(model.activeDirections, input);
  const neighbors: Array<{ distance: number; sample: ActiveSubspaceFfnTrainingSample }> = [];
  for (const sample of model.samples) {
    insertNeighbor(neighbors, { distance: squaredDistance(coords, sample.coords), sample }, model.neighborCount);
  }
  if (neighbors.length < model.activeRank + 1) return meanTarget(model.samples);
  const feature = (values: number[]): number[] => {
    const row = [1, ...values];
    for (let i = 0; i < values.length; i++) {
      for (let j = i; j < values.length; j++) row.push(values[i] * values[j]);
    }
    return row;
  };
  const queryRow = feature(coords);
  const width = queryRow.length;
  const xtx = Array.from({ length: width }, () => new Array(width).fill(0));
  const xty = new Array(width).fill(0);
  for (const neighbor of neighbors) {
    const row = feature(neighbor.sample.coords);
    const weight = 1 / Math.max(1e-8, neighbor.distance);
    for (let i = 0; i < width; i++) {
      xty[i] += weight * row[i] * neighbor.sample.target;
      for (let j = 0; j < width; j++) xtx[i][j] += weight * row[i] * row[j];
    }
  }
  for (let i = 1; i < width; i++) xtx[i][i] += model.ridge;
  try {
    const coefficients = solveSystem(xtx, xty);
    const value = dot(coefficients, queryRow);
    return Number.isFinite(value) ? value : meanTarget(neighbors.map(neighbor => neighbor.sample));
  } catch {
    return meanTarget(neighbors.map(neighbor => neighbor.sample));
  }
}

export function predictActiveSubspaceFfnBatch(model: ActiveSubspaceFfnDecompilerModel, inputs: number[][]): number[] {
  return inputs.map(input => predictActiveSubspaceFfnModel(model, input));
}
