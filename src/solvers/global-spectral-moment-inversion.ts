export type GlobalSpectralMomentTerm = {
  normal: number[];
  offset: number;
  readout: number;
};

export type GlobalSpectralMomentDiagnostics = {
  totalOracleCalls: number;
  sampleCount: number;
  sampleSigma: number;
  maximumAbsoluteInput: number;
  computeDevice: string;
  deviceName: string;
  cudaMemoryAllocatedBytes: number;
  momentAndInversionTimeMs: number;
  estimatedRank: number;
  rankThreshold: number;
  absoluteMomentEigenvalues: number[];
  unresolvedComponents: number;
  matrixPencilResidual: number | null;
  momentOrder: 'second-and-third' | 'second-only';
};

export type GlobalSpectralMomentModel = {
  dimension: number;
  baseIntercept: number;
  baseWeights: number[];
  terms: GlobalSpectralMomentTerm[];
  diagnostics: GlobalSpectralMomentDiagnostics;
  reconstructionComplete: boolean;
};

export type GlobalSpectralMomentConfig = {
  dimension: number;
  oracleBudget: number;
  randomSeed?: number;
  sampleCount?: number;
  sampleSigma?: number;
  maximumAbsoluteInput?: number;
  momentOrder?: 'second-and-third' | 'second-only';
};

export type GlobalSpectralMomentSamples = {
  inputs: number[][];
  outputs: number[];
  diagnostics: Pick<GlobalSpectralMomentDiagnostics,
    'totalOracleCalls' | 'sampleCount' | 'sampleSigma' | 'maximumAbsoluteInput'>;
};

export type GlobalSpectralMomentBackendResult = {
  baseIntercept: number;
  baseWeights: number[];
  terms: GlobalSpectralMomentTerm[];
  reconstructionComplete: boolean;
  diagnostics: Pick<GlobalSpectralMomentDiagnostics,
    'computeDevice' | 'deviceName' | 'cudaMemoryAllocatedBytes' | 'momentAndInversionTimeMs'
    | 'estimatedRank' | 'rankThreshold' | 'absoluteMomentEigenvalues'
    | 'unresolvedComponents' | 'matrixPencilResidual' | 'momentOrder'>;
};

export type GlobalSpectralMomentBackend = {
  invert: (
    samples: GlobalSpectralMomentSamples,
    config: Required<Pick<GlobalSpectralMomentConfig,
      'dimension' | 'sampleSigma' | 'maximumAbsoluteInput' | 'momentOrder'>>
      & Pick<GlobalSpectralMomentConfig, 'randomSeed'>
  ) => Promise<GlobalSpectralMomentBackendResult>;
};

function makeRandom(seed: number): () => number {
  let state = Math.floor(seed) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = state * 16807 % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function gaussian(random: () => number): number {
  const first = Math.max(random(), Number.MIN_VALUE);
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random());
}

function validateConfig(config: GlobalSpectralMomentConfig): void {
  if (!Number.isSafeInteger(config.dimension) || config.dimension < 1) {
    throw new Error('Global spectral moment inversion dimension must be positive');
  }
  if (!Number.isSafeInteger(config.oracleBudget) || config.oracleBudget < 0) {
    throw new Error('Global spectral moment inversion oracleBudget must be non-negative');
  }
  for (const [name, value, minimum] of [
    ['sampleCount', config.sampleCount, 1],
    ['randomSeed', config.randomSeed, 0]
  ] as const) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < minimum)) {
      throw new Error(`Global spectral moment inversion ${name} is invalid`);
    }
  }
  if (config.sampleSigma !== undefined
    && (!Number.isFinite(config.sampleSigma) || config.sampleSigma <= 0)) {
    throw new Error('Global spectral moment inversion sampleSigma must be finite and positive');
  }
  if (config.maximumAbsoluteInput !== undefined
    && (!Number.isFinite(config.maximumAbsoluteInput) || config.maximumAbsoluteInput <= 0)) {
    throw new Error('Global spectral moment inversion maximumAbsoluteInput must be finite and positive');
  }
  if ((config.maximumAbsoluteInput ?? 1) / (config.sampleSigma ?? 0.2) < 4) {
    throw new Error('Global spectral moment inversion requires maximumAbsoluteInput/sampleSigma >= 4');
  }
  if (config.momentOrder !== undefined
    && config.momentOrder !== 'second-and-third' && config.momentOrder !== 'second-only') {
    throw new Error('Global spectral moment inversion momentOrder is unsupported');
  }
}

export function createGlobalSpectralMomentSamples(
  oracle: (input: number[]) => number,
  config: GlobalSpectralMomentConfig
): GlobalSpectralMomentSamples {
  validateConfig(config);
  const sampleSigma = config.sampleSigma ?? 0.2;
  const maximumAbsoluteInput = config.maximumAbsoluteInput ?? 1;
  const sampleCount = Math.min(config.sampleCount ?? 4096, config.oracleBudget);
  const random = makeRandom(config.randomSeed ?? 1);
  const inputs: number[][] = [];
  const outputs: number[] = [];

  while (inputs.length < sampleCount) {
    const point = Array.from({ length: config.dimension }, () =>
      sampleSigma * gaussian(random));
    if (point.some(value => Math.abs(value) > maximumAbsoluteInput)) continue;
    const output = oracle(point);
    if (!Number.isFinite(output)) {
      throw new Error('Global spectral moment inversion oracle returned a non-finite value');
    }
    inputs.push(point);
    outputs.push(output);
  }

  return {
    inputs,
    outputs,
    diagnostics: {
      totalOracleCalls: outputs.length,
      sampleCount: outputs.length,
      sampleSigma,
      maximumAbsoluteInput
    }
  };
}

export async function decompileBlackBoxReluNdGlobalSpectralMomentInversion(
  oracle: (input: number[]) => number,
  config: GlobalSpectralMomentConfig,
  backend: GlobalSpectralMomentBackend
): Promise<GlobalSpectralMomentModel> {
  validateConfig(config);
  const samples = createGlobalSpectralMomentSamples(oracle, config);
  const backendConfig = {
    dimension: config.dimension,
    sampleSigma: samples.diagnostics.sampleSigma,
    maximumAbsoluteInput: samples.diagnostics.maximumAbsoluteInput,
    momentOrder: config.momentOrder ?? 'second-and-third' as const,
    randomSeed: config.randomSeed
  };
  if (samples.outputs.length < 2) {
    return {
      dimension: config.dimension,
      baseIntercept: samples.outputs[0] ?? 0,
      baseWeights: Array.from({ length: config.dimension }, () => 0),
      terms: [],
      reconstructionComplete: false,
      diagnostics: {
        ...samples.diagnostics,
        computeDevice: 'none',
        deviceName: 'none',
        cudaMemoryAllocatedBytes: 0,
        momentAndInversionTimeMs: 0,
        estimatedRank: 0,
        rankThreshold: 0,
        absoluteMomentEigenvalues: [],
        unresolvedComponents: 0,
        matrixPencilResidual: null,
        momentOrder: backendConfig.momentOrder
      }
    };
  }
  const reconstructed = await backend.invert(samples, backendConfig);
  return {
    dimension: config.dimension,
    ...reconstructed,
    diagnostics: {
      ...samples.diagnostics,
      ...reconstructed.diagnostics
    }
  };
}

export function predictBlackBoxReluNdGlobalSpectralMomentInversion(
  model: GlobalSpectralMomentModel,
  input: number[]
): number {
  if (input.length !== model.dimension) {
    throw new Error('Global spectral moment inversion prediction dimension mismatch');
  }
  let output = model.baseIntercept;
  for (let axis = 0; axis < model.dimension; axis++) {
    output += model.baseWeights[axis] * input[axis];
  }
  for (const term of model.terms) {
    let preactivation = term.offset;
    for (let axis = 0; axis < model.dimension; axis++) {
      preactivation += term.normal[axis] * input[axis];
    }
    output += term.readout * Math.max(0, preactivation);
  }
  return output;
}
