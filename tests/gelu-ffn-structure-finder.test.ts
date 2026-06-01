import {
  canonicalizeGeluFfnStructure,
  compareGeluFfnCanonicalSignatures,
  decompileGeluFfnStructure,
  decompileRankAdaptiveGeluFfnStructure,
  GeluFfnStructureAtom,
  predictGeluFfnStructureBatch
} from '../src/solvers/gelu-ffn-structure-finder';
import {
  GeluFfnSourceNeuron,
  verifyGeluFfnNeuronProvenance
} from '../src/solvers/gelu-ffn-provenance-verifier';
import { regressionMetrics } from '../src/solvers/symbolic-residual-ensemble';

describe('GELU FFN structure finder', () => {
  function makeRandom(seed: number): () => number {
    let state = seed % 2147483647;
    if (state <= 0) state += 2147483646;
    return () => {
      state = (state * 16807) % 2147483647;
      return (state - 1) / 2147483646;
    };
  }

  function gaussian(random: () => number): number {
    const u1 = Math.max(1e-12, random());
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function dot(a: number[], b: number[]): number {
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
  }

  function normalize(values: number[]): number[] {
    const length = Math.sqrt(Math.max(1e-12, dot(values, values)));
    return values.map(value => value / length);
  }

  function gelu(value: number): number {
    return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
  }

  function makeDirection(seed: number, dimension: number): number[] {
    const random = makeRandom(seed);
    return normalize(Array.from({ length: dimension }, () => gaussian(random)));
  }

  function makeInputs(seed: number, dimension: number, count: number): number[][] {
    const random = makeRandom(seed);
    return Array.from({ length: count }, () => (
      Array.from({ length: dimension }, () => -1.5 + 3 * random())
    ));
  }

  function makeAtom(direction: number[], inputWeightNorm: number, inputBias: number, outputWeight: number): GeluFfnStructureAtom {
    return {
      inputWeight: direction.map(value => value * inputWeightNorm),
      inputWeightNorm,
      inputBias,
      outputWeight
    };
  }

  function makeSourceNeuron(
    layer: string,
    index: number,
    direction: number[],
    inputWeightNorm: number,
    inputBias: number,
    outputWeight: number
  ): GeluFfnSourceNeuron {
    return {
      id: `${layer}[${index}]`,
      layer,
      index,
      origin: `unit test source neuron ${index}`,
      inputWeight: direction.map(value => value * inputWeightNorm),
      inputBias,
      outputWeight,
      activation: 'gelu'
    };
  }

  it('canonical GELU signatures survive permutation, duplicate splits, and cancellation pairs', () => {
    const directions = [
      normalize([1, 0.2, -0.1]),
      normalize([-0.3, 0.9, 0.1]),
      normalize([0.2, -0.1, 0.97]),
      normalize([-0.7, -0.4, 0.2])
    ];
    const minimalAtoms = [
      makeAtom(directions[0], 1.2, -0.25, 0.9),
      makeAtom(directions[1], 0.95, 0.12, -0.65),
      makeAtom(directions[2], 0.8, 0.4, 0.5)
    ];
    const reparameterizedAtoms = [
      makeAtom(directions[2], 0.8, 0.4, 0.5),
      makeAtom(directions[0], 1.2, -0.25, 0.35),
      makeAtom(directions[3], 1.05, -0.1, 0.7),
      makeAtom(directions[1], 0.95, 0.12, -0.65),
      makeAtom(directions[0], 1.2, -0.25, 0.55),
      makeAtom(directions[3], 1.05, -0.1, -0.7)
    ];

    const left = canonicalizeGeluFfnStructure({ atoms: minimalAtoms });
    const right = canonicalizeGeluFfnStructure({ atoms: reparameterizedAtoms });
    const comparison = compareGeluFfnCanonicalSignatures(left, right);

    expect(left.canonicalAtomCount).toBe(3);
    expect(right.canonicalAtomCount).toBe(3);
    expect(right.mergedDuplicateCount).toBeGreaterThanOrEqual(2);
    expect(right.removedCancellationCount).toBe(2);
    expect(comparison.atomCountDelta).toBe(0);
    expect(comparison.signatureDistance).toBeLessThan(1e-10);
    expect(comparison.maxAtomDistance).toBeLessThan(1e-10);
  });

  it('canonical GELU signatures reject genuinely different atom structure', () => {
    const directions = [
      normalize([1, 0.2, -0.1]),
      normalize([-0.3, 0.9, 0.1]),
      normalize([0.2, -0.1, 0.97])
    ];
    const left = canonicalizeGeluFfnStructure({
      atoms: [
        makeAtom(directions[0], 1.2, -0.25, 0.9),
        makeAtom(directions[1], 0.95, 0.12, -0.65),
        makeAtom(directions[2], 0.8, 0.4, 0.5)
      ]
    });
    const right = canonicalizeGeluFfnStructure({
      atoms: [
        makeAtom(directions[0], 1.2, -0.25, 0.9),
        makeAtom(directions[1], 0.95, 0.42, -0.65),
        makeAtom(directions[2], 0.8, 0.4, 0.5)
      ]
    });
    const comparison = compareGeluFfnCanonicalSignatures(left, right);

    expect(comparison.atomCountDelta).toBe(0);
    expect(comparison.signatureDistance).toBeGreaterThan(0.09);
  });

  it('recovers a single high-dimensional GELU neuron as explicit structure', () => {
    const dimension = 96;
    const direction = makeDirection(2711, dimension);
    const oracle = (input: number[]): number => 0.17 + 1.35 * gelu(1.45 * dot(direction, input) - 0.22);
    const model = decompileGeluFfnStructure(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
      channelCount: 1,
      gradientProbeCount: 10,
      trainingSampleCount: 1600,
      validationSampleCount: 700,
      randomSeed: 3711,
      restarts: 8,
      iterations: 1000,
      learningRate: 0.035
    });
    const inputs = makeInputs(4711, dimension, 900);
    const metrics = regressionMetrics(inputs.map(oracle), predictGeluFfnStructureBatch(model, inputs));
    const atom = model.atoms[0];
    const recoveredDirection = normalize(atom.inputWeight);

    expect(model.recoveryKind).toBe('gelu-ffn-structure');
    expect(model.atoms).toHaveLength(1);
    expect(Math.abs(dot(direction, recoveredDirection))).toBeGreaterThan(0.995);
    expect(Math.abs(atom.inputWeightNorm - 1.45)).toBeLessThan(0.12);
    expect(Math.abs(atom.inputBias + 0.22)).toBeLessThan(0.12);
    expect(Math.abs(Math.abs(atom.outputWeight) - 1.35)).toBeLessThan(0.18);
    expect(metrics.mae).toBeLessThan(0.015);
    expect(metrics.r2).toBeGreaterThan(0.999);
  });

  it('discovers effective mixed-channel GELU atom count without a true channel hint', () => {
    const dimension = 16;
    const directions: number[][] = [];
    for (const seed of [7101, 7202, 7303]) directions.push(makeDirection(seed, dimension));
    const orthogonalDirections: number[][] = [];
    for (const direction of directions) {
      let residual = [...direction];
      for (const previous of orthogonalDirections) {
        const projection = dot(residual, previous);
        residual = residual.map((value, index) => value - projection * previous[index]);
      }
      orthogonalDirections.push(normalize(residual));
    }
    const atoms = [
      { inputScale: 1.25, inputBias: -0.2, outputWeight: 0.9 },
      { inputScale: 1.05, inputBias: 0.15, outputWeight: -0.7 },
      { inputScale: 0.85, inputBias: 0.32, outputWeight: 0.5 }
    ];
    const oracle = (input: number[]): number => 0.07 + atoms.reduce((sum, atom, index) => (
      sum + atom.outputWeight * gelu(atom.inputScale * dot(orthogonalDirections[index], input) + atom.inputBias)
    ), 0);
    const model = decompileRankAdaptiveGeluFfnStructure(oracle, {
      dimension,
      ranges: Array.from({ length: dimension }, () => [-1.5, 1.5] as [number, number]),
      maxChannelCount: 4,
      selectionRestarts: 1,
      gradientProbeCount: 14,
      trainingSampleCount: 1000,
      validationSampleCount: 300,
      randomSeed: 8811,
      restarts: 6,
      iterations: 420,
      learningRate: 0.03
    });
    const inputs = makeInputs(9101, dimension, 700);
    const metrics = regressionMetrics(inputs.map(oracle), predictGeluFfnStructureBatch(model, inputs));
    const projectionScores = orthogonalDirections.map(direction => (
      Math.max(...model.atoms.map(atom => Math.abs(dot(direction, normalize(atom.inputWeight)))))
    ));

    expect(model.recoveryKind).toBe('rank-adaptive-gelu-ffn-structure');
    expect(model.rawChannelCount).toBe(4);
    expect(model.effectiveChannelCount).toBe(3);
    expect(model.atoms).toHaveLength(3);
    expect(model.adaptiveSelection?.candidateCount).toBeGreaterThan(2);
    expect(model.adaptiveSelection?.selectedCandidateKind).toContain('trained-3');
    expect(model.adaptiveSelection?.duplicatePairCount).toBe(0);
    expect(Math.min(...projectionScores)).toBeGreaterThan(0.995);
    expect(metrics.mae).toBeLessThan(0.02);
    expect(metrics.r2).toBeGreaterThan(0.999);
  });

  it('verifies recovered GELU atoms against exact source neuron provenance groups', () => {
    const dimension = 6;
    const firstDirection = normalize([1, 0.2, -0.35, 0.12, -0.18, 0.08]);
    const secondDirection = normalize([-0.22, 0.91, 0.11, -0.16, 0.27, -0.14]);
    const sources = [
      makeSourceNeuron('transformer.h2.mlp.c_fc', 11, firstDirection, 1.12, -0.18, 0.31),
      makeSourceNeuron('transformer.h2.mlp.c_fc', 12, firstDirection, 1.12, -0.18, 0.43),
      makeSourceNeuron('transformer.h2.mlp.c_fc', 87, secondDirection, 0.9, 0.14, -0.52)
    ];
    const model = {
      dimension,
      atoms: [
        makeAtom(firstDirection, 1.12, -0.18, 0.74),
        makeAtom(secondDirection, 0.9, 0.14, -0.52)
      ]
    } as any;
    const report = verifyGeluFfnNeuronProvenance(model, sources, {
      probeInputs: makeInputs(88317, dimension, 90),
      minDirectionCosine: 0.999,
      maxInputWeightNormError: 1e-10,
      maxInputBiasError: 1e-10,
      maxOutputWeightError: 1e-10,
      minActivationCorrelation: 0.999999,
      minContributionCorrelation: 0.999999,
      maxContributionMae: 1e-10,
      minConfidence: 0.999999
    });

    expect(report.verified).toBe(true);
    expect(report.sourceNeuronCount).toBe(3);
    expect(report.sourceGroupCount).toBe(2);
    expect(report.recoveredAtomCount).toBe(2);
    expect(report.duplicateSourceGroups).toHaveLength(1);
    expect(report.duplicateSourceGroups[0].sourceIds).toEqual([
      'transformer.h2.mlp.c_fc[11]',
      'transformer.h2.mlp.c_fc[12]'
    ]);
    expect(report.matches).toHaveLength(2);
    expect(report.matches.every(match => match.whyReacts.formula.includes('GELU'))).toBe(true);
    expect(report.matches.every(match => match.whyReacts.topTriggers.length >= 3)).toBe(true);
    expect(report.matches.some(match => match.sourceIds.length === 2 && match.duplicateSplit)).toBe(true);
    expect(report.unmatchedRecoveredAtomIndices).toHaveLength(0);
    expect(report.unmatchedSourceGroupIds).toHaveLength(0);
  });
});
