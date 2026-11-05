import {
  createGlobalSpectralMomentSamples,
  decompileBlackBoxReluNdGlobalSpectralMomentInversion,
  predictBlackBoxReluNdGlobalSpectralMomentInversion
} from '../src/solvers/global-spectral-moment-inversion';
import { GsiCudaClient } from '../scripts/gsi-cuda-client';

describe('Global Spectral-Moment Inversion', () => {
  test('uses deterministic bounded global Gaussian samples and counts actual oracle calls', () => {
    let calls = 0;
    const config = {
      dimension: 8,
      oracleBudget: 40,
      sampleCount: 31,
      randomSeed: 81
    };
    const first = createGlobalSpectralMomentSamples(input => {
      calls++;
      return input.reduce((sum, value) => sum + value, 0);
    }, config);
    const second = createGlobalSpectralMomentSamples(() =>
      0, config);

    expect(calls).toBe(31);
    expect(first.diagnostics.totalOracleCalls).toBe(calls);
    expect(first.inputs).toEqual(second.inputs);
    expect(first.inputs.flat().every(value => Math.abs(value) <= 1)).toBe(true);
    expect(second.outputs).toHaveLength(31);
  });

  test('zero oracle budget returns an explicit empty incomplete reconstruction', async () => {
    let calls = 0;
    const model = await decompileBlackBoxReluNdGlobalSpectralMomentInversion(
      () => {
        calls++;
        return 2;
      },
      { dimension: 2, oracleBudget: 0 },
      { invert: async () => { throw new Error('backend must not be called'); } }
    );

    expect(calls).toBe(0);
    expect(model.diagnostics.totalOracleCalls).toBe(0);
    expect(model.reconstructionComplete).toBe(false);
    expect(predictBlackBoxReluNdGlobalSpectralMomentInversion(model, [0.2, -0.3])).toBe(0);
  });

  test('CUDA matrix-pencil inversion recovers a one-plane global function from values', async () => {
    const worker = await GsiCudaClient.start(process.cwd());
    let calls = 0;
    try {
      const teacher = ([x, y]: number[]) =>
        0.2 - 0.15 * x + 0.1 * y + 0.8 * Math.max(0, 0.6 * x + 0.8 * y + 0.12);
      const model = await decompileBlackBoxReluNdGlobalSpectralMomentInversion(
        input => {
          calls++;
          return teacher(input);
        },
        {
          dimension: 2,
          oracleBudget: 4096,
          sampleCount: 4096,
          sampleSigma: 0.2,
          maximumAbsoluteInput: 1,
          randomSeed: 29
        },
        worker
      );
      const testPoints = Array.from({ length: 256 }, (_, index) => {
        const angle = 2 * Math.PI * index / 256;
        return [0.85 * Math.cos(angle), 0.85 * Math.sin(angle)];
      });
      const mae = testPoints.reduce((sum, point) =>
        sum + Math.abs(teacher(point)
          - predictBlackBoxReluNdGlobalSpectralMomentInversion(model, point)), 0) / testPoints.length;

      expect(worker.device.computeDevice).toBe('cuda:0');
      expect(model.diagnostics.computeDevice).toBe('cuda:0');
      expect(model.diagnostics.cudaMemoryAllocatedBytes).toBeGreaterThan(0);
      expect(model.diagnostics.momentAndInversionTimeMs).toBeGreaterThan(0);
      expect(calls).toBe(model.diagnostics.totalOracleCalls);
      expect(calls).toBeLessThanOrEqual(4096);
      expect(model.diagnostics.estimatedRank).toBe(1);
      expect(model.terms).toHaveLength(1);
      expect(Math.abs(model.terms[0].offset + 0.12)).toBeLessThan(0.06);
      expect(mae).toBeLessThan(0.08);
    } finally {
      await worker.close();
    }
  });

  test('rejects unusable sampling scales before querying the oracle', () => {
    let calls = 0;
    expect(() => createGlobalSpectralMomentSamples(() => {
      calls++;
      return 0;
    }, {
      dimension: 3,
      oracleBudget: 5,
      sampleSigma: 0.5,
      maximumAbsoluteInput: 1
    })).toThrow(/maximumAbsoluteInput\/sampleSigma/);
    expect(calls).toBe(0);
  });
});
