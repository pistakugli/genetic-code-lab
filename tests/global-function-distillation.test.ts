import {
  decompileBlackBoxReluNdGlobalFunctionDistillation,
  predictBlackBoxReluNdGlobalFunctionDistillation
} from '../src/solvers/global-function-distillation';
import { GfdCudaClient } from '../scripts/gfd-vp-cuda-client';

describe('Global function distillation with variable projection', () => {
  test('recovers an analytic one-boundary function from held-out value accuracy', () => {
    const teacher = (input: number[]) =>
      0.4 - 0.2 * input[0] + 0.7 * Math.max(0, input[0] + 0.15);
    const model = decompileBlackBoxReluNdGlobalFunctionDistillation(teacher, {
      dimension: 1,
      ranges: [[-1, 1]],
      termCount: 1,
      oracleBudget: 512,
      sampleCount: 512,
      restarts: 8,
      optimizationSteps: 400,
      batchSize: 64,
      randomSeed: 17
    });
    const testPoints = Array.from({ length: 1000 }, (_, index) => [-1 + 2 * index / 999]);
    const error = testPoints.reduce((sum, point) =>
      sum + Math.abs(teacher(point)
        - predictBlackBoxReluNdGlobalFunctionDistillation(model, point)), 0) / testPoints.length;

    expect(error).toBeLessThan(1e-6);
    expect(Math.abs(model.terms[0].offset - 0.15)).toBeLessThan(1e-3);
    expect(model.diagnostics.validationMae).toBeLessThan(1e-6);
  });

  test('uses exactly the sampled oracle values and keeps hidden normals normalized', () => {
    let oracleCalls = 0;
    const model = decompileBlackBoxReluNdGlobalFunctionDistillation(input => {
      oracleCalls++;
      return 0.4 - 0.2 * input[0] + 0.7 * Math.max(0, input[0] + 0.15);
    }, {
      dimension: 1,
      ranges: [[-1, 1]],
      termCount: 1,
      oracleBudget: 192,
      sampleCount: 192,
      restarts: 2,
      optimizationSteps: 40,
      randomSeed: 17
    });

    expect(oracleCalls).toBe(192);
    expect(model.diagnostics.totalOracleCalls).toBe(oracleCalls);
    expect(model.terms).toHaveLength(1);
    expect(Math.abs(Math.abs(model.terms[0].normal[0]) - 1)).toBeLessThan(1e-12);
    expect(Number.isFinite(predictBlackBoxReluNdGlobalFunctionDistillation(model, [0.2]))).toBe(true);
  });

  test('zero oracle budget performs no calls and returns a finite prior model', () => {
    let oracleCalls = 0;
    const model = decompileBlackBoxReluNdGlobalFunctionDistillation(() => {
      oracleCalls++;
      return 1;
    }, {
      dimension: 2,
      ranges: [[-1, 1], [-1, 1]],
      termCount: 2,
      oracleBudget: 0,
      randomSeed: 9
    });

    expect(oracleCalls).toBe(0);
    expect(model.diagnostics.totalOracleCalls).toBe(0);
    expect(model.diagnostics.restartsAttempted).toBe(0);
    expect(Number.isFinite(predictBlackBoxReluNdGlobalFunctionDistillation(model, [0, 0]))).toBe(true);
  });

  test('supports the joint-fit ablation without reporting profiled output fitting', () => {
    let oracleCalls = 0;
    const model = decompileBlackBoxReluNdGlobalFunctionDistillation(input => {
      oracleCalls++;
      return 0.3 + Math.max(0, input[0] - 0.1);
    }, {
      dimension: 1,
      ranges: [[-1, 1]],
      termCount: 1,
      oracleBudget: 128,
      sampleCount: 128,
      restarts: 1,
      optimizationSteps: 30,
      profileOutputLayer: false,
      randomSeed: 3
    });

    expect(oracleCalls).toBe(128);
    expect(model.diagnostics.totalOracleCalls).toBe(oracleCalls);
    expect(Number.isFinite(model.diagnostics.trainingMae)).toBe(true);
  });

  test('rejects invalid oracle budgets before querying', () => {
    let oracleCalls = 0;
    expect(() => decompileBlackBoxReluNdGlobalFunctionDistillation(() => {
      oracleCalls++;
      return 0;
    }, {
      dimension: 1,
      ranges: [[-1, 1]],
      termCount: 1,
      oracleBudget: -1
    })).toThrow(/oracleBudget/);
    expect(oracleCalls).toBe(0);
  });

  test('runs batched hidden-unit optimization on the CUDA device and reports allocated VRAM', async () => {
    const client = await GfdCudaClient.start(process.cwd());
    try {
      const inputs = Array.from({ length: 192 }, (_, index) => {
        const x = -1 + 2 * index / 191;
        return [x, Math.sin(index * 0.13)];
      });
      const outputs = inputs.map(([x, y]) =>
        0.2 * x - 0.1 * y + 0.8 * Math.max(0, 0.6 * x + 0.8 * y + 0.1));
      const model = await client.fit({
        inputs,
        outputs,
        dimension: 2,
        termCount: 1,
        randomSeed: 11,
        restarts: 2,
        optimizationSteps: 12,
        batchSize: 64,
        profileOutputLayer: true
      });

      expect(client.device.computeDevice).toBe('cuda:0');
      expect(model.diagnostics.computeDevice).toBe('cuda:0');
      expect(model.diagnostics.cudaMemoryAllocatedBytes).toBeGreaterThan(0);
      expect(model.diagnostics.trainingTimeMs).toBeGreaterThan(0);
    } finally {
      await client.close();
    }
  });
});
