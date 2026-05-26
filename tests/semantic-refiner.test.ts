import { BlockEvaluator } from '../src/core/block-evaluator';
import { PopulationBuffer, OpCode } from '../src/core/pop-buffer';
import { SemanticArchive } from '../src/core/semantic-archive';
import { SemanticRefiner } from '../src/core/semantic-refiner';

function mae(predictions: Float64Array, expected: Float64Array): number {
  let total = 0;
  for (let i = 0; i < expected.length; i++) total += Math.abs(predictions[i] - expected[i]);
  return total / expected.length;
}

describe('SemanticRefiner', () => {
  it('recombines evolved subexpressions into a fitted sparse genome', () => {
    const cases = 41;
    const inputs = new Float64Array(cases);
    const expected = new Float64Array(cases);
    for (let i = 0; i < cases; i++) {
      const x = -2 + (4 * i) / (cases - 1);
      inputs[i] = x;
      expected[i] = 0.5 * Math.sin(x) + 2 * x * x + 0.25;
    }

    const buffer = new PopulationBuffer(2, 64, 10, cases);
    const sourceOpcodes = [
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.SIN, 0),
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.MUL, 0),
      PopulationBuffer.pack(OpCode.ADD, 0),
      PopulationBuffer.pack(OpCode.CONST, 0),
      PopulationBuffer.pack(OpCode.ADD, 0)
    ];
    buffer.writeGenome(0, sourceOpcodes, [], new Float64Array([1]));
    BlockEvaluator.evaluateBlock(buffer, 0, 1, inputs, expected, 1);
    const before = mae(BlockEvaluator.getPredictions(buffer, 0, inputs, 1), expected);

    const injected = SemanticRefiner.refineRegressionElites(
      buffer,
      [0],
      [1],
      inputs,
      expected,
      1,
      64
    );

    expect(injected).toBe(1);
    BlockEvaluator.evaluateBlock(buffer, 1, 1, inputs, expected, 1);
    const after = mae(BlockEvaluator.getPredictions(buffer, 1, inputs, 1), expected);

    expect(after).toBeLessThan(before * 0.05);
    expect(buffer.fitness[1]).toBeGreaterThan(buffer.fitness[0]);
  });

  it('stores useful subexpressions across elites and injects an archive candidate', () => {
    const cases = 41;
    const inputs = new Float64Array(cases);
    const expected = new Float64Array(cases);
    for (let i = 0; i < cases; i++) {
      const x = -2 + (4 * i) / (cases - 1);
      inputs[i] = x;
      expected[i] = 0.5 * Math.sin(x) + 2 * x * x + 0.25;
    }

    const buffer = new PopulationBuffer(3, 64, 10, cases);
    buffer.writeGenome(0, [
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.SIN, 0)
    ], [], new Float64Array([]));
    buffer.writeGenome(1, [
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.MUL, 0)
    ], [], new Float64Array([]));
    BlockEvaluator.evaluateBlock(buffer, 0, 2, inputs, expected, 1);
    const before = mae(BlockEvaluator.getPredictions(buffer, 0, inputs, 1), expected);

    const archive = new SemanticArchive();
    archive.recordElites(buffer, [0, 1], inputs, expected, 1);
    const injected = archive.injectCandidate(buffer, 0, 2, inputs, expected, 1, 64);

    expect(injected).toBe(true);
    expect(archive.size()).toBeGreaterThanOrEqual(2);
    BlockEvaluator.evaluateBlock(buffer, 2, 1, inputs, expected, 1);
    const after = mae(BlockEvaluator.getPredictions(buffer, 2, inputs, 1), expected);

    expect(after).toBeLessThan(before * 0.05);
    expect(buffer.fitness[2]).toBeGreaterThan(buffer.fitness[0]);
  });

  it('uses behavior novelty to prune semantic duplicates before diverse bases', () => {
    const cases = 41;
    const inputs = new Float64Array(cases);
    const expected = new Float64Array(cases);
    for (let i = 0; i < cases; i++) {
      const x = -2 + (4 * i) / (cases - 1);
      inputs[i] = x;
      expected[i] = x + 0.3 * Math.sin(x);
    }

    const buffer = new PopulationBuffer(3, 64, 10, cases);
    buffer.writeGenome(0, [
      PopulationBuffer.pack(OpCode.XN, 0)
    ], [], new Float64Array([]));
    buffer.writeGenome(1, [
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.CONST, 0),
      PopulationBuffer.pack(OpCode.MUL, 0)
    ], [], new Float64Array([2]));
    buffer.writeGenome(2, [
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.SIN, 0)
    ], [], new Float64Array([]));

    const archive = new SemanticArchive(2);
    archive.recordElites(buffer, [0, 1, 2], inputs, expected, 1);
    const keys = archive.diagnostics().map(entry => entry.key);

    expect(keys).toContain('x0');
    expect(keys).toContain('sin(x0)');
    expect(keys.some(key => key.startsWith('mul('))).toBe(false);
  });
});
