import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';

describe('PopulationBuffer', () => {
  test('writeGenome clamps opcodes and constants to fixed genome storage', () => {
    const buffer = new PopulationBuffer(1, 4, 10, 2);
    const opcodes = new Uint32Array([
      PopulationBuffer.pack(OpCode.XN, 0),
      PopulationBuffer.pack(OpCode.CONST, 0),
      PopulationBuffer.pack(OpCode.ADD, 0),
      PopulationBuffer.pack(OpCode.CONST, 1),
      PopulationBuffer.pack(OpCode.MUL, 0)
    ]);
    const constants = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

    expect(() => buffer.writeGenome(0, opcodes, new Uint32Array(opcodes.length), constants)).not.toThrow();

    expect(buffer.lengths[0]).toBe(4);
    expect(Array.from(buffer.getOpcodes(0))).toEqual(Array.from(opcodes.slice(0, 4)));
    expect(Array.from(buffer.getConstants(0))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });
});
