import { LinearOperators } from '../src/core/linear-operators';
import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';

type ComposeStrategy = 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or' | 'paired-witness';

type DecodedInstruction = {
  op: string;
  data: number;
};

const x = (idx: number) => PopulationBuffer.pack(OpCode.XN, idx);
function decode(buffer: PopulationBuffer, idx: number): DecodedInstruction[] {
  return Array.from(buffer.getOpcodes(idx)).map(raw => {
    const unpacked = PopulationBuffer.unpack(raw);
    return {
      op: OpCode[unpacked.op],
      data: unpacked.data
    };
  });
}

function makeParentBuffer(): PopulationBuffer {
  const source = new PopulationBuffer(2, 16, 10, 1);
  source.writeGenome(0, [x(0)], [], new Float64Array(10));
  source.writeGenome(1, [x(1)], [], new Float64Array(10));
  return source;
}

function compose(strategy: ComposeStrategy): DecodedInstruction[] {
  const source = makeParentBuffer();
  const target = new PopulationBuffer(1, 64, 10, 1);

  LinearOperators.crossoverBooleanThresholdCompose(
    source,
    0,
    1,
    target,
    0,
    64,
    5,
    'upper',
    strategy
  );

  return decode(target, 0);
}

describe('threshold compose operators', () => {
  test('cross emits a structural parent-and-bridge formula, not a baked threshold solution', () => {
    expect(compose('cross')).toEqual([
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 0 },
      { op: 'XN', data: 3 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'XN', data: 2 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'XN', data: 3 },
      { op: 'AND', data: 0 },
      { op: 'OR', data: 0 },
      { op: 'OR', data: 0 },
      { op: 'OR', data: 0 }
    ]);
  });

  test('witnessed-or emits the documented generic challenger formula', () => {
    expect(compose('witnessed-or')).toEqual([
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'OR', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'XN', data: 3 },
      { op: 'OR', data: 0 },
      { op: 'AND', data: 0 },
      { op: 'OR', data: 0 }
    ]);
  });

  test('mirrored-cross emits the opposite bridge orientation without baking a threshold solution', () => {
    expect(compose('mirrored-cross')).toEqual([
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'XN', data: 3 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'XN', data: 3 },
      { op: 'AND', data: 0 },
      { op: 'OR', data: 0 },
      { op: 'OR', data: 0 },
      { op: 'OR', data: 0 }
    ]);
  });

  test('triad-witness emits a parent-gated two-of-three bridge formula', () => {
    expect(compose('triad-witness')).toEqual([
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'OR', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'XN', data: 3 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'XN', data: 4 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 3 },
      { op: 'XN', data: 4 },
      { op: 'AND', data: 0 },
      { op: 'OR', data: 0 },
      { op: 'OR', data: 0 },
      { op: 'AND', data: 0 },
      { op: 'OR', data: 0 }
    ]);
  });

  test('paired-witness emits the incumbent formula used as the adaptive-duel veto baseline', () => {
    expect(compose('paired-witness')).toEqual([
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 0 },
      { op: 'XN', data: 1 },
      { op: 'OR', data: 0 },
      { op: 'XN', data: 2 },
      { op: 'AND', data: 0 },
      { op: 'XN', data: 3 },
      { op: 'AND', data: 0 },
      { op: 'OR', data: 0 }
    ]);
  });
});
