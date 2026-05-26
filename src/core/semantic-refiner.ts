import { BlockEvaluator } from './block-evaluator';
import { OpCode, PopulationBuffer } from './pop-buffer';

export type SemanticBasis = {
  key: string;
  opcodes: number[];
  constants: number[];
  values: Float64Array;
  complexity: number;
};

type SemanticTerm = {
  basis: SemanticBasis;
  coefficient: number;
};

/**
 * Semantic refinement turns useful subexpressions from evolved programs into a
 * sparse fitted ensemble, then writes the fitted expression back as a normal
 * postfix genome. It is deliberately train-only and uses an internal validation
 * split before injecting a candidate.
 */
export class SemanticRefiner {
  public static refineRegressionElites(
    buffer: PopulationBuffer,
    eliteIndices: number[],
    targetIndices: number[],
    inputs: Float64Array,
    expected: Float64Array,
    inputSize: number,
    maxGenomeSize: number,
    maxConstants: number = 10
  ): number {
    if (process.env.GCL_SEMANTIC_REFINER === '0') return 0;
    if (expected.length < 8 || eliteIndices.length === 0 || targetIndices.length === 0) return 0;

    let injected = 0;
    for (let i = 0; i < eliteIndices.length && injected < targetIndices.length; i++) {
      const sourceIdx = eliteIndices[i];
      const targetIdx = targetIndices[injected];
      const candidate = this.buildCandidate(buffer, sourceIdx, inputs, expected, inputSize, maxGenomeSize, maxConstants);
      if (!candidate) continue;
      buffer.writeGenome(targetIdx, candidate.opcodes, [], candidate.constants);
      injected++;
    }
    return injected;
  }

  public static synthesizeFromBases(
    bases: SemanticBasis[],
    expected: Float64Array,
    baselinePredictions: Float64Array | null,
    maxGenomeSize: number,
    maxConstants: number = 10
  ): { opcodes: Uint32Array; constants: Float64Array } | null {
    if (process.env.GCL_SEMANTIC_REFINER === '0') return null;
    if (bases.length < 2 || expected.length < 8) return null;

    const split = this.validationSplit(expected.length);
    if (split.train.length < 3 || split.validation.length < 2) return null;

    const selected = this.selectTerms(bases, expected, split.train, split.validation, maxGenomeSize, maxConstants);
    if (selected.length === 0) return null;

    if (baselinePredictions) {
      const selectedValidationMae = this.termMae(selected, expected, split.validation);
      const selectedTrainMae = this.termMae(selected, expected, split.train);
      const baselineValidationMae = this.mae(baselinePredictions, expected, split.validation);
      const baselineTrainMae = this.mae(baselinePredictions, expected, split.train);
      if (selectedValidationMae > baselineValidationMae - 1e-9) return null;
      if (selectedTrainMae > baselineTrainMae + 1e-8) return null;
    }

    return this.emitTerms(selected, maxGenomeSize, maxConstants);
  }

  private static buildCandidate(
    buffer: PopulationBuffer,
    sourceIdx: number,
    inputs: Float64Array,
    expected: Float64Array,
    inputSize: number,
    maxGenomeSize: number,
    maxConstants: number
  ): { opcodes: Uint32Array; constants: Float64Array } | null {
    const bases = this.extractSemanticBases(buffer, sourceIdx, inputs, inputSize);
    if (bases.length < 2) return null;

    const split = this.validationSplit(expected.length);
    if (split.train.length < 3 || split.validation.length < 2) return null;

    const sourcePredictions = BlockEvaluator.getPredictions(buffer, sourceIdx, inputs, inputSize);
    return this.synthesizeFromBases(bases, expected, sourcePredictions, maxGenomeSize, maxConstants);
  }

  public static extractSemanticBases(
    buffer: PopulationBuffer,
    sourceIdx: number,
    inputs: Float64Array,
    inputSize: number
  ): SemanticBasis[] {
    const numCases = inputs.length / inputSize;
    const opcodes = buffer.getOpcodes(sourceIdx);
    const constants = buffer.getConstants(sourceIdx);
    const stack: SemanticBasis[] = [];
    const bases: SemanticBasis[] = [];
    const seen = new Set<string>();

    const addBasis = (basis: SemanticBasis): SemanticBasis => {
      if (this.valuesAreUsable(basis.values) && !seen.has(basis.key)) {
        seen.add(basis.key);
        bases.push(basis);
      }
      return basis;
    };

    for (let i = 0; i < opcodes.length; i++) {
      const instruction = opcodes[i];
      const op = (instruction >>> 24) as OpCode;
      const data = instruction & 0xFFFFFF;

      switch (op) {
        case OpCode.XN: {
          const values = new Float64Array(numCases);
          for (let row = 0; row < numCases; row++) values[row] = data < inputSize ? inputs[row * inputSize + data] : 0;
          stack.push(addBasis({
            key: `x${data}`,
            opcodes: [PopulationBuffer.pack(OpCode.XN, data)],
            constants: [],
            values,
            complexity: 1
          }));
          break;
        }
        case OpCode.CONST: {
          const value = constants[data] ?? 0;
          const values = new Float64Array(numCases).fill(value);
          stack.push(addBasis({
            key: `c:${this.roundKey(value)}`,
            opcodes: [PopulationBuffer.pack(OpCode.CONST, 0)],
            constants: [value],
            values,
            complexity: 1
          }));
          break;
        }
        case OpCode.ADD:
        case OpCode.SUB:
        case OpCode.MUL:
        case OpCode.DIV: {
          if (stack.length < 2) break;
          const right = stack.pop()!;
          const left = stack.pop()!;
          const values = new Float64Array(numCases);
          for (let row = 0; row < numCases; row++) {
            const a = left.values[row];
            const b = right.values[row];
            if (op === OpCode.ADD) values[row] = a + b;
            else if (op === OpCode.SUB) values[row] = a - b;
            else if (op === OpCode.MUL) values[row] = a * b;
            else values[row] = a / (b >= 0 ? b + 1e-4 : b - 1e-4);
          }
          const name = op === OpCode.ADD ? 'add' : op === OpCode.SUB ? 'sub' : op === OpCode.MUL ? 'mul' : 'div';
          stack.push(addBasis({
            key: `${name}(${left.key},${right.key})`,
            opcodes: [
              ...left.opcodes,
              ...this.shiftConstants(right.opcodes, left.constants.length),
              PopulationBuffer.pack(op, 0)
            ],
            constants: [...left.constants, ...right.constants],
            values,
            complexity: left.complexity + right.complexity + 1
          }));
          break;
        }
        case OpCode.SIN:
        case OpCode.COS:
        case OpCode.EXP:
        case OpCode.LOG:
        case OpCode.SQRT: {
          if (stack.length < 1) break;
          const base = stack.pop()!;
          const values = new Float64Array(numCases);
          for (let row = 0; row < numCases; row++) {
            const value = base.values[row];
            if (op === OpCode.SIN) values[row] = Math.sin(value);
            else if (op === OpCode.COS) values[row] = Math.cos(value);
            else if (op === OpCode.EXP) values[row] = Math.exp(Math.max(-40, Math.min(40, value)));
            else if (op === OpCode.LOG) values[row] = Math.log(Math.abs(value) + 1e-12);
            else values[row] = value >= 0 ? Math.sqrt(value) : 0;
          }
          const name = op === OpCode.SIN ? 'sin' : op === OpCode.COS ? 'cos' : op === OpCode.EXP ? 'exp' : op === OpCode.LOG ? 'log' : 'sqrt';
          stack.push(addBasis({
            key: `${name}(${base.key})`,
            opcodes: [...base.opcodes, PopulationBuffer.pack(op, 0)],
            constants: [...base.constants],
            values,
            complexity: base.complexity + 1
          }));
          break;
        }
      }
    }

    return bases.sort((a, b) => a.complexity - b.complexity || a.key.localeCompare(b.key));
  }

  private static selectTerms(
    bases: SemanticBasis[],
    expected: Float64Array,
    trainRows: number[],
    validationRows: number[],
    maxGenomeSize: number,
    maxConstants: number
  ): SemanticTerm[] {
    const selected: SemanticBasis[] = [];
    let coefficients: number[] = [];
    let bestScore = Number.POSITIVE_INFINITY;
    const maxTerms = Math.min(8, bases.length);

    for (let step = 0; step < maxTerms; step++) {
      let bestTrial: { basis: SemanticBasis; coefficients: number[]; score: number } | null = null;

      for (const basis of bases) {
        if (selected.some(item => item.key === basis.key)) continue;
        const trial = [...selected, basis];
        if (!this.fitsGenomeBudget(trial, maxGenomeSize, maxConstants)) continue;
        const trialCoefficients = this.fitLinearBasis(trial.map(item => item.values), expected, trainRows);
        const validationMae = this.basisMae(trial, trialCoefficients, expected, validationRows);
        const trainMae = this.basisMae(trial, trialCoefficients, expected, trainRows);
        const score = validationMae + trainMae * 0.05 + basis.complexity * 1e-8;
        if (Number.isFinite(score) && (!bestTrial || score < bestTrial.score)) {
          bestTrial = { basis, coefficients: trialCoefficients, score };
        }
      }

      if (!bestTrial) break;
      if (bestScore - bestTrial.score < 1e-10 && selected.length > 0) break;
      selected.push(bestTrial.basis);
      coefficients = bestTrial.coefficients;
      bestScore = bestTrial.score;
    }

    return selected
      .map((basis, index) => ({ basis, coefficient: coefficients[index] || 0 }))
      .filter(term => Number.isFinite(term.coefficient) && Math.abs(term.coefficient) > 1e-9);
  }

  private static emitTerms(
    terms: SemanticTerm[],
    maxGenomeSize: number,
    maxConstants: number
  ): { opcodes: Uint32Array; constants: Float64Array } | null {
    const opcodes: number[] = [];
    const constants: number[] = [];
    let emitted = 0;

    for (const term of terms) {
      if (constants.length + 1 + term.basis.constants.length > maxConstants) continue;
      const coefficientIndex = constants.length;
      constants.push(term.coefficient);
      opcodes.push(PopulationBuffer.pack(OpCode.CONST, coefficientIndex));
      opcodes.push(...this.shiftConstants(term.basis.opcodes, constants.length));
      constants.push(...term.basis.constants);
      opcodes.push(PopulationBuffer.pack(OpCode.MUL, 0));
      emitted++;
      if (emitted > 1) opcodes.push(PopulationBuffer.pack(OpCode.ADD, 0));
    }

    if (emitted === 0 || opcodes.length > maxGenomeSize) return null;
    const paddedConstants = new Float64Array(maxConstants);
    for (let i = 0; i < constants.length && i < maxConstants; i++) paddedConstants[i] = constants[i];
    return { opcodes: new Uint32Array(opcodes), constants: paddedConstants };
  }

  private static validationSplit(length: number): { train: number[]; validation: number[] } {
    const train: number[] = [];
    const validation: number[] = [];
    for (let i = 0; i < length; i++) {
      if (i % 4 === 3) validation.push(i);
      else train.push(i);
    }
    return { train, validation };
  }

  private static fitsGenomeBudget(bases: SemanticBasis[], maxGenomeSize: number, maxConstants: number): boolean {
    let opcodeCount = 0;
    let constantCount = 0;
    for (let i = 0; i < bases.length; i++) {
      opcodeCount += 2 + bases[i].opcodes.length;
      constantCount += 1 + bases[i].constants.length;
      if (i > 0) opcodeCount += 1;
    }
    return opcodeCount <= maxGenomeSize && constantCount <= maxConstants;
  }

  private static valuesAreUsable(values: Float64Array): boolean {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (!Number.isFinite(value) || Math.abs(value) > 1e9) return false;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    return max - min > 1e-12 || Math.abs(max) > 1e-12;
  }

  private static shiftConstants(opcodes: ArrayLike<number>, shift: number): number[] {
    const shifted: number[] = [];
    for (let i = 0; i < opcodes.length; i++) {
      const instruction = opcodes[i];
      const op = (instruction >>> 24) as OpCode;
      const data = instruction & 0xFFFFFF;
      shifted.push(op === OpCode.CONST ? PopulationBuffer.pack(OpCode.CONST, data + shift) : instruction);
    }
    return shifted;
  }

  private static fitLinearBasis(columns: Float64Array[], target: Float64Array, rows: number[]): number[] {
    const n = columns.length;
    const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const rhs = new Array<number>(n).fill(0);
    for (const row of rows) {
      for (let i = 0; i < n; i++) {
        const vi = columns[i][row];
        rhs[i] += vi * target[row];
        for (let j = 0; j < n; j++) matrix[i][j] += vi * columns[j][row];
      }
    }
    for (let i = 0; i < n; i++) matrix[i][i] += i === 0 ? 1e-12 : 1e-10;
    return this.solveLinearSystem(matrix, rhs);
  }

  private static solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
    const n = rhs.length;
    const a = matrix.map((row, i) => [...row, rhs[i]]);
    for (let pivot = 0; pivot < n; pivot++) {
      let best = pivot;
      for (let row = pivot + 1; row < n; row++) {
        if (Math.abs(a[row][pivot]) > Math.abs(a[best][pivot])) best = row;
      }
      if (Math.abs(a[best][pivot]) < 1e-12) continue;
      if (best !== pivot) [a[pivot], a[best]] = [a[best], a[pivot]];
      const divisor = a[pivot][pivot];
      for (let col = pivot; col <= n; col++) a[pivot][col] /= divisor;
      for (let row = 0; row < n; row++) {
        if (row === pivot) continue;
        const factor = a[row][pivot];
        for (let col = pivot; col <= n; col++) a[row][col] -= factor * a[pivot][col];
      }
    }
    return a.map(row => Number.isFinite(row[n]) ? row[n] : 0);
  }

  private static basisMae(bases: SemanticBasis[], coefficients: number[], expected: Float64Array, rows: number[]): number {
    let total = 0;
    for (const row of rows) {
      let predicted = 0;
      for (let i = 0; i < bases.length; i++) predicted += (coefficients[i] || 0) * bases[i].values[row];
      total += Math.abs(predicted - expected[row]);
    }
    return total / Math.max(1, rows.length);
  }

  private static termMae(terms: SemanticTerm[], expected: Float64Array, rows: number[]): number {
    let total = 0;
    for (const row of rows) {
      let predicted = 0;
      for (const term of terms) predicted += term.coefficient * term.basis.values[row];
      total += Math.abs(predicted - expected[row]);
    }
    return total / Math.max(1, rows.length);
  }

  private static mae(predictions: Float64Array, expected: Float64Array, rows: number[]): number {
    let total = 0;
    for (const row of rows) total += Math.abs(predictions[row] - expected[row]);
    return total / Math.max(1, rows.length);
  }

  private static roundKey(value: number): string {
    return Number.isFinite(value) ? value.toPrecision(8) : 'invalid';
  }
}
