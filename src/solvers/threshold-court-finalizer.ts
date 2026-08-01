import { OpCode, PopulationBuffer } from '../core/pop-buffer';
import type { TestCase } from '../core/types';
import { auditThresholdCases } from './threshold-audit';

export type BinaryProgram = {
    opcodes: Uint32Array;
    constants: Float64Array;
    fitness: number;
};

type BoundaryError = {
    mask: string;
    expected: number;
    predicted: number;
};

type Audit = {
    fitness: number;
    boundaryErrors: BoundaryError[];
};

function appendFoldedTerm(out: number[], vars: number[], op: OpCode.AND | OpCode.OR): void {
    if (vars.length === 0) return;
    for (const variable of vars) out.push(PopulationBuffer.pack(OpCode.XN, variable));
    for (let i = 1; i < vars.length; i++) out.push(PopulationBuffer.pack(op, 0));
}

function inactiveVarsFromMask(mask: string): number[] {
    const vars: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i] === '0') vars.push(i);
    return vars;
}

function activeVarsFromMask(mask: string): number[] {
    const vars: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i] === '1') vars.push(i);
    return vars;
}

function appendNegativeGuard(candidate: number[], mask: string): void {
    appendFoldedTerm(candidate, inactiveVarsFromMask(mask), OpCode.OR);
    candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
}

function appendPositiveTerm(candidate: number[], mask: string): void {
    appendFoldedTerm(candidate, activeVarsFromMask(mask), OpCode.AND);
    candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
}

function auditThresholdBoundary(
    testCases: TestCase<number[], number>[],
    threshold: number,
    opcodes: ArrayLike<number>,
    constants: Float64Array
): Audit {
    const audit = auditThresholdCases(testCases, threshold, opcodes, constants);
    return {
        fitness: audit.exactAccuracy,
        boundaryErrors: audit.boundaryErrors.map(err => ({
            mask: err.mask,
            expected: err.expected,
            predicted: err.predicted
        }))
    };
}

function collectMasksBySubset(items: string[], subsetMask: number): string[] {
    const selected: string[] = [];
    for (let i = 0; i < items.length; i++) {
        if ((subsetMask & (1 << i)) !== 0) selected.push(items[i]);
    }
    return selected;
}

/**
 * Refuse to run unless the supplied cases are the complete truth table for their input size.
 *
 * Exactness here is what separates synthesis from memorisation, so it is verified rather
 * than assumed: the count must be 2^n, every input must be binary and of the same width, and
 * no assignment may repeat.
 */
function assertExhaustiveTruthTable(testCases: TestCase<number[], number>[]): void {
    if (testCases.length === 0) throw new Error('threshold court finalizer requires a non-empty truth table');
    const inputSize = testCases[0].input.length;
    if (inputSize <= 0 || inputSize > 30) {
        throw new Error(`threshold court finalizer input size ${inputSize} is outside the enumerable range`);
    }
    const expected = 1 << inputSize;
    if (testCases.length !== expected) {
        throw new Error(
            `threshold court finalizer requires an exhaustive truth table: got ${testCases.length} cases for ` +
            `${inputSize} inputs, expected ${expected}. On a sampled set this finaliser would memorise the cases it is scored on.`
        );
    }
    const seen = new Set<number>();
    for (const testCase of testCases) {
        if (testCase.input.length !== inputSize) {
            throw new Error('threshold court finalizer requires every case to have the same input width');
        }
        let key = 0;
        for (let bit = 0; bit < inputSize; bit++) {
            const value = testCase.input[bit];
            if (value !== 0 && value !== 1) {
                throw new Error('threshold court finalizer requires binary inputs');
            }
            if (value === 1) key |= 1 << bit;
        }
        if (seen.has(key)) throw new Error('threshold court finalizer requires each assignment to appear once');
        seen.add(key);
    }
}

export function applyThresholdCourtFinalizer(
    testCases: TestCase<number[], number>[],
    threshold: number,
    current: BinaryProgram,
    maxFinalGenomeSize: number = 576,
    minBaseFitness: number = 0.98,
    maxBoundaryErrors: number = 6
): BinaryProgram {
    if (current.fitness < minBaseFitness) return current;
    // This finaliser reads the exact inputs the program got wrong, appends literal guard
    // clauses for those masks, and then re-scores against the same testCases. That is
    // legitimate only because the threshold lane hands it the complete truth table:
    // createMajorityProblem enumerates all 1 << n assignments, so there is no held-out data
    // to overfit to and the specification is total. On a sampled set the identical code
    // would be memorising the test cases and reporting the memorised score as accuracy.
    //
    // That safety was an unstated assumption about the caller. Checking it means a future
    // change to sampled cases fails loudly here instead of quietly turning this into
    // overfitting that still reports a high number.
    assertExhaustiveTruthTable(testCases);

    const baseline = auditThresholdBoundary(testCases, threshold, current.opcodes, current.constants);
    if (baseline.boundaryErrors.length === 0 || baseline.boundaryErrors.length > maxBoundaryErrors) return current;

    const negativeMasks = baseline.boundaryErrors
        .filter(err => err.expected === 0 && err.predicted === 1)
        .map(err => err.mask);
    const positiveMasks = baseline.boundaryErrors
        .filter(err => err.expected === 1 && err.predicted === 0)
        .map(err => err.mask);

    let bestFitness = baseline.fitness;
    let bestOpcodes = current.opcodes;

    const negativeCombos = 1 << negativeMasks.length;
    const positiveCombos = 1 << positiveMasks.length;

    for (let negSubset = negativeCombos - 1; negSubset >= 0; negSubset--) {
        for (let posSubset = positiveCombos - 1; posSubset >= 0; posSubset--) {
            if (negSubset === 0 && posSubset === 0) continue;

            const candidate = Array.from(current.opcodes);
            for (const mask of collectMasksBySubset(negativeMasks, negSubset)) appendNegativeGuard(candidate, mask);
            for (const mask of collectMasksBySubset(positiveMasks, posSubset)) appendPositiveTerm(candidate, mask);
            if (candidate.length > maxFinalGenomeSize) continue;

            const audit = auditThresholdBoundary(testCases, threshold, candidate, current.constants);
            if (audit.fitness > bestFitness + 1e-9) {
                bestFitness = audit.fitness;
                bestOpcodes = Uint32Array.from(candidate);
                continue;
            }
            if (Math.abs(audit.fitness - bestFitness) <= 1e-9 && candidate.length < bestOpcodes.length) {
                bestOpcodes = Uint32Array.from(candidate);
            }
        }
    }

    if (bestOpcodes === current.opcodes) return current;
    return {
        opcodes: bestOpcodes,
        constants: current.constants,
        fitness: bestFitness
    };
}
