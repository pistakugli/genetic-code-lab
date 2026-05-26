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

export function applyThresholdCourtFinalizer(
    testCases: TestCase<number[], number>[],
    threshold: number,
    current: BinaryProgram,
    maxFinalGenomeSize: number = 576,
    minBaseFitness: number = 0.98,
    maxBoundaryErrors: number = 6
): BinaryProgram {
    if (current.fitness < minBaseFitness) return current;

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
