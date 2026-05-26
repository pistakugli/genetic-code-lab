import { OpCode } from '../core/pop-buffer';
import type { TestCase } from '../core/types';

export type ThresholdBoundaryError = {
    ones: number;
    mask: string;
    expected: number;
    predicted: number;
};

export type ThresholdWeightRow = {
    ones: number;
    total: number;
    correct: number;
    accuracy: number;
};

export type ThresholdAudit = {
    exactAccuracy: number;
    byWeight: ThresholdWeightRow[];
    boundaryErrors: ThresholdBoundaryError[];
    lowerBoundary: number;
    upperBoundary: number;
    boundary: {
        total: number;
        fp4: number;
        fn5: number;
    };
};

export function evalThresholdBinary(opcodes: ArrayLike<number>, constants: Float64Array, input: number[]): number {
    const stack: number[] = [];
    for (let i = 0; i < opcodes.length; i++) {
        const raw = opcodes[i];
        const op = raw >>> 24;
        const data = raw & 0xFFFFFF;
        switch (op) {
            case OpCode.XN:
                stack.push(input[data] || 0);
                break;
            case OpCode.CONST:
                stack.push(constants[data] || 0);
                break;
            case OpCode.AND: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 && b > 0.5 ? 1 : 0);
                break;
            }
            case OpCode.OR: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 || b > 0.5 ? 1 : 0);
                break;
            }
            case OpCode.NAND: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 && b > 0.5 ? 0 : 1);
                break;
            }
            case OpCode.NOR: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 || b > 0.5 ? 0 : 1);
                break;
            }
            case OpCode.XOR: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                const aBit = a > 0.5 ? 1 : 0;
                const bBit = b > 0.5 ? 1 : 0;
                stack.push(aBit + bBit === 1 ? 1 : 0);
                break;
            }
            case OpCode.ITE: {
                const whenFalse = stack.pop() || 0;
                const whenTrue = stack.pop() || 0;
                const cond = stack.pop() || 0;
                stack.push(cond > 0.5 ? whenTrue : whenFalse);
                break;
            }
            default:
                stack.push(0);
                break;
        }
    }
    return stack.length > 0 ? stack[stack.length - 1] : 0;
}

export function auditThresholdCases(
    testCases: TestCase<number[], number>[],
    threshold: number,
    opcodes: ArrayLike<number>,
    constants: Float64Array
): ThresholdAudit {
    const inputSize = testCases.length > 0 ? testCases[0].input.length : 0;
    const byWeight = Array.from({ length: inputSize + 1 }, (_, ones) => ({
        ones,
        total: 0,
        correct: 0
    }));
    const boundaryErrors: ThresholdBoundaryError[] = [];
    let correct = 0;
    let lowerTotal = 0;
    let lowerCorrect = 0;
    let upperTotal = 0;
    let upperCorrect = 0;

    for (const tc of testCases) {
        const ones = tc.input.reduce((sum, value) => sum + (value ? 1 : 0), 0);
        const pred = evalThresholdBinary(opcodes, constants, tc.input) > 0 ? 1 : 0;
        byWeight[ones].total++;
        if (ones === threshold - 1) lowerTotal++;
        if (ones === threshold) upperTotal++;

        if (pred === tc.expectedOutput) {
            correct++;
            byWeight[ones].correct++;
            if (ones === threshold - 1) lowerCorrect++;
            if (ones === threshold) upperCorrect++;
        } else if (ones === threshold - 1 || ones === threshold) {
            boundaryErrors.push({
                ones,
                mask: tc.input.map(value => value ? '1' : '0').join(''),
                expected: tc.expectedOutput,
                predicted: pred
            });
        }
    }

    return {
        exactAccuracy: correct / Math.max(1, testCases.length),
        byWeight: byWeight.map(row => ({
            ones: row.ones,
            total: row.total,
            correct: row.correct,
            accuracy: row.correct / Math.max(1, row.total)
        })),
        boundaryErrors,
        lowerBoundary: lowerCorrect / Math.max(1, lowerTotal),
        upperBoundary: upperCorrect / Math.max(1, upperTotal),
        boundary: {
            total: boundaryErrors.length,
            fp4: boundaryErrors.filter(err => err.ones === threshold - 1 && err.expected === 0 && err.predicted === 1).length,
            fn5: boundaryErrors.filter(err => err.ones === threshold && err.expected === 1 && err.predicted === 0).length
        }
    };
}
