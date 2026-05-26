import { OpCode, PopulationBuffer } from './pop-buffer';

/**
 * Genetic Code Lab - Block Evaluator (DOGP)
 * High-throughput kernel for linearized genomes with Asymptotic Robust Loss.
 */
export class BlockEvaluator {

    // Configurable division mode for scientific auditing
    public static divisionMode: 'strict' | 'mild' | 'safe' | 'strict-quotient' = 'strict';

    public static evaluateBlock(
        buffer: PopulationBuffer,
        startIndex: number,
        count: number,
        inputs: Float64Array,
        expected: Float64Array,
        inputSize: number,
        fitnessMode: 'regression' | 'classification' = 'regression',
        classificationWeight: number = 0.8
    ): Float64Array {
        const numTestCases = expected.length;
        const results = new Float64Array(count);
        const stackVec = new Float64Array(64 * numTestCases);

        for (let g = 0; g < count; g++) {
            const idx = startIndex + g;
            const offset = buffer.offsets[idx];
            const len = buffer.lengths[idx];
            const cOffset = idx * 10;

            let sp = 0;
            let quotientPenalty = 0;
            let invalidVariablePenalty = 0;

            for (let i = 0; i < len; i++) {
                const instruction = buffer.geneBuffer[offset + i];
                const op = (instruction >>> 24) as OpCode;
                const data = instruction & 0xFFFFFF;

                switch (op) {
                    case OpCode.XN:
                        if (data < inputSize) {
                            for (let t = 0; t < numTestCases; t++) stackVec[sp * numTestCases + t] = inputs[t * inputSize + data];
                        } else {
                            invalidVariablePenalty += 1;
                            for (let t = 0; t < numTestCases; t++) stackVec[sp * numTestCases + t] = 0;
                        }
                        sp++;
                        break;
                    case OpCode.CONST:
                        const val = buffer.constBuffer[cOffset + data];
                        for (let t = 0; t < numTestCases; t++) stackVec[sp * numTestCases + t] = val;
                        sp++;
                        break;
                    case OpCode.ADD:
                        sp--;
                        for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] += stackVec[sp * numTestCases + t];
                        break;
                    case OpCode.SUB:
                        sp--;
                        for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] -= stackVec[sp * numTestCases + t];
                        break;
                    case OpCode.MUL:
                        sp--;
                        for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] *= stackVec[sp * numTestCases + t];
                        break;
                    case OpCode.DIV:
                        sp--;
                        for (let t = 0; t < numTestCases; t++) {
                            const b = stackVec[sp * numTestCases + t];
                            const a = stackVec[(sp - 1) * numTestCases + t];
                            
                            if (BlockEvaluator.divisionMode === 'strict-quotient') {
                                const absA = Math.abs(a);
                                const absB = Math.abs(b);

                                if (absA < 0.1 && absB < 0.1) {
                                    quotientPenalty += (0.1 - absA) + (0.1 - absB); 
                                }
                                if (absB < 0.1) {
                                    quotientPenalty += (0.1 - absB) * 10; 
                                }

                                let safeB = b;
                                if (Math.abs(b) < 1e-3) {
                                    safeB = b >= 0 ? 1e-3 : -1e-3;
                                    quotientPenalty += 1.0; 
                                }
                                
                                stackVec[(sp - 1) * numTestCases + t] = a / safeB;
                            } else {
                                // Default regularization
                                const safeB = b >= 0 ? b + 1e-4 : b - 1e-4;
                                stackVec[(sp - 1) * numTestCases + t] = a / safeB;
                            }
                        }
                        break;
                    case OpCode.SIN:
                        if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.sin(stackVec[(sp - 1) * numTestCases + t]);
                        break;
                    case OpCode.COS:
                        if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.cos(stackVec[(sp - 1) * numTestCases + t]);
                        break;
                    case OpCode.EXP:
                        if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.exp(Math.max(-40, Math.min(40, stackVec[(sp - 1) * numTestCases + t])));
                        break;
                    case OpCode.LOG:
                        if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.log(Math.abs(stackVec[(sp - 1) * numTestCases + t]) + 1e-12);
                        break;
                    case OpCode.SQRT:
                        if (sp > 0) for (let t = 0; t < numTestCases; t++) {
                            const value = stackVec[(sp - 1) * numTestCases + t];
                            stackVec[(sp - 1) * numTestCases + t] = value >= 0 ? Math.sqrt(value) : 0;
                        }
                        break;
                    case OpCode.AND:
                        if (sp > 1) {
                            sp--;
                            for (let t = 0; t < numTestCases; t++) {
                                const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                                const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                                stackVec[(sp - 1) * numTestCases + t] = (a && b) ? 1 : 0;
                            }
                        }
                        break;
                    case OpCode.OR:
                        if (sp > 1) {
                            sp--;
                            for (let t = 0; t < numTestCases; t++) {
                                const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                                const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                                stackVec[(sp - 1) * numTestCases + t] = (a || b) ? 1 : 0;
                            }
                        }
                        break;
                    case OpCode.NAND:
                        if (sp > 1) {
                            sp--;
                            for (let t = 0; t < numTestCases; t++) {
                                const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                                const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                                stackVec[(sp - 1) * numTestCases + t] = (a && b) ? 0 : 1;
                            }
                        }
                        break;
                    case OpCode.NOR:
                        if (sp > 1) {
                            sp--;
                            for (let t = 0; t < numTestCases; t++) {
                                const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                                const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                                stackVec[(sp - 1) * numTestCases + t] = (a || b) ? 0 : 1;
                            }
                        }
                        break;
                    case OpCode.XOR:
                        if (sp > 1) {
                            sp--;
                            for (let t = 0; t < numTestCases; t++) {
                                const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                                const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                                stackVec[(sp - 1) * numTestCases + t] = (a !== b) ? 1 : 0;
                            }
                        }
                        break;
                    case OpCode.ITE:
                        if (sp > 2) {
                            sp -= 2;
                            for (let t = 0; t < numTestCases; t++) {
                                const cond = stackVec[(sp - 1) * numTestCases + t];
                                const whenTrue = stackVec[sp * numTestCases + t];
                                const whenFalse = stackVec[(sp + 1) * numTestCases + t];
                                stackVec[(sp - 1) * numTestCases + t] = cond > 0.5 ? whenTrue : whenFalse;
                            }
                        }
                        break;
                }
            }

            let totalAbsoluteError = 0;
            let weightSum = 0;
            const finalPos = 0;
            const errBase = idx * buffer.maxTestCases;

            for (let t = 0; t < numTestCases; t++) {
                let pred = stackVec[finalPos + t];
                if (!isFinite(pred)) pred = 1e12;
                if (Math.abs(pred) > 1e11) pred = (pred >= 0 ? 1e12 : -1e12);

                const target = expected[t];
                const weight = buffer.caseWeights[t] || 1.0;

                let diff = Math.abs(pred - target);
                // Phase 56: Unified Asymptotic Sentinel (1e12)
                if (!isFinite(target) || Math.abs(target) > 1e11) {
                    diff = Math.abs(pred - 1e12);
                }

                // Phase 59: Direct MAE (Eliminating Log-Loss mismatch)
                const err = diff + (quotientPenalty / numTestCases) + (invalidVariablePenalty / numTestCases);

                totalAbsoluteError += err * weight;
                weightSum += weight;
                buffer.caseErrors[errBase + t] = err;
            }

            // Phase 67: Variable Coverage Enforcement (Frontier Protocol)
            // Detect which variables are actually touched by the opcodes
            let variableMask = 0;
            for (let i = 0; i < len; i++) {
                const instruction = buffer.geneBuffer[offset + i];
                if ((instruction >>> 24) === OpCode.XN && (instruction & 0xFFFFFF) < inputSize) {
                    variableMask |= (1 << (instruction & 0xFFFFFF));
                }
            }
            
            // Expected mask for inputSize variables
            const expectedMask = (1 << inputSize) - 1;
            const variablesUsed = Array.from({length: inputSize}, (_, i) => (variableMask >> i) & 1).filter(v => v === 1).length;
            const variableCoverageRatio = variablesUsed / inputSize;

            const meanMAE = totalAbsoluteError / (weightSum || 1);
            let regressionFitness = 1 / (1 + meanMAE);

            // Apply proportional penalty for variable blindness
            // If only 1 out of 2 variables is used, fitness is halved.
            if (inputSize > 1) {
                regressionFitness *= variableCoverageRatio;
            }

            if (isNaN(regressionFitness) || !isFinite(regressionFitness)) regressionFitness = 0.000001;

            // Phase 43: Classification-aware fitness blending
            if (fitnessMode === 'classification') {
                let correctWeight = 0;
                let totalClassWeight = 0;
                for (let t = 0; t < numTestCases; t++) {
                    const pred = stackVec[finalPos + t];
                    const target = expected[t];
                    const weight = buffer.caseWeights[t] || 1.0;
                    totalClassWeight += weight;
                    // Decision boundary: pred > 0 → class 1, else class 0
                    if ((pred > 0 && target === 1) || (pred <= 0 && target === 0)) {
                        correctWeight += weight;
                    }
                }
                const accuracy = correctWeight / (totalClassWeight || numTestCases || 1);
                // Blend: classificationWeight * accuracy + (1 - classificationWeight) * regression
                results[g] = classificationWeight * accuracy + (1 - classificationWeight) * regressionFitness;
            } else {
                results[g] = regressionFitness;
            }

            buffer.fitness[idx] = results[g];
        }

        return results;
    }

    /**
     * Phase 44: Returns raw numerical predictions for a single individual.
     */
    public static getPredictions(
        buffer: PopulationBuffer,
        individualIdx: number,
        inputs: Float64Array,
        inputSize: number
    ): Float64Array {
        const numTestCases = inputs.length / inputSize;
        const results = new Float64Array(numTestCases);
        const stackVec = new Float64Array(64 * numTestCases);

        const offset = buffer.offsets[individualIdx];
        const len = buffer.lengths[individualIdx];
        const cOffset = individualIdx * 10;

        let sp = 0;

        for (let i = 0; i < len; i++) {
            const instruction = buffer.geneBuffer[offset + i];
            const op = (instruction >>> 24) as OpCode;
            const data = instruction & 0xFFFFFF;

            switch (op) {
                case OpCode.XN:
                    for (let t = 0; t < numTestCases; t++) stackVec[sp * numTestCases + t] = inputs[t * inputSize + data];
                    sp++;
                    break;
                case OpCode.CONST:
                    const val = buffer.constBuffer[cOffset + data];
                    for (let t = 0; t < numTestCases; t++) stackVec[sp * numTestCases + t] = val;
                    sp++;
                    break;
                case OpCode.ADD:
                    sp--;
                    for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] += stackVec[sp * numTestCases + t];
                    break;
                case OpCode.SUB:
                    sp--;
                    for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] -= stackVec[sp * numTestCases + t];
                    break;
                case OpCode.MUL:
                    sp--;
                    for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] *= stackVec[sp * numTestCases + t];
                    break;
                case OpCode.DIV:
                    sp--;
                    for (let t = 0; t < numTestCases; t++) {
                        const b = stackVec[sp * numTestCases + t];
                        if (Math.abs(b) < 1e-12) {
                            if (BlockEvaluator.divisionMode === 'strict') {
                                stackVec[(sp - 1) * numTestCases + t] = 1e12;
                            } else if (BlockEvaluator.divisionMode === 'mild') {
                                stackVec[(sp - 1) * numTestCases + t] = 1000;
                            } else {
                                stackVec[(sp - 1) * numTestCases + t] = 1.0;
                            }
                        } else {
                            stackVec[(sp - 1) * numTestCases + t] /= b;
                        }
                    }
                    break;
                case OpCode.SIN:
                    if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.sin(stackVec[(sp - 1) * numTestCases + t]);
                    break;
                case OpCode.COS:
                    if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.cos(stackVec[(sp - 1) * numTestCases + t]);
                    break;
                case OpCode.EXP:
                    if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.exp(Math.max(-40, Math.min(40, stackVec[(sp - 1) * numTestCases + t])));
                    break;
                case OpCode.LOG:
                    if (sp > 0) for (let t = 0; t < numTestCases; t++) stackVec[(sp - 1) * numTestCases + t] = Math.log(Math.abs(stackVec[(sp - 1) * numTestCases + t]) + 1e-12);
                    break;
                case OpCode.SQRT:
                    if (sp > 0) for (let t = 0; t < numTestCases; t++) {
                        const value = stackVec[(sp - 1) * numTestCases + t];
                        stackVec[(sp - 1) * numTestCases + t] = value >= 0 ? Math.sqrt(value) : 0;
                    }
                    break;
                case OpCode.AND:
                    if (sp > 1) {
                        sp--;
                        for (let t = 0; t < numTestCases; t++) {
                            const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                            const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                            stackVec[(sp - 1) * numTestCases + t] = (a && b) ? 1 : 0;
                        }
                    }
                    break;
                case OpCode.OR:
                    if (sp > 1) {
                        sp--;
                        for (let t = 0; t < numTestCases; t++) {
                            const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                            const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                            stackVec[(sp - 1) * numTestCases + t] = (a || b) ? 1 : 0;
                        }
                    }
                    break;
                case OpCode.NAND:
                    if (sp > 1) {
                        sp--;
                        for (let t = 0; t < numTestCases; t++) {
                            const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                            const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                            stackVec[(sp - 1) * numTestCases + t] = (a && b) ? 0 : 1;
                        }
                    }
                    break;
                case OpCode.NOR:
                    if (sp > 1) {
                        sp--;
                        for (let t = 0; t < numTestCases; t++) {
                            const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                            const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                            stackVec[(sp - 1) * numTestCases + t] = (a || b) ? 0 : 1;
                        }
                    }
                    break;
                case OpCode.XOR:
                    if (sp > 1) {
                        sp--;
                        for (let t = 0; t < numTestCases; t++) {
                            const a = stackVec[(sp - 1) * numTestCases + t] > 0.5 ? 1 : 0;
                            const b = stackVec[sp * numTestCases + t] > 0.5 ? 1 : 0;
                            stackVec[(sp - 1) * numTestCases + t] = (a !== b) ? 1 : 0;
                        }
                    }
                    break;
                case OpCode.ITE:
                    if (sp > 2) {
                        sp -= 2;
                        for (let t = 0; t < numTestCases; t++) {
                            const cond = stackVec[(sp - 1) * numTestCases + t];
                            const whenTrue = stackVec[sp * numTestCases + t];
                            const whenFalse = stackVec[(sp + 1) * numTestCases + t];
                            stackVec[(sp - 1) * numTestCases + t] = cond > 0.5 ? whenTrue : whenFalse;
                        }
                    }
                    break;
            }
        }

        // Return index 0 of the stack as the final result vector
        for (let t = 0; t < numTestCases; t++) results[t] = stackVec[t];
        return results;
    }
}
