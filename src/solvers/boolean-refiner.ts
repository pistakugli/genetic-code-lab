import { PopulationBuffer, OpCode } from '../core/pop-buffer';
import { BlockEvaluator } from '../core/block-evaluator';
import { SymbolicSimplifier } from '../core/simplifier';

type CandidateResult = {
    fitness: number;
    length: number;
    improved: boolean;
};

type BoundaryScore = {
    lowerError: number;
    upperError: number;
    strictError: number;
    shoulderError: number;
};

type BoundaryFocus = 'lower' | 'upper' | null;

type ThresholdBoundaryProfile = {
    threshold: number;
    onesPerCase: Uint8Array;
    strictBoundaryMask: Uint8Array;
    shoulderMask: Uint8Array;
};

type BoundaryDonor = {
    idx: number;
    mask: number;
    score: number;
    lowerGain: number;
    upperGain: number;
    strictGain: number;
    shoulderGain: number;
    worse: number;
    improvedCases: number[];
};

type BoundaryRepairCandidate = {
    opcodes: number[];
    estimatedFixes: number;
    cost: number;
    kind: 'core-lift' | 'positive-term' | 'negative-guard' | 'balanced-pair';
    key: string;
};

export class BooleanRefiner {
    private static tryReplaceGenome(
        buffer: PopulationBuffer,
        genomeIdx: number,
        candidate: number[],
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        fitnessMode: 'regression' | 'classification',
        classificationWeight: number,
        bestFitness: number,
        bestLength: number,
        boundaryProfile: ThresholdBoundaryProfile | null = null,
        allowBoundaryStep: boolean = false,
        simplifyCandidate: boolean = true,
        boundaryFocus: BoundaryFocus = null
    ): CandidateResult {
        const baseOffset = buffer.offsets[genomeIdx];
        const len = buffer.lengths[genomeIdx];
        const capacity = genomeIdx + 1 < buffer.offsets.length
            ? buffer.offsets[genomeIdx + 1] - baseOffset
            : buffer.geneBuffer.length - baseOffset;

        if (candidate.length > capacity) return { fitness: bestFitness, length: bestLength, improved: false };

        const previousGenes = buffer.geneBuffer.slice(baseOffset, baseOffset + len);
        const previousConstants = new Float64Array(buffer.getConstants(genomeIdx));
        const previousFitness = buffer.fitness[genomeIdx];
        const caseOffset = genomeIdx * buffer.maxTestCases;
        const previousCaseErrors = buffer.caseErrors.slice(caseOffset, caseOffset + buffer.maxTestCases);
        const previousBoundary = boundaryProfile
            ? this.measureBoundaryScore(buffer, genomeIdx, boundaryProfile)
            : null;

        buffer.geneBuffer.set(candidate, baseOffset);
        buffer.lengths[genomeIdx] = candidate.length;
        if (simplifyCandidate) {
            SymbolicSimplifier.simplify(buffer, genomeIdx, Array.from({ length: inputSize }, (_, i) => `b${i}`));
        }
        BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
        const candidateFitness = buffer.fitness[genomeIdx];
        const candidateLength = buffer.lengths[genomeIdx];
        const candidateBoundary = boundaryProfile
            ? this.measureBoundaryScore(buffer, genomeIdx, boundaryProfile)
            : null;

        if (
            this.shouldAcceptCandidate(candidateFitness, candidateLength, bestFitness, bestLength, inputSize)
            || (
                previousBoundary
                && candidateBoundary
                && this.shouldAcceptBoundaryStep(
                    candidateFitness,
                    candidateLength,
                    bestFitness,
                    bestLength,
                    inputSize,
                    previousBoundary,
                    candidateBoundary,
                    allowBoundaryStep
                )
            ) || (
                previousBoundary
                && candidateBoundary
                && this.shouldAcceptFocusedBoundaryRepair(
                    boundaryFocus,
                    candidateFitness,
                    candidateLength,
                    bestFitness,
                    bestLength,
                    previousBoundary,
                    candidateBoundary,
                    allowBoundaryStep
                )
            )
        ) {
            return { fitness: candidateFitness, length: candidateLength, improved: true };
        }

        buffer.geneBuffer.set(previousGenes, baseOffset);
        buffer.constBuffer.set(previousConstants, genomeIdx * previousConstants.length);
        buffer.lengths[genomeIdx] = len;
        buffer.fitness[genomeIdx] = previousFitness;
        buffer.caseErrors.set(previousCaseErrors, caseOffset);
        return { fitness: bestFitness, length: bestLength, improved: false };
    }

    private static tryAppendPattern(
        buffer: PopulationBuffer,
        genomeIdx: number,
        pattern: number[],
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        fitnessMode: 'regression' | 'classification',
        classificationWeight: number,
        bestFitness: number,
        bestLength: number
    ): CandidateResult {
        const baseOffset = buffer.offsets[genomeIdx];
        const len = buffer.lengths[genomeIdx];
        const capacity = genomeIdx + 1 < buffer.offsets.length
            ? buffer.offsets[genomeIdx + 1] - baseOffset
            : buffer.geneBuffer.length - baseOffset;

        if (len + pattern.length > capacity) return { fitness: bestFitness, length: bestLength, improved: false };

        const previousConstants = new Float64Array(buffer.getConstants(genomeIdx));
        const previousFitness = buffer.fitness[genomeIdx];
        const caseOffset = genomeIdx * buffer.maxTestCases;
        const previousCaseErrors = buffer.caseErrors.slice(caseOffset, caseOffset + buffer.maxTestCases);

        buffer.geneBuffer.set(pattern, baseOffset + len);
        buffer.lengths[genomeIdx] = len + pattern.length;
        SymbolicSimplifier.simplify(buffer, genomeIdx, Array.from({ length: inputSize }, (_, i) => `b${i}`));
        BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
        const candidateFitness = buffer.fitness[genomeIdx];
        const candidateLength = buffer.lengths[genomeIdx];

        if (this.shouldAcceptCandidate(candidateFitness, candidateLength, bestFitness, bestLength, inputSize)) {
            return { fitness: candidateFitness, length: candidateLength, improved: true };
        }

        buffer.lengths[genomeIdx] = len;
        buffer.constBuffer.set(previousConstants, genomeIdx * previousConstants.length);
        buffer.fitness[genomeIdx] = previousFitness;
        buffer.caseErrors.set(previousCaseErrors, caseOffset);
        return { fitness: bestFitness, length: bestLength, improved: false };
    }

    private static structuralScore(fitness: number, length: number, inputSize: number): number {
        const perInstructionCost = inputSize >= 9 ? 0.00012 : inputSize >= 7 ? 0.00009 : 0;
        const phaseMultiplier = fitness >= 0.95 ? 1.4 : fitness >= 0.85 ? 1.15 : 1.0;
        return fitness - (length * perInstructionCost * phaseMultiplier);
    }

    private static shouldAcceptCandidate(
        candidateFitness: number,
        candidateLength: number,
        bestFitness: number,
        bestLength: number,
        inputSize: number
    ): boolean {
        if (inputSize < 7) {
            return candidateFitness > bestFitness + 1e-6;
        }

        if (candidateFitness > bestFitness + 1e-6) {
            if (candidateLength <= bestLength) return true;
            const gain = candidateFitness - bestFitness;
            const addedInstructions = candidateLength - bestLength;
            const requiredGain = inputSize >= 9 ? addedInstructions * 0.00012 : inputSize >= 7 ? addedInstructions * 0.00009 : 0;
            if (gain + 1e-9 >= requiredGain) return true;
            return this.structuralScore(candidateFitness, candidateLength, inputSize) >
                this.structuralScore(bestFitness, bestLength, inputSize) + 1e-6;
        }

        if (Math.abs(candidateFitness - bestFitness) <= 1e-6 && candidateLength < bestLength) {
            return true;
        }

        return false;
    }

    private static measureBoundaryScore(
        buffer: PopulationBuffer,
        genomeIdx: number,
        profile: ThresholdBoundaryProfile
    ): BoundaryScore {
        const errBase = genomeIdx * buffer.maxTestCases;
        let lowerError = 0;
        let upperError = 0;
        let shoulderError = 0;

        for (let t = 0; t < profile.onesPerCase.length; t++) {
            const err = buffer.caseErrors[errBase + t];
            if (profile.strictBoundaryMask[t]) {
                if (profile.onesPerCase[t] < profile.threshold) lowerError += err;
                else upperError += err;
            } else if (profile.shoulderMask[t]) {
                shoulderError += err;
            }
        }

        return {
            lowerError,
            upperError,
            strictError: lowerError + upperError,
            shoulderError
        };
    }

    private static shouldAcceptBoundaryStep(
        candidateFitness: number,
        candidateLength: number,
        bestFitness: number,
        bestLength: number,
        inputSize: number,
        before: BoundaryScore,
        after: BoundaryScore,
        allowFitnessDrop: boolean
    ): boolean {
        if (inputSize < 9 || bestFitness < 0.9) return false;

        const strictGain = before.strictError - after.strictError;
        const lowerGain = before.lowerError - after.lowerError;
        const upperGain = before.upperError - after.upperError;
        const shoulderGain = before.shoulderError - after.shoulderError;
        const balanceBefore = Math.abs(before.lowerError - before.upperError);
        const balanceAfter = Math.abs(after.lowerError - after.upperError);
        const boundaryGain = strictGain + shoulderGain * 0.35 + (balanceBefore - balanceAfter) * 0.2;

        if (after.strictError > before.strictError + 1e-9) return false;
        if (strictGain <= 1e-9 && boundaryGain <= 0.75) return false;

        const lowerRegression = after.lowerError - before.lowerError;
        const upperRegression = after.upperError - before.upperError;
        const worstSideRegression = Math.max(lowerRegression, upperRegression);
        const bestSideGain = Math.max(lowerGain, upperGain);
        if (worstSideRegression > Math.max(2.0, bestSideGain * 0.45)) return false;

        const lengthGrowth = candidateLength - bestLength;
        if (lengthGrowth > 0 && boundaryGain < lengthGrowth * 0.012) return false;

        if (candidateFitness + 1e-9 >= bestFitness) return boundaryGain > 0.25;
        if (!allowFitnessDrop) return false;

        const toleratedDrop = Math.min(0.004, Math.max(0.0012, boundaryGain * 0.00045));
        return candidateFitness + toleratedDrop >= bestFitness
            && strictGain >= 2.0
            && boundaryGain >= 1.5;
    }

    private static getBoundaryFocus(boundary: BoundaryScore): BoundaryFocus {
        if (boundary.lowerError > boundary.upperError + 1e-9) return 'lower';
        if (boundary.upperError > boundary.lowerError + 1e-9) return 'upper';
        return null;
    }

    private static shouldAcceptFocusedBoundaryRepair(
        focus: BoundaryFocus,
        candidateFitness: number,
        candidateLength: number,
        bestFitness: number,
        bestLength: number,
        before: BoundaryScore,
        after: BoundaryScore,
        allowFitnessDrop: boolean
    ): boolean {
        if (!focus) return false;

        const targetBefore = focus === 'lower' ? before.lowerError : before.upperError;
        const targetAfter = focus === 'lower' ? after.lowerError : after.upperError;
        const offsideBefore = focus === 'lower' ? before.upperError : before.lowerError;
        const offsideAfter = focus === 'lower' ? after.upperError : after.lowerError;

        const targetGain = targetBefore - targetAfter;
        const offsideRegression = offsideAfter - offsideBefore;
        const strictGain = before.strictError - after.strictError;
        const shoulderGain = before.shoulderError - after.shoulderError;
        const lengthGrowth = candidateLength - bestLength;

        if (targetGain <= 0.5) return false;
        if (offsideRegression > Math.max(0.25, targetGain * 0.2)) return false;
        if (strictGain + shoulderGain * 0.2 <= 0.5) return false;
        if (lengthGrowth > 0 && targetGain < lengthGrowth * 0.03) return false;

        if (candidateFitness + 1e-9 >= bestFitness) return true;
        if (!allowFitnessDrop) return false;

        const toleratedDrop = Math.min(0.0025, Math.max(0.0008, targetGain * 0.00045));
        return candidateFitness + toleratedDrop >= bestFitness
            && offsideRegression <= 0.25;
    }

    private static getVariableMask(genome: Uint32Array, inputSize: number): number {
        let mask = 0;
        for (let i = 0; i < genome.length; i++) {
            const op = (genome[i] >>> 24) as OpCode;
            const data = genome[i] & 0xFFFFFF;
            if (op === OpCode.XN && data < inputSize) mask |= (1 << data);
        }
        return mask >>> 0;
    }

    private static popcount(mask: number): number {
        let count = 0;
        let value = mask >>> 0;
        while (value !== 0) {
            value &= value - 1;
            count++;
        }
        return count;
    }

    private static getCaseMask(testX: Float64Array, caseIdx: number, inputSize: number): number {
        let mask = 0;
        const offset = caseIdx * inputSize;
        for (let i = 0; i < inputSize; i++) {
            if (testX[offset + i] > 0.5) mask |= (1 << i);
        }
        return mask >>> 0;
    }

    private static maskToVariables(mask: number, inputSize: number): number[] {
        const vars: number[] = [];
        for (let bit = 0; bit < inputSize; bit++) {
            if ((mask & (1 << bit)) !== 0) vars.push(bit);
        }
        return vars;
    }

    private static appendFoldedTerm(out: number[], vars: number[], op: OpCode.AND | OpCode.OR): void {
        if (vars.length === 0) return;
        for (const variable of vars) out.push(PopulationBuffer.pack(OpCode.XN, variable));
        for (let i = 1; i < vars.length; i++) out.push(PopulationBuffer.pack(op, 0));
    }

    private static buildBoundaryCoreLift(currentGenome: Uint32Array, coreMask: number, inputSize: number): number[] | null {
        const limitMask = inputSize >= 31 ? 0x7fffffff : ((1 << inputSize) - 1);
        const outsideMask = (~coreMask) & limitMask;
        const coreVars = this.maskToVariables(coreMask, inputSize);
        const outsideVars = this.maskToVariables(outsideMask, inputSize);
        if (coreVars.length === 0 || outsideVars.length === 0) return null;

        const candidate = Array.from(currentGenome);
        this.appendFoldedTerm(candidate, coreVars, OpCode.AND);
        candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
        this.appendFoldedTerm(candidate, outsideVars, OpCode.OR);
        candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
        return candidate;
    }

    private static buildPositiveBoundaryTerm(currentGenome: Uint32Array, activeMask: number, inputSize: number): number[] | null {
        const activeVars = this.maskToVariables(activeMask, inputSize);
        if (activeVars.length === 0) return null;

        const candidate = Array.from(currentGenome);
        this.appendFoldedTerm(candidate, activeVars, OpCode.AND);
        candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
        return candidate;
    }

    private static buildNegativeBoundaryGuard(currentGenome: Uint32Array, inactiveMask: number, inputSize: number): number[] | null {
        const inactiveVars = this.maskToVariables(inactiveMask, inputSize);
        if (inactiveVars.length === 0) return null;

        const candidate = Array.from(currentGenome);
        this.appendFoldedTerm(candidate, inactiveVars, OpCode.OR);
        candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
        return candidate;
    }

    private static buildBalancedBoundaryPair(
        currentGenome: Uint32Array,
        primaryMask: number,
        secondaryMask: number,
        primaryKind: 'negative-guard' | 'positive-term',
        inputSize: number
    ): number[] | null {
        const candidate = Array.from(currentGenome);

        const appendPrimary = primaryKind === 'negative-guard'
            ? this.appendNegativeBoundaryGuard(candidate, primaryMask, inputSize)
            : this.appendPositiveBoundaryTerm(candidate, primaryMask, inputSize);
        if (!appendPrimary) return null;

        const appendSecondary = primaryKind === 'negative-guard'
            ? this.appendPositiveBoundaryTerm(candidate, secondaryMask, inputSize)
            : this.appendNegativeBoundaryGuard(candidate, secondaryMask, inputSize);
        if (!appendSecondary) return null;

        return candidate;
    }

    private static appendNegativeBoundaryGuard(candidate: number[], inactiveMask: number, inputSize: number): boolean {
        const inactiveVars = this.maskToVariables(inactiveMask, inputSize);
        if (inactiveVars.length === 0) return false;
        this.appendFoldedTerm(candidate, inactiveVars, OpCode.OR);
        candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
        return true;
    }

    private static appendPositiveBoundaryTerm(candidate: number[], activeMask: number, inputSize: number): boolean {
        const activeVars = this.maskToVariables(activeMask, inputSize);
        if (activeVars.length === 0) return false;
        this.appendFoldedTerm(candidate, activeVars, OpCode.AND);
        candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
        return true;
    }

    private static buildThresholdBoundaryProfile(
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number
    ): ThresholdBoundaryProfile | null {
        const numCases = testY.length;
        if (numCases === 0 || testX.length !== numCases * inputSize) return null;

        const totals = new Uint16Array(inputSize + 1);
        const positives = new Uint16Array(inputSize + 1);
        const onesPerCase = new Uint8Array(numCases);

        for (let t = 0; t < numCases; t++) {
            let ones = 0;
            const offset = t * inputSize;
            for (let i = 0; i < inputSize; i++) {
                if (testX[offset + i] > 0.5) ones++;
            }
            onesPerCase[t] = ones;
            totals[ones]++;
            if (testY[t] > 0.5) positives[ones]++;
        }

        let threshold = -1;
        for (let ones = 0; ones <= inputSize; ones++) {
            if (totals[ones] === 0) continue;
            if (positives[ones] * 2 >= totals[ones]) {
                threshold = ones;
                break;
            }
        }
        if (threshold <= 0 || threshold > inputSize) return null;

        const strictBoundaryMask = new Uint8Array(numCases);
        const shoulderMask = new Uint8Array(numCases);
        for (let t = 0; t < numCases; t++) {
            const ones = onesPerCase[t];
            if (ones === threshold || ones === threshold - 1) {
                strictBoundaryMask[t] = 1;
            } else if (ones === threshold + 1 || ones === threshold - 2) {
                shoulderMask[t] = 1;
            }
        }

        return { threshold, onesPerCase, strictBoundaryMask, shoulderMask };
    }

    private static estimateCoreLiftFixes(
        buffer: PopulationBuffer,
        genomeIdx: number,
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        coreMask: number
    ): number {
        const errBase = genomeIdx * buffer.maxTestCases;
        let fixes = 0;
        for (let t = 0; t < testY.length; t++) {
            if (buffer.caseErrors[errBase + t] <= 0.5) continue;

            const caseMask = this.getCaseMask(testX, t, inputSize);
            if (testY[t] > 0.5) {
                if ((caseMask & coreMask) === coreMask && (caseMask & ~coreMask) !== 0) fixes++;
            } else if ((caseMask & ~coreMask) === 0) {
                fixes++;
            }
        }
        return fixes;
    }

    private static estimatePositiveTermFixes(
        buffer: PopulationBuffer,
        genomeIdx: number,
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        activeMask: number
    ): number {
        const errBase = genomeIdx * buffer.maxTestCases;
        let fixes = 0;
        for (let t = 0; t < testY.length; t++) {
            if (testY[t] <= 0.5 || buffer.caseErrors[errBase + t] <= 0.5) continue;
            const caseMask = this.getCaseMask(testX, t, inputSize);
            if ((caseMask & activeMask) === activeMask) fixes++;
        }
        return fixes;
    }

    private static estimateNegativeGuardFixes(
        buffer: PopulationBuffer,
        genomeIdx: number,
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        inactiveMask: number
    ): number {
        const errBase = genomeIdx * buffer.maxTestCases;
        let fixes = 0;
        for (let t = 0; t < testY.length; t++) {
            if (testY[t] > 0.5 || buffer.caseErrors[errBase + t] <= 0.5) continue;
            const caseMask = this.getCaseMask(testX, t, inputSize);
            if ((caseMask & inactiveMask) === 0) fixes++;
        }
        return fixes;
    }

    private static collectStrictBoundaryMasks(
        buffer: PopulationBuffer,
        genomeIdx: number,
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        profile: ThresholdBoundaryProfile
    ): {
        positiveMasks: Set<number>;
        negativeGuardMasks: Set<number>;
        strictErrorCount: number;
    } {
        const errBase = genomeIdx * buffer.maxTestCases;
        const limitMask = inputSize >= 31 ? 0x7fffffff : ((1 << inputSize) - 1);
        const positiveMasks = new Set<number>();
        const negativeGuardMasks = new Set<number>();
        let strictErrorCount = 0;

        for (let t = 0; t < testY.length; t++) {
            if (!profile.strictBoundaryMask[t] || buffer.caseErrors[errBase + t] <= 0.5) continue;
            strictErrorCount++;

            const caseMask = this.getCaseMask(testX, t, inputSize);
            if (testY[t] > 0.5) {
                positiveMasks.add(caseMask);
            } else {
                negativeGuardMasks.add(((~caseMask) & limitMask) >>> 0);
            }
        }

        return { positiveMasks, negativeGuardMasks, strictErrorCount };
    }

    private static buildExactBoundaryCourtCandidate(
        currentGenome: Uint32Array,
        positiveMasks: Iterable<number>,
        negativeGuardMasks: Iterable<number>,
        inputSize: number
    ): number[] | null {
        const candidate = Array.from(currentGenome);
        let appended = 0;

        for (const inactiveMask of negativeGuardMasks) {
            if (this.appendNegativeBoundaryGuard(candidate, inactiveMask, inputSize)) appended++;
        }

        for (const activeMask of positiveMasks) {
            if (this.appendPositiveBoundaryTerm(candidate, activeMask, inputSize)) appended++;
        }

        return appended > 0 ? candidate : null;
    }

    private static buildBoundaryRepairCandidates(
        buffer: PopulationBuffer,
        genomeIdx: number,
        currentGenome: Uint32Array,
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        profile: ThresholdBoundaryProfile,
        currentBoundary: BoundaryScore
    ): BoundaryRepairCandidate[] {
        const errBase = genomeIdx * buffer.maxTestCases;
        const limitMask = inputSize >= 31 ? 0x7fffffff : ((1 << inputSize) - 1);
        const coreMasks = new Set<number>();
        const positiveMasks = new Set<number>();
        const negativeGuardMasks = new Set<number>();
        const positiveFixMap = new Map<number, number>();
        const negativeFixMap = new Map<number, number>();

        for (let t = 0; t < testY.length; t++) {
            if (!profile.strictBoundaryMask[t] || buffer.caseErrors[errBase + t] <= 0.5) continue;

            const caseMask = this.getCaseMask(testX, t, inputSize);
            const ones = profile.onesPerCase[t];
            if (testY[t] > 0.5) {
                positiveMasks.add(caseMask);

                if (ones === profile.threshold) {
                    for (let bit = 0; bit < inputSize; bit++) {
                        if ((caseMask & (1 << bit)) !== 0) {
                            const coreMask = caseMask & ~(1 << bit);
                            if (this.popcount(coreMask) === profile.threshold - 1) coreMasks.add(coreMask >>> 0);
                        }
                    }
                }
            } else {
                negativeGuardMasks.add(((~caseMask) & limitMask) >>> 0);
                if (ones === profile.threshold - 1) coreMasks.add(caseMask);
            }
        }

        const candidates: BoundaryRepairCandidate[] = [];
        const seen = new Set<string>();
        const addCandidate = (candidate: BoundaryRepairCandidate | null) => {
            if (!candidate || candidate.estimatedFixes <= 0 || seen.has(candidate.key)) return;
            seen.add(candidate.key);
            candidates.push(candidate);
        };

        for (const coreMask of coreMasks) {
            const opcodes = this.buildBoundaryCoreLift(currentGenome, coreMask, inputSize);
            if (!opcodes) continue;
            addCandidate({
                opcodes,
                estimatedFixes: this.estimateCoreLiftFixes(buffer, genomeIdx, testX, testY, inputSize, coreMask),
                cost: opcodes.length - currentGenome.length,
                kind: 'core-lift',
                key: `core:${coreMask}`
            });
        }

        for (const activeMask of positiveMasks) {
            const estimatedFixes = this.estimatePositiveTermFixes(buffer, genomeIdx, testX, testY, inputSize, activeMask);
            positiveFixMap.set(activeMask, estimatedFixes);
            const opcodes = this.buildPositiveBoundaryTerm(currentGenome, activeMask, inputSize);
            if (!opcodes) continue;
            addCandidate({
                opcodes,
                estimatedFixes,
                cost: opcodes.length - currentGenome.length,
                kind: 'positive-term',
                key: `pos:${activeMask}`
            });
        }

        for (const inactiveMask of negativeGuardMasks) {
            const estimatedFixes = this.estimateNegativeGuardFixes(buffer, genomeIdx, testX, testY, inputSize, inactiveMask);
            negativeFixMap.set(inactiveMask, estimatedFixes);
            const opcodes = this.buildNegativeBoundaryGuard(currentGenome, inactiveMask, inputSize);
            if (!opcodes) continue;
            addCandidate({
                opcodes,
                estimatedFixes,
                cost: opcodes.length - currentGenome.length,
                kind: 'negative-guard',
                key: `neg:${inactiveMask}`
            });
        }

        const lowerDominant = currentBoundary.lowerError > currentBoundary.upperError + 1e-9;
        const upperDominant = currentBoundary.upperError > currentBoundary.lowerError + 1e-9;
        const sideDelta = Math.abs(currentBoundary.lowerError - currentBoundary.upperError);
        const sortedPositiveMasks = Array.from(positiveMasks).sort((a, b) => (positiveFixMap.get(b) || 0) - (positiveFixMap.get(a) || 0));
        const sortedNegativeMasks = Array.from(negativeGuardMasks).sort((a, b) => (negativeFixMap.get(b) || 0) - (negativeFixMap.get(a) || 0));

        const emitBalancedPairs = (
            primaryKind: 'negative-guard' | 'positive-term',
            primaryMasks: number[],
            secondaryMasks: number[],
            primaryFixes: Map<number, number>,
            secondaryFixes: Map<number, number>
        ) => {
            const primaryLimit = inputSize >= 9 ? 4 : 2;
            const secondaryLimit = inputSize >= 9 ? 3 : 2;

            for (const primaryMask of primaryMasks.slice(0, primaryLimit)) {
                for (const secondaryMask of secondaryMasks.slice(0, secondaryLimit)) {
                    const opcodes = this.buildBalancedBoundaryPair(
                        currentGenome,
                        primaryMask,
                        secondaryMask,
                        primaryKind,
                        inputSize
                    );
                    if (!opcodes) continue;

                    const primaryEstimate = primaryFixes.get(primaryMask) || 0;
                    const secondaryEstimate = secondaryFixes.get(secondaryMask) || 0;
                    const combinedEstimate =
                        primaryEstimate
                        + (secondaryEstimate * 0.75)
                        + (Math.min(primaryEstimate, secondaryEstimate) * 0.35);

                    addCandidate({
                        opcodes,
                        estimatedFixes: combinedEstimate,
                        cost: opcodes.length - currentGenome.length,
                        kind: 'balanced-pair',
                        key: `pair:${primaryKind}:${primaryMask}:${secondaryMask}`
                    });
                }
            }
        };

        if (lowerDominant && sortedNegativeMasks.length > 0 && sortedPositiveMasks.length > 0) {
            emitBalancedPairs('negative-guard', sortedNegativeMasks, sortedPositiveMasks, negativeFixMap, positiveFixMap);
        } else if (upperDominant && sortedPositiveMasks.length > 0 && sortedNegativeMasks.length > 0) {
            emitBalancedPairs('positive-term', sortedPositiveMasks, sortedNegativeMasks, positiveFixMap, negativeFixMap);
        } else if (sortedPositiveMasks.length > 0 && sortedNegativeMasks.length > 0) {
            emitBalancedPairs('negative-guard', sortedNegativeMasks, sortedPositiveMasks, negativeFixMap, positiveFixMap);
            emitBalancedPairs('positive-term', sortedPositiveMasks, sortedNegativeMasks, positiveFixMap, negativeFixMap);
        }

        const kindBias = (kind: BoundaryRepairCandidate['kind']) => {
            if (kind === 'core-lift') return 0.35 + Math.min(0.45, sideDelta * 0.04);
            if (kind === 'negative-guard' && lowerDominant) return 0.45 + Math.min(0.6, sideDelta * 0.08);
            if (kind === 'positive-term' && upperDominant) return 0.45 + Math.min(0.6, sideDelta * 0.08);
            if (kind === 'balanced-pair') return 0.55 + Math.min(0.75, sideDelta * 0.1);
            return 0;
        };
        return candidates
            .sort((a, b) => {
                const aScore = (a.estimatedFixes * 10) - (a.cost * 0.16) + kindBias(a.kind);
                const bScore = (b.estimatedFixes * 10) - (b.cost * 0.16) + kindBias(b.kind);
                return bScore - aScore;
            })
            .slice(0, 16);
    }

    private static tryBoundaryCounterexampleRepair(
        buffer: PopulationBuffer,
        genomeIdx: number,
        currentGenome: Uint32Array,
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        fitnessMode: 'regression' | 'classification',
        classificationWeight: number,
        bestFitness: number,
        bestLength: number,
        profile: ThresholdBoundaryProfile
    ): CandidateResult {
        const currentBoundary = this.measureBoundaryScore(buffer, genomeIdx, profile);
        const boundaryFocus = this.getBoundaryFocus(currentBoundary);
        const strictMasks = this.collectStrictBoundaryMasks(
            buffer,
            genomeIdx,
            testX,
            testY,
            inputSize,
            profile
        );
        const courtWidth = strictMasks.positiveMasks.size + strictMasks.negativeGuardMasks.size;
        if (strictMasks.strictErrorCount > 0 && strictMasks.strictErrorCount <= 6 && courtWidth <= 6) {
            const exactCourt = this.buildExactBoundaryCourtCandidate(
                currentGenome,
                strictMasks.positiveMasks,
                strictMasks.negativeGuardMasks,
                inputSize
            );
            if (exactCourt) {
                const courtUpdate = this.tryReplaceGenome(
                    buffer,
                    genomeIdx,
                    exactCourt,
                    testX,
                    testY,
                    inputSize,
                    fitnessMode,
                    classificationWeight,
                    bestFitness,
                    bestLength,
                    profile,
                    bestFitness >= 0.955,
                    false,
                    boundaryFocus
                );
                if (courtUpdate.improved) {
                    return courtUpdate;
                }
            }
        }

        const candidates = this.buildBoundaryRepairCandidates(
            buffer,
            genomeIdx,
            currentGenome,
            testX,
            testY,
            inputSize,
            profile,
            currentBoundary
        );

        for (const candidate of candidates) {
            const updated = this.tryReplaceGenome(
                buffer,
                genomeIdx,
                candidate.opcodes,
                testX,
                testY,
                inputSize,
                fitnessMode,
                classificationWeight,
                bestFitness,
                bestLength,
                profile,
                bestFitness >= 0.955,
                false,
                boundaryFocus
            );
            if (updated.improved) {
                return updated;
            }
        }
        return { fitness: bestFitness, length: bestLength, improved: false };
    }

    private static rankGlobalDonors(
        buffer: PopulationBuffer,
        indices: number[],
        inputSize: number
    ): number[] {
        return indices
            .map(idx => {
                const genome = buffer.getOpcodes(idx);
                const mask = this.getVariableMask(genome, inputSize);
                const coverage = this.popcount(mask);
                const compactness = coverage / Math.max(1, buffer.lengths[idx]);
                const score = (buffer.fitness[idx] * 5.0) + (coverage * 0.45) + (compactness * 2.0) - (buffer.lengths[idx] * 0.01);
                return { idx, score };
            })
            .sort((a, b) => b.score - a.score)
            .slice(0, inputSize >= 9 ? 8 : 5)
            .map(entry => entry.idx);
    }

    private static rankCompatibleDonors(
        buffer: PopulationBuffer,
        genomeIdx: number,
        indices: number[],
        inputSize: number
    ): number[] {
        const currentGenome = buffer.getOpcodes(genomeIdx);
        const currentMask = this.getVariableMask(currentGenome, inputSize);
        const currentLength = buffer.lengths[genomeIdx];

        return indices
            .filter(idx => idx !== genomeIdx)
            .map(idx => {
                const donorGenome = buffer.getOpcodes(idx);
                const donorMask = this.getVariableMask(donorGenome, inputSize);
                const complement = this.popcount(donorMask & ~currentMask);
                const overlap = this.popcount(donorMask & currentMask);
                const union = this.popcount(donorMask | currentMask);
                const lengthPenalty = Math.abs(buffer.lengths[idx] - currentLength) * 0.015;
                const score = (buffer.fitness[idx] * 4.0) + (complement * 0.55) + (overlap * 0.2) + (union * 0.1) - lengthPenalty;
                return { idx, score };
            })
            .sort((a, b) => b.score - a.score)
            .slice(0, inputSize >= 9 ? 8 : 4)
            .map(entry => entry.idx);
    }

    private static rankBoundaryDonors(
        buffer: PopulationBuffer,
        genomeIdx: number,
        indices: number[],
        inputSize: number,
        profile: ThresholdBoundaryProfile
    ): BoundaryDonor[] {
        const currentGenome = buffer.getOpcodes(genomeIdx);
        const currentMask = this.getVariableMask(currentGenome, inputSize);
        const currentLength = buffer.lengths[genomeIdx];
        const currentErrBase = genomeIdx * buffer.maxTestCases;
        const descriptors: BoundaryDonor[] = [];

        for (const idx of indices) {
            if (idx === genomeIdx) continue;

            const donorGenome = buffer.getOpcodes(idx);
            const donorMask = this.getVariableMask(donorGenome, inputSize);
            const donorErrBase = idx * buffer.maxTestCases;

            let lowerGain = 0;
            let upperGain = 0;
            let strictGain = 0;
            let shoulderGain = 0;
            let worse = 0;
            const improvedCases: number[] = [];

            for (let t = 0; t < profile.onesPerCase.length; t++) {
                const currentErr = buffer.caseErrors[currentErrBase + t];
                const donorErr = buffer.caseErrors[donorErrBase + t];
                const delta = currentErr - donorErr;

                if (profile.strictBoundaryMask[t]) {
                    if (delta > 1e-9) {
                        strictGain += delta;
                        improvedCases.push(t);
                        if (profile.onesPerCase[t] < profile.threshold) lowerGain += delta;
                        else upperGain += delta;
                    } else if (delta < -1e-9) {
                        worse += -delta;
                    }
                } else if (profile.shoulderMask[t]) {
                    if (delta > 1e-9) shoulderGain += delta;
                    else if (delta < -1e-9) worse += (-delta) * 0.35;
                }
            }

            const complement = this.popcount(donorMask & ~currentMask);
            const overlap = this.popcount(donorMask & currentMask);
            const union = this.popcount(donorMask | currentMask);
            const balance = Math.min(lowerGain, upperGain);
            const lengthPenalty = Math.abs(buffer.lengths[idx] - currentLength) * 0.015;
            const score =
                (buffer.fitness[idx] * 4.0)
                + (complement * 0.8)
                + (overlap * 0.1)
                + (union * 0.15)
                + (balance * 10.0)
                + (strictGain * 2.2)
                + (shoulderGain * 0.75)
                - (worse * 1.8)
                - lengthPenalty;

            descriptors.push({
                idx,
                mask: donorMask,
                score,
                lowerGain,
                upperGain,
                strictGain,
                shoulderGain,
                worse,
                improvedCases
            });
        }

        return descriptors
            .sort((a, b) => b.score - a.score)
            .slice(0, inputSize >= 9 ? 10 : 6);
    }

    private static scoreBoundaryCoalitionPair(
        currentMask: number,
        donorA: BoundaryDonor,
        donorB: BoundaryDonor,
        inputSize: number
    ): number {
        const unionCoverage = this.popcount(currentMask | donorA.mask | donorB.mask);
        const overlapPenalty = this.popcount(donorA.mask & donorB.mask);
        const improved = new Set<number>();
        for (const caseIdx of donorA.improvedCases) improved.add(caseIdx);
        let duplicateImprovement = 0;
        for (const caseIdx of donorB.improvedCases) {
            if (improved.has(caseIdx)) duplicateImprovement++;
            improved.add(caseIdx);
        }

        const strictGain = donorA.strictGain + donorB.strictGain;
        const lowerGain = donorA.lowerGain + donorB.lowerGain;
        const upperGain = donorA.upperGain + donorB.upperGain;
        const balance = Math.min(lowerGain, upperGain);

        return donorA.score
            + donorB.score
            + (unionCoverage * 0.7)
            + (strictGain * 1.5)
            + (balance * 7.0)
            + (improved.size * 0.18)
            - (duplicateImprovement * 0.14)
            - (overlapPenalty * (inputSize >= 9 ? 0.4 : 0.25))
            - ((donorA.worse + donorB.worse) * 0.6);
    }

    private static buildBridgePriority(currentMask: number, donorMask: number, inputSize: number): number[] {
        const limitMask = inputSize >= 31 ? 0x7fffffff : ((1 << inputSize) - 1);
        const missingMask = (~(currentMask | donorMask)) & limitMask;
        const frontierMask = (currentMask ^ donorMask) & limitMask;
        const sharedMask = (currentMask & donorMask) & limitMask;
        const ordered: number[] = [];

        const pushBits = (mask: number) => {
            for (let bit = 0; bit < inputSize; bit++) {
                if ((mask & (1 << bit)) !== 0) ordered.push(bit);
            }
        };

        pushBits(missingMask);
        pushBits(frontierMask);
        pushBits(sharedMask);

        if (ordered.length === 0) {
            for (let bit = 0; bit < inputSize; bit++) ordered.push(bit);
        }

        return ordered;
    }

    private static buildSelfMajorityLift(currentGenome: Uint32Array, leftVar: number, rightVar: number): number[] {
        const firstBranch = Array.from(currentGenome);
        const secondBranch = Array.from(currentGenome);
        return [
            ...firstBranch,
            PopulationBuffer.pack(OpCode.XN, leftVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...secondBranch,
            PopulationBuffer.pack(OpCode.XN, rightVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.XN, leftVar),
            PopulationBuffer.pack(OpCode.XN, rightVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildSelfThreshold2Of4(currentGenome: Uint32Array, a: number, b: number, c: number): number[] {
        const current = Array.from(currentGenome);
        return [
            ...current,
            PopulationBuffer.pack(OpCode.XN, a),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...current,
            PopulationBuffer.pack(OpCode.XN, b),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            ...current,
            PopulationBuffer.pack(OpCode.XN, c),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.XN, a),
            PopulationBuffer.pack(OpCode.XN, b),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.XN, a),
            PopulationBuffer.pack(OpCode.XN, c),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.XN, b),
            PopulationBuffer.pack(OpCode.XN, c),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildSelfThreshold3Of4(currentGenome: Uint32Array, a: number, b: number, c: number): number[] {
        const current = Array.from(currentGenome);
        return [
            ...current,
            PopulationBuffer.pack(OpCode.XN, a),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, b),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...current,
            PopulationBuffer.pack(OpCode.XN, a),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, c),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            ...current,
            PopulationBuffer.pack(OpCode.XN, b),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, c),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.XN, a),
            PopulationBuffer.pack(OpCode.XN, b),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, c),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildAppendMajority3(leftVar: number, midVar: number, rightVar: number): number[] {
        return [
            PopulationBuffer.pack(OpCode.XN, leftVar),
            PopulationBuffer.pack(OpCode.XN, midVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, leftVar),
            PopulationBuffer.pack(OpCode.XN, rightVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, midVar),
            PopulationBuffer.pack(OpCode.XN, rightVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildCoalitionMajorityLift(
        leftGenome: Uint32Array,
        rightGenome: Uint32Array,
        bridgeVar: number
    ): number[] {
        return [
            ...Array.from(leftGenome),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(leftGenome),
            PopulationBuffer.pack(OpCode.XN, bridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.XN, bridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildCoalitionMajority3(
        leftGenome: Uint32Array,
        middleGenome: Uint32Array,
        rightGenome: Uint32Array
    ): number[] {
        return [
            ...Array.from(leftGenome),
            ...Array.from(middleGenome),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(leftGenome),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(middleGenome),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildThresholdMerge2(
        leftGenome: Uint32Array,
        rightGenome: Uint32Array,
        leftBridgeVar: number,
        rightBridgeVar: number
    ): number[] {
        return [
            ...Array.from(leftGenome),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(leftGenome),
            PopulationBuffer.pack(OpCode.XN, leftBridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.XN, rightBridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, leftBridgeVar),
            PopulationBuffer.pack(OpCode.XN, rightBridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildThresholdMerge2Cross(
        leftGenome: Uint32Array,
        rightGenome: Uint32Array,
        leftBridgeVar: number,
        rightBridgeVar: number
    ): number[] {
        return [
            ...Array.from(leftGenome),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(leftGenome),
            PopulationBuffer.pack(OpCode.XN, rightBridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            ...Array.from(rightGenome),
            PopulationBuffer.pack(OpCode.XN, leftBridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.XN, leftBridgeVar),
            PopulationBuffer.pack(OpCode.XN, rightBridgeVar),
            PopulationBuffer.pack(OpCode.AND, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0),
            PopulationBuffer.pack(OpCode.OR, 0)
        ];
    }

    private static buildThresholdMerge3(
        leftGenome: Uint32Array,
        middleGenome: Uint32Array,
        rightGenome: Uint32Array
    ): number[] {
        return this.buildCoalitionMajority3(leftGenome, middleGenome, rightGenome);
    }

    public static refineDiscrete(
        buffer: PopulationBuffer,
        indices: number[],
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number,
        fitnessMode: 'regression' | 'classification' = 'classification',
        classificationWeight: number = 0.8,
        monotoneThreshold: boolean = false,
        _logicThreshold?: number
    ): void {
        const booleanOps = monotoneThreshold
            ? [OpCode.AND, OpCode.OR]
            : [OpCode.AND, OpCode.OR, OpCode.NAND, OpCode.NOR, OpCode.XOR];
        const thresholdBoundaryProfile = monotoneThreshold && inputSize >= 7
            ? this.buildThresholdBoundaryProfile(testX, testY, inputSize)
            : null;

        for (const genomeIdx of indices) {
            let bestFitness = buffer.fitness[genomeIdx];
            let bestLength = buffer.lengths[genomeIdx];
            if (bestFitness >= 1.0) continue;

            if (bestFitness === -Infinity || bestFitness === 0) {
                BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                bestFitness = buffer.fitness[genomeIdx];
                bestLength = buffer.lengths[genomeIdx];
            }

            let anyImprovement = true;
            let iter = 0;
            const MAX_ITERS = thresholdBoundaryProfile && inputSize >= 9 ? 22 : inputSize >= 9 ? 18 : inputSize >= 7 ? 14 : 10;
            
            // Hillclimbing loop
            while (anyImprovement && iter < MAX_ITERS && bestFitness < 1.0) {
                anyImprovement = false;
                iter++;
                const len = buffer.lengths[genomeIdx];
                const baseOffset = buffer.offsets[genomeIdx];

                // Try 1-flip neighborhood
                for (let i = 0; i < len; i++) {
                    const instr = buffer.geneBuffer[baseOffset + i];
                    const op = (instr >>> 24) as OpCode;
                    const data = instr & 0xFFFFFF;

                    // Flip Variable Indices
                    if (op === OpCode.XN) {
                        for (let v = 0; v < inputSize; v++) {
                            if (v === data) continue;
                            
                            buffer.geneBuffer[baseOffset + i] = PopulationBuffer.pack(OpCode.XN, v);
                            BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                            const newFit = buffer.fitness[genomeIdx];
                            
                            if (newFit > bestFitness) {
                                bestFitness = newFit;
                                anyImprovement = true;
                            } else {
                                // Revert
                                buffer.geneBuffer[baseOffset + i] = instr;
                                buffer.fitness[genomeIdx] = bestFitness; // Restore best fitness
                            }
                        }
                    } 
                    // Flip Boolean Operators
                    else if (booleanOps.includes(op)) {
                        for (const newOp of booleanOps) {
                            if (newOp === op) continue;
                            
                            buffer.geneBuffer[baseOffset + i] = PopulationBuffer.pack(newOp, data);
                            BlockEvaluator.evaluateBlock(buffer, genomeIdx, 1, testX, testY, inputSize, fitnessMode, classificationWeight);
                            const newFit = buffer.fitness[genomeIdx];
                            
                            if (newFit > bestFitness) {
                                bestFitness = newFit;
                                anyImprovement = true;
                            } else {
                                // Revert
                                buffer.geneBuffer[baseOffset + i] = instr;
                                buffer.fitness[genomeIdx] = bestFitness;
                            }
                        }
                    }
                }

                if (!monotoneThreshold && !anyImprovement && bestFitness < 1.0) {
                    for (let v = 0; v < inputSize; v++) {
                        const xorCandidate = [
                            PopulationBuffer.pack(OpCode.XN, v),
                            PopulationBuffer.pack(OpCode.XOR, 0)
                        ];
                        const updated = this.tryAppendPattern(
                            buffer,
                            genomeIdx,
                            xorCandidate,
                            testX,
                            testY,
                            inputSize,
                            fitnessMode,
                            classificationWeight,
                            bestFitness,
                            bestLength
                        );
                        if (updated.improved) {
                            bestFitness = updated.fitness;
                            bestLength = updated.length;
                            anyImprovement = true;
                            break;
                        }
                    }
                }

                if (!anyImprovement && bestFitness < 1.0) {
                    const currentGenome = buffer.geneBuffer.slice(baseOffset, baseOffset + buffer.lengths[genomeIdx]);
                    for (let a = 0; a < inputSize && !anyImprovement; a++) {
                        for (let b = a + 1; b < inputSize; b++) {
                            const liftedMajority = this.buildSelfMajorityLift(currentGenome, a, b);
                            const updated = this.tryReplaceGenome(
                                buffer,
                                genomeIdx,
                                liftedMajority,
                                testX,
                                testY,
                                inputSize,
                                fitnessMode,
                                classificationWeight,
                                bestFitness,
                                bestLength
                            );
                            if (updated.improved) {
                                bestFitness = updated.fitness;
                                bestLength = updated.length;
                                anyImprovement = true;
                                break;
                            }
                        }
                    }
                }

                if (!anyImprovement && bestFitness < 1.0 && bestFitness >= 0.94 && monotoneThreshold && thresholdBoundaryProfile && inputSize >= 9) {
                    const currentGenome = buffer.geneBuffer.slice(baseOffset, baseOffset + buffer.lengths[genomeIdx]);
                    const currentMask = this.getVariableMask(currentGenome, inputSize);
                    const bridgeVars = this.buildBridgePriority(currentMask, 0, inputSize);
                    const limit = Math.min(bridgeVars.length, bestFitness >= 0.965 ? 5 : 4);
                    const allowBoundaryStep = genomeIdx !== indices[0] && bestFitness >= 0.92;

                    for (let ai = 0; ai < limit - 2 && !anyImprovement; ai++) {
                        for (let bi = ai + 1; bi < limit - 1 && !anyImprovement; bi++) {
                            for (let ci = bi + 1; ci < limit; ci++) {
                                const a = bridgeVars[ai];
                                const b = bridgeVars[bi];
                                const c = bridgeVars[ci];
                                const candidates = [
                                    this.buildSelfThreshold2Of4(currentGenome, a, b, c),
                                    this.buildSelfThreshold3Of4(currentGenome, a, b, c)
                                ];

                                for (const candidate of candidates) {
                                    const updated = this.tryReplaceGenome(
                                        buffer,
                                        genomeIdx,
                                        candidate,
                                        testX,
                                        testY,
                                        inputSize,
                                        fitnessMode,
                                        classificationWeight,
                                        bestFitness,
                                        bestLength,
                                        thresholdBoundaryProfile,
                                        allowBoundaryStep
                                    );
                                    if (updated.improved) {
                                        bestFitness = updated.fitness;
                                        bestLength = updated.length;
                                        anyImprovement = true;
                                        break;
                                    }
                                }
                                if (anyImprovement) break;
                            }
                        }
                    }
                }

                if (!anyImprovement && bestFitness < 1.0 && inputSize >= 7) {
                    if (thresholdBoundaryProfile && inputSize >= 9 && bestFitness >= 0.94) {
                        const currentGenome = buffer.geneBuffer.slice(baseOffset, baseOffset + buffer.lengths[genomeIdx]);
                        const repaired = this.tryBoundaryCounterexampleRepair(
                            buffer,
                            genomeIdx,
                            currentGenome,
                            testX,
                            testY,
                            inputSize,
                            fitnessMode,
                            classificationWeight,
                            bestFitness,
                            bestLength,
                            thresholdBoundaryProfile
                        );
                        if (repaired.improved) {
                            bestFitness = repaired.fitness;
                            bestLength = repaired.length;
                            anyImprovement = true;
                        }
                    }
                }

                if (!anyImprovement && bestFitness < 1.0 && inputSize >= 7) {
                    const currentGenome = buffer.geneBuffer.slice(baseOffset, baseOffset + buffer.lengths[genomeIdx]);
                    const currentMask = this.getVariableMask(currentGenome, inputSize);
                    const boundaryDonors = thresholdBoundaryProfile
                        ? this.rankBoundaryDonors(buffer, genomeIdx, indices, inputSize, thresholdBoundaryProfile)
                        : [];
                    const donorPool = boundaryDonors.length > 0
                        ? boundaryDonors.map(donor => donor.idx)
                        : this.rankCompatibleDonors(buffer, genomeIdx, indices, inputSize);
                    for (const donorIdx of donorPool) {
                        const donorGenome = buffer.getOpcodes(donorIdx);
                        const donorMask = this.getVariableMask(donorGenome, inputSize);
                        const bridgeVars = this.buildBridgePriority(currentMask, donorMask, inputSize);

                        for (let bi = 0; bi < Math.min(bridgeVars.length, thresholdBoundaryProfile && inputSize >= 9 ? 8 : inputSize >= 9 ? 6 : 4) && !anyImprovement; bi++) {
                            const leftBridge = bridgeVars[bi];
                            for (let bj = bi; bj < Math.min(bridgeVars.length, thresholdBoundaryProfile && inputSize >= 9 ? 8 : inputSize >= 9 ? 6 : 4); bj++) {
                                const rightBridge = bridgeVars[bj];
                                const thresholdMerge = this.buildThresholdMerge2(currentGenome, donorGenome, leftBridge, rightBridge);
                                const thresholdUpdated = this.tryReplaceGenome(
                                    buffer,
                                    genomeIdx,
                                    thresholdMerge,
                                    testX,
                                    testY,
                                    inputSize,
                                    fitnessMode,
                                    classificationWeight,
                                    bestFitness,
                                    bestLength,
                                    thresholdBoundaryProfile,
                                    genomeIdx !== indices[0] && monotoneThreshold && inputSize >= 9 && bestFitness >= 0.92
                                );
                                if (thresholdUpdated.improved) {
                                    bestFitness = thresholdUpdated.fitness;
                                    bestLength = thresholdUpdated.length;
                                    anyImprovement = true;
                                    break;
                                }

                                if (leftBridge !== rightBridge) {
                                    const thresholdCross = this.buildThresholdMerge2Cross(currentGenome, donorGenome, leftBridge, rightBridge);
                                    const crossUpdated = this.tryReplaceGenome(
                                        buffer,
                                        genomeIdx,
                                        thresholdCross,
                                        testX,
                                        testY,
                                        inputSize,
                                        fitnessMode,
                                        classificationWeight,
                                        bestFitness,
                                        bestLength,
                                        thresholdBoundaryProfile,
                                        genomeIdx !== indices[0] && monotoneThreshold && inputSize >= 9 && bestFitness >= 0.92
                                    );
                                    if (crossUpdated.improved) {
                                        bestFitness = crossUpdated.fitness;
                                        bestLength = crossUpdated.length;
                                        anyImprovement = true;
                                        break;
                                    }
                                }
                            }
                        }

                        for (let bridgeVarIdx = 0; bridgeVarIdx < Math.min(bridgeVars.length, thresholdBoundaryProfile && inputSize >= 9 ? 7 : inputSize >= 9 ? 5 : 3) && !anyImprovement; bridgeVarIdx++) {
                            const bridgeVar = bridgeVars[bridgeVarIdx];
                            const coalitionCandidate = this.buildCoalitionMajorityLift(currentGenome, donorGenome, bridgeVar);
                            const updated = this.tryReplaceGenome(
                                buffer,
                                genomeIdx,
                                coalitionCandidate,
                                testX,
                                testY,
                                inputSize,
                                fitnessMode,
                                classificationWeight,
                                bestFitness,
                                bestLength
                            );
                            if (updated.improved) {
                                bestFitness = updated.fitness;
                                bestLength = updated.length;
                                anyImprovement = true;
                                break;
                            }
                        }
                        if (anyImprovement) break;
                    }
                }

                if (!anyImprovement && bestFitness < 1.0 && inputSize >= 9) {
                    const currentGenome = buffer.geneBuffer.slice(baseOffset, baseOffset + buffer.lengths[genomeIdx]);
                    const currentMask = this.getVariableMask(currentGenome, inputSize);
                    const boundaryDonors = thresholdBoundaryProfile
                        ? this.rankBoundaryDonors(buffer, genomeIdx, indices, inputSize, thresholdBoundaryProfile)
                        : [];
                    const donorPool = boundaryDonors.length > 1
                        ? boundaryDonors.map(donor => donor.idx)
                        : this.rankCompatibleDonors(buffer, genomeIdx, indices, inputSize);

                    const donorDescriptors: BoundaryDonor[] = boundaryDonors.length > 1
                        ? boundaryDonors
                        : donorPool.map(idx => ({
                            idx,
                            mask: this.getVariableMask(buffer.getOpcodes(idx), inputSize),
                            score: buffer.fitness[idx],
                            lowerGain: 0,
                            upperGain: 0,
                            strictGain: 0,
                            shoulderGain: 0,
                            worse: 0,
                            improvedCases: []
                        }));

                    const pairOrder: Array<[number, number, number]> = [];
                    for (let i = 0; i < donorDescriptors.length; i++) {
                        for (let j = i + 1; j < donorDescriptors.length; j++) {
                            const score = this.scoreBoundaryCoalitionPair(currentMask, donorDescriptors[i], donorDescriptors[j], inputSize);
                            pairOrder.push([i, j, score]);
                        }
                    }
                    pairOrder.sort((a, b) => b[2] - a[2]);

                    const pairsToTry = pairOrder.length > 0 ? pairOrder.slice(0, Math.min(pairOrder.length, thresholdBoundaryProfile ? 18 : 12)) : [];
                    for (const [i, j] of pairsToTry) {
                        if (anyImprovement) break;
                        const donorAGenome = buffer.getOpcodes(donorDescriptors[i].idx);
                        const donorBGenome = buffer.getOpcodes(donorDescriptors[j].idx);
                        const unionCoverage = this.popcount(currentMask | donorDescriptors[i].mask | donorDescriptors[j].mask);
                        const overlapPenalty = this.popcount(donorDescriptors[i].mask & donorDescriptors[j].mask);
                        if (unionCoverage < Math.min(inputSize, 7)) continue;
                        if (overlapPenalty > unionCoverage + 1) continue;

                        const coalition3Candidate = this.buildThresholdMerge3(currentGenome, donorAGenome, donorBGenome);
                        const updated = this.tryReplaceGenome(
                            buffer,
                            genomeIdx,
                            coalition3Candidate,
                            testX,
                            testY,
                            inputSize,
                            fitnessMode,
                            classificationWeight,
                            bestFitness,
                            bestLength
                        );
                        if (updated.improved) {
                            bestFitness = updated.fitness;
                            bestLength = updated.length;
                            anyImprovement = true;
                            break;
                        }

                    }

                    for (let i = 0; i < donorPool.length && !anyImprovement; i++) {
                        const donorAGenome = buffer.getOpcodes(donorPool[i]);
                        for (let j = i + 1; j < donorPool.length; j++) {
                            const donorBGenome = buffer.getOpcodes(donorPool[j]);
                            const coalition3Candidate = this.buildThresholdMerge3(currentGenome, donorAGenome, donorBGenome);
                            const updated = this.tryReplaceGenome(
                                buffer,
                                genomeIdx,
                                coalition3Candidate,
                                testX,
                                testY,
                                inputSize,
                                fitnessMode,
                                classificationWeight,
                                bestFitness,
                                bestLength
                            );
                            if (updated.improved) {
                                bestFitness = updated.fitness;
                                bestLength = updated.length;
                                anyImprovement = true;
                                break;
                            }
                        }
                    }
                }

                // Phase 2: k-of-n Multi-Donor Bounded Coalition (3-parent external merge)
                if (!anyImprovement && bestFitness < 1.0 && inputSize >= 9 && monotoneThreshold) {
                    const globalDonorPool = this.rankGlobalDonors(buffer, indices, inputSize);
                    for (let i = 0; i < globalDonorPool.length && !anyImprovement; i++) {
                        const donorAGenome = buffer.getOpcodes(globalDonorPool[i]);
                        const donorAMask = this.getVariableMask(donorAGenome, inputSize);
                        for (let j = i + 1; j < globalDonorPool.length && !anyImprovement; j++) {
                            const donorBGenome = buffer.getOpcodes(globalDonorPool[j]);
                            const donorBMask = this.getVariableMask(donorBGenome, inputSize);
                            for (let k = j + 1; k < globalDonorPool.length; k++) {
                                const donorCGenome = buffer.getOpcodes(globalDonorPool[k]);
                                const donorCMask = this.getVariableMask(donorCGenome, inputSize);
                                const unionCoverage = this.popcount(donorAMask | donorBMask | donorCMask);
                                const overlapPenalty =
                                    this.popcount(donorAMask & donorBMask)
                                    + this.popcount(donorAMask & donorCMask)
                                    + this.popcount(donorBMask & donorCMask);
                                
                                if (unionCoverage < Math.min(inputSize, 7)) continue;
                                if (overlapPenalty > unionCoverage + 2) continue;

                                const donorCoalition = this.buildThresholdMerge3(donorAGenome, donorBGenome, donorCGenome);
                                const donorUpdated = this.tryReplaceGenome(
                                    buffer,
                                    genomeIdx,
                                    donorCoalition,
                                    testX,
                                    testY,
                                    inputSize,
                                    fitnessMode,
                                    classificationWeight,
                                    bestFitness,
                                    bestLength
                                );
                                if (donorUpdated.improved) {
                                    bestFitness = donorUpdated.fitness;
                                    bestLength = donorUpdated.length;
                                    anyImprovement = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!anyImprovement && bestFitness < 1.0) {
                    for (let a = 0; a < inputSize && !anyImprovement; a++) {
                        for (let b = a + 1; b < inputSize && !anyImprovement; b++) {
                            for (let c = b + 1; c < inputSize; c++) {
                                const majority3Clause = this.buildAppendMajority3(a, b, c);
                                const updated = this.tryAppendPattern(
                                    buffer,
                                    genomeIdx,
                                    majority3Clause,
                                    testX,
                                    testY,
                                    inputSize,
                                    fitnessMode,
                                    classificationWeight,
                                    bestFitness,
                                    bestLength
                                );
                                if (updated.improved) {
                                    bestFitness = updated.fitness;
                                    bestLength = updated.length;
                                    anyImprovement = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!anyImprovement && bestFitness < 1.0) {
                    for (let a = 0; a < inputSize && !anyImprovement; a++) {
                        for (let b = a + 1; b < inputSize; b++) {
                            const majorityClause = [
                                PopulationBuffer.pack(OpCode.XN, a),
                                PopulationBuffer.pack(OpCode.XN, b),
                                PopulationBuffer.pack(OpCode.AND, 0),
                                PopulationBuffer.pack(OpCode.OR, 0)
                            ];
                            const updated = this.tryAppendPattern(
                                buffer,
                                genomeIdx,
                                majorityClause,
                                testX,
                                testY,
                                inputSize,
                                fitnessMode,
                                classificationWeight,
                                bestFitness,
                                bestLength
                            );
                            if (updated.improved) {
                                bestFitness = updated.fitness;
                                bestLength = updated.length;
                                anyImprovement = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}
