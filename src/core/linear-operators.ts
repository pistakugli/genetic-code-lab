import { OpCode, PopulationBuffer } from './pop-buffer';
import { random } from '../utils/helpers';

/**
 * Genetic Code Lab - Linear Operators (DOGP)
 * Structural genetic operators for postfix buffers.
 */
export class LinearOperators {

    private static ArityTable = new Uint8Array(256);
    static {
        this.ArityTable[OpCode.NONE] = 0;
        this.ArityTable[OpCode.ADD] = 2;
        this.ArityTable[OpCode.SUB] = 2;
        this.ArityTable[OpCode.MUL] = 2;
        this.ArityTable[OpCode.DIV] = 2;
        this.ArityTable[OpCode.SIN] = 1;
        this.ArityTable[OpCode.COS] = 1;
        this.ArityTable[OpCode.EXP] = 1;
        this.ArityTable[OpCode.LOG] = 1;
        this.ArityTable[OpCode.SQRT] = 1;
        this.ArityTable[OpCode.XN] = 0;
        this.ArityTable[OpCode.CONST] = 0;
        this.ArityTable[OpCode.TERMINATOR] = 0;
        this.ArityTable[OpCode.AND] = 2;
        this.ArityTable[OpCode.OR] = 2;
        this.ArityTable[OpCode.NAND] = 2;
        this.ArityTable[OpCode.NOR] = 2;
        this.ArityTable[OpCode.XOR] = 2;
        this.ArityTable[OpCode.ITE] = 3;
    }

    private static getThresholdUpperComposeStrategy(): 'adaptive' | 'coalition' | 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or' | 'paired-witness' {
        const raw = (process.env.GCL_THRESHOLD_UPPER_COMPOSE || '').trim().toLowerCase();
        if (raw === 'cross') return 'cross';
        if (raw === 'mirrored-cross' || raw === 'mirrored_cross' || raw === 'mirroredcross') return 'mirrored-cross';
        if (raw === 'triad-witness' || raw === 'triad_witness' || raw === 'triadwitness') return 'triad-witness';
        if (raw === 'coalition') return 'coalition';
        if (raw === 'paired-witness' || raw === 'paired_witness' || raw === 'pairedwitness') return 'paired-witness';
        if (raw === 'witnessed-or' || raw === 'witnessed_or' || raw === 'witnessedor') return 'witnessed-or';
        return 'adaptive';
    }

    /**
     * Finds the starting index and size of a subtree ending at 'endIdx'.
     */
    public static getSubtreeRange(buffer: PopulationBuffer, genomeIdx: number, endIdx: number): { start: number, size: number } {
        const baseOffset = buffer.offsets[genomeIdx];
        let neededNodes = 1;
        let p = endIdx;

        while (neededNodes > 0 && p >= 0) {
            const instr = buffer.geneBuffer[baseOffset + p];
            const op = (instr >>> 24) as OpCode;
            neededNodes += this.ArityTable[op] - 1;
            p--;
        }

        const start = p + 1;
        const size = endIdx - start + 1;
        return { start, size };
    }

    /**
     * Subtree Crossover (Cross-Buffer: offspring written to targetBuffer)
     */
    public static crossover(
        sourceBuffer: PopulationBuffer,
        parent1Idx: number,
        parent2Idx: number,
        targetBuffer: PopulationBuffer,
        offspringIdx: number,
        maxSize: number,
        quotientMode: boolean = false
    ): void {
        const len1 = sourceBuffer.lengths[parent1Idx];
        const len2 = sourceBuffer.lengths[parent2Idx];

        let p1, p2;

        if (quotientMode && len1 > 3 && len2 > 3) {
            // Strict Root Contract Crossover
            // Denominator ends at len - 2
            const denRange1 = this.getSubtreeRange(sourceBuffer, parent1Idx, len1 - 2);
            const denRange2 = this.getSubtreeRange(sourceBuffer, parent2Idx, len2 - 2);

            const isNum = random() < 0.5;
            if (isNum) {
                // Numerator spans [0, denRange.start - 1]
                const numEnd1 = denRange1.start - 1;
                const numEnd2 = denRange2.start - 1;
                p1 = numEnd1 > 0 ? Math.floor(random() * numEnd1) : 0;
                p2 = numEnd2 > 0 ? Math.floor(random() * numEnd2) : 0;
            } else {
                // Denominator spans [denRange.start, len - 2]
                p1 = denRange1.start + Math.floor(random() * denRange1.size);
                p2 = denRange2.start + Math.floor(random() * denRange2.size);
            }
        } else {
            p1 = Math.floor(random() * len1);
            p2 = Math.floor(random() * len2);
        }

        const range1 = this.getSubtreeRange(sourceBuffer, parent1Idx, p1);
        const range2 = this.getSubtreeRange(sourceBuffer, parent2Idx, p2);

        const newLen = len2 - range2.size + range1.size;

        if (newLen > maxSize) {
            targetBuffer.copyGenome(parent2Idx, offspringIdx, sourceBuffer);
            return;
        }

        const offOffset = targetBuffer.offsets[offspringIdx];
        const p1Offset = sourceBuffer.offsets[parent1Idx];
        const p2Offset = sourceBuffer.offsets[parent2Idx];

        // Constants per Genome
        const cPerG = sourceBuffer.constBuffer.length / sourceBuffer.lengths.length;
        const srcC1Base = parent1Idx * cPerG;
        const dstCBase = offspringIdx * cPerG;

        // 1. Copy parent2 head
        targetBuffer.geneBuffer.set(
            sourceBuffer.geneBuffer.subarray(p2Offset, p2Offset + range2.start),
            offOffset
        );

        // 2. Copy parent1 subtree AND Migrate Constants
        const subtreeStart = p1Offset + range1.start;
        for (let i = 0; i < range1.size; i++) {
            const instr = sourceBuffer.geneBuffer[subtreeStart + i];
            const op = (instr >>> 24) as OpCode;

            if (op === OpCode.CONST) {
                const localIdx = instr & 0xFFFFFF;
                // Migrate the constant value
                targetBuffer.constBuffer[dstCBase + localIdx] = sourceBuffer.constBuffer[srcC1Base + localIdx];
            }

            targetBuffer.geneBuffer[offOffset + range2.start + i] = instr;
        }

        // 3. Copy parent2 tail
        const tailSourceStart = p2Offset + range2.start + range2.size;
        const tailDestStart = offOffset + range2.start + range1.size;
        const tailLen = len2 - (range2.start + range2.size);

        if (tailLen > 0 && tailDestStart + tailLen <= offOffset + maxSize) {
            targetBuffer.geneBuffer.set(
                sourceBuffer.geneBuffer.subarray(tailSourceStart, p2Offset + len2),
                tailDestStart
            );
        }

        targetBuffer.lengths[offspringIdx] = newLen;
        targetBuffer.ancestorMap[offspringIdx] = (sourceBuffer.ancestorMap[parent1Idx] | sourceBuffer.ancestorMap[parent2Idx]);
        targetBuffer.generationMap[offspringIdx] = Math.max(
            sourceBuffer.generationMap[parent1Idx],
            sourceBuffer.generationMap[parent2Idx]
        );
    }

    private static containsConst(buffer: PopulationBuffer, genomeIdx: number): boolean {
        const len = buffer.lengths[genomeIdx];
        const baseOffset = buffer.offsets[genomeIdx];
        for (let i = 0; i < len; i++) {
            if ((buffer.geneBuffer[baseOffset + i] >>> 24) === OpCode.CONST) return true;
        }
        return false;
    }

    private static appendGenome(
        sourceBuffer: PopulationBuffer,
        parentIdx: number,
        targetBuffer: PopulationBuffer,
        offspringIdx: number,
        assembled: number[]
    ): void {
        const len = sourceBuffer.lengths[parentIdx];
        const baseOffset = sourceBuffer.offsets[parentIdx];
        const cPerG = sourceBuffer.constBuffer.length / sourceBuffer.lengths.length;
        const srcCBase = parentIdx * cPerG;
        const dstCBase = offspringIdx * cPerG;

        for (let i = 0; i < len; i++) {
            const instr = sourceBuffer.geneBuffer[baseOffset + i];
            const op = (instr >>> 24) as OpCode;
            if (op === OpCode.CONST) {
                const localIdx = instr & 0xFFFFFF;
                targetBuffer.constBuffer[dstCBase + localIdx] = sourceBuffer.constBuffer[srcCBase + localIdx];
            }
            assembled.push(instr);
        }
    }

    private static getVariableMask(
        sourceBuffer: PopulationBuffer,
        genomeIdx: number,
        varCount: number
    ): number {
        let mask = 0;
        const len = sourceBuffer.lengths[genomeIdx];
        const baseOffset = sourceBuffer.offsets[genomeIdx];
        for (let i = 0; i < len; i++) {
            const instr = sourceBuffer.geneBuffer[baseOffset + i];
            const op = (instr >>> 24) as OpCode;
            const data = instr & 0xFFFFFF;
            if (op === OpCode.XN && data < varCount) mask |= (1 << data);
        }
        return mask >>> 0;
    }

    private static pickBridgeVariables(maskA: number, maskB: number, varCount: number): [number, number, number] {
        const allMask = varCount >= 31 ? 0x7fffffff : ((1 << varCount) - 1);
        const combined = (maskA | maskB) & allMask;
        const missing = (~combined) & allMask;
        const frontier = (maskA ^ maskB) & allMask;
        const shared = (maskA & maskB) & allMask;
        const ordered: number[] = [];

        const pushBits = (mask: number) => {
            for (let bit = 0; bit < varCount; bit++) {
                if ((mask & (1 << bit)) !== 0) ordered.push(bit);
            }
        };

        pushBits(missing);
        pushBits(frontier);
        pushBits(shared);
        if (ordered.length === 0) {
            for (let bit = 0; bit < varCount; bit++) ordered.push(bit);
        }

        const left = ordered[0] ?? 0;
        const right = ordered[1] ?? left;
        const third = ordered[2] ?? right;
        return [left, right, third];
    }

    public static crossoverBooleanCompose(
        sourceBuffer: PopulationBuffer,
        parent1Idx: number,
        parent2Idx: number,
        targetBuffer: PopulationBuffer,
        offspringIdx: number,
        maxSize: number,
        varCount: number
    ): void {
        const assembled: number[] = [];
        const mergeRoll = random();

        if (mergeRoll < 0.25) {
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(random() < 0.55 ? OpCode.OR : OpCode.AND, 0));
        } else if (mergeRoll < 0.45) {
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.XOR, 0));
        } else if (mergeRoll < 0.7 && varCount > 0) {
            const selector = Math.floor(random() * varCount);
            assembled.push(PopulationBuffer.pack(OpCode.XN, selector));
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.ITE, 0));
        } else if (varCount > 1) {
            const leftVar = Math.floor(random() * varCount);
            const rightVar = (leftVar + 1 + Math.floor(random() * (varCount - 1))) % varCount;
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.XN, leftVar));
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.XN, rightVar));
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
            assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            assembled.push(PopulationBuffer.pack(OpCode.XN, leftVar));
            assembled.push(PopulationBuffer.pack(OpCode.XN, rightVar));
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
            assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
        } else {
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
        }

        if (assembled.length > maxSize) {
            targetBuffer.copyGenome(random() < 0.5 ? parent1Idx : parent2Idx, offspringIdx, sourceBuffer);
            return;
        }

        const offOffset = targetBuffer.offsets[offspringIdx];
        targetBuffer.geneBuffer.set(assembled, offOffset);
        targetBuffer.lengths[offspringIdx] = assembled.length;
        targetBuffer.ancestorMap[offspringIdx] = (sourceBuffer.ancestorMap[parent1Idx] | sourceBuffer.ancestorMap[parent2Idx]);
        targetBuffer.generationMap[offspringIdx] = Math.max(
            sourceBuffer.generationMap[parent1Idx],
            sourceBuffer.generationMap[parent2Idx]
        );
    }

    public static crossoverBooleanThresholdCompose(
        sourceBuffer: PopulationBuffer,
        parent1Idx: number,
        parent2Idx: number,
        targetBuffer: PopulationBuffer,
        offspringIdx: number,
        maxSize: number,
        varCount: number,
        focus: 'lower' | 'upper' | null = null,
        strategyOverride: 'coalition' | 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or' | 'paired-witness' | null = null
    ): void {
        const assembled: number[] = [];
        const maskA = this.getVariableMask(sourceBuffer, parent1Idx, varCount);
        const maskB = this.getVariableMask(sourceBuffer, parent2Idx, varCount);
        const [leftBridge, rightBridge, thirdBridge] = this.pickBridgeVariables(maskA, maskB, varCount);

        if (focus === 'upper' && varCount > 0) {
            const strategy = strategyOverride || this.getThresholdUpperComposeStrategy();

            if (strategy === 'cross') {
                // Cross-shoulder merge:
                // (A AND B) OR (A AND rightBridge) OR (B AND leftBridge) OR (leftBridge AND rightBridge)
                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            } else if (strategy === 'mirrored-cross') {
                // Mirrored cross-shoulder merge:
                // (A AND B) OR (A AND leftBridge) OR (B AND rightBridge) OR (leftBridge AND rightBridge)
                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            } else if (strategy === 'triad-witness') {
                // Triad-witness lift:
                // (A AND B) OR ((A OR B) AND atLeastTwo(leftBridge, rightBridge, thirdBridge))
                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));

                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.XN, thirdBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.XN, thirdBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            } else if (strategy === 'witnessed-or') {
                // Witnessed OR:
                // (A AND B) OR ((A OR B) AND (leftBridge OR rightBridge))
                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            } else if (strategy === 'paired-witness') {
                // Paired-witness lift:
                // (A AND B) OR ((A OR B) AND leftBridge AND rightBridge)
                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));
                assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            } else {
                const witness = random() < 0.5 ? leftBridge : rightBridge;

                // 2-of-3 coalition lift:
                // (A AND B) OR (A AND witness) OR (B AND witness)
                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.XN, witness));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
                assembled.push(PopulationBuffer.pack(OpCode.XN, witness));
                assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
                assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            }
        } else if ((focus === 'lower' || focus === null) && random() < 0.65 && varCount > 0) {
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

            assembled.push(PopulationBuffer.pack(OpCode.XN, leftBridge));
            assembled.push(PopulationBuffer.pack(OpCode.XN, rightBridge));
            assembled.push(PopulationBuffer.pack(OpCode.AND, 0));

            assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
            assembled.push(PopulationBuffer.pack(OpCode.OR, 0));
        } else {
            this.appendGenome(sourceBuffer, parent1Idx, targetBuffer, offspringIdx, assembled);
            this.appendGenome(sourceBuffer, parent2Idx, targetBuffer, offspringIdx, assembled);
            assembled.push(PopulationBuffer.pack(random() < 0.65 ? OpCode.OR : OpCode.AND, 0));
        }

        if (assembled.length > maxSize) {
            targetBuffer.copyGenome(random() < 0.5 ? parent1Idx : parent2Idx, offspringIdx, sourceBuffer);
            return;
        }

        const offOffset = targetBuffer.offsets[offspringIdx];
        targetBuffer.geneBuffer.set(assembled, offOffset);
        targetBuffer.lengths[offspringIdx] = assembled.length;
        targetBuffer.ancestorMap[offspringIdx] = (sourceBuffer.ancestorMap[parent1Idx] | sourceBuffer.ancestorMap[parent2Idx]);
        targetBuffer.generationMap[offspringIdx] = Math.max(
            sourceBuffer.generationMap[parent1Idx],
            sourceBuffer.generationMap[parent2Idx]
        );
    }

    /**
     * Adaptive Mutation: Changes multiple opcodes at random based on intensity.
     */
    public static mutateAdaptive(
        buffer: PopulationBuffer,
        genomeIdx: number,
        availableOps: OpCode[],
        intensity: number = 1,
        quotientMode: boolean = false
    ): void {
        const len = buffer.lengths[genomeIdx];
        const baseOffset = buffer.offsets[genomeIdx];

        const mutations = Math.max(1, Math.floor(intensity));
        for (let m = 0; m < mutations; m++) {
            let p;
            if (quotientMode && len > 3) {
                // Avoid mutating the final DIV opcode
                p = Math.floor(random() * (len - 1));
            } else {
                p = Math.floor(random() * len);
            }
            const oldInstr = buffer.geneBuffer[baseOffset + p];
            const oldOp = (oldInstr >>> 24) as OpCode;
            const oldArity = this.ArityTable[oldOp];

            const candidates = availableOps.filter(op => this.ArityTable[op] === oldArity && op !== oldOp);
            if (candidates.length > 0) {
                const nextOp = candidates[Math.floor(random() * candidates.length)];
                const data = oldInstr & 0xFFFFFF;
                buffer.geneBuffer[baseOffset + p] = PopulationBuffer.pack(nextOp, data);
            }
        }
    }

    /**
     * Subtree Macro-Mutation (Phase 47: Structural Surge)
     * Replaces a random node and its subtree with a new random unit.
     */
    public static mutateSubtree(
        buffer: PopulationBuffer,
        genomeIdx: number,
        availableOps: OpCode[],
        maxSize: number,
        varCount: number,
        maxDepth: number = 4,
        intensity: number = 1,
        quotientMode: boolean = false
    ): void {
        const len = buffer.lengths[genomeIdx];
        const baseOffset = buffer.offsets[genomeIdx];

        let p;
        if (quotientMode && len > 3) {
            // Avoid mutating the final DIV opcode
            p = Math.floor(random() * (len - 1));
        } else {
            p = Math.floor(random() * len);
        }
        
        const { start, size } = this.getSubtreeRange(buffer, genomeIdx, p);

        const newTree: number[] = [];
        const newConsts: number[] = [];
        const depthCap = quotientMode ? Math.min(Math.max(2, maxDepth), 4) : Math.max(2, maxDepth);
        const structuralDepth = Math.max(2, Math.min(depthCap, 2 + Math.floor(Math.max(0, intensity)) + (random() < 0.35 ? 1 : 0)));
        this.generateRandomSubtree(newTree, newConsts, availableOps, structuralDepth, varCount);

        const newLen = len - size + newTree.length;
        if (newLen > maxSize) return;

        if (newLen > len) {
            const tailSource = baseOffset + start + size;
            const tailDest = baseOffset + start + newTree.length;
            const tailLen = len - (start + size);
            if (tailLen > 0) {
                buffer.geneBuffer.copyWithin(tailDest, tailSource, tailSource + tailLen);
            }
        } else if (newLen < len) {
            // Gap closing
            const tailSource = baseOffset + start + size;
            const tailDest = baseOffset + start + newTree.length;
            const tailLen = len - (start + size);
            if (tailLen > 0) {
                buffer.geneBuffer.copyWithin(tailDest, tailSource, tailSource + tailLen);
            }
        }

        const cPerG = buffer.constBuffer.length / buffer.lengths.length;
        const cBase = genomeIdx * cPerG;
        for (let i = 0; i < newTree.length; i++) {
            const instr = newTree[i];
            const op = (instr >>> 24) as OpCode;
            if (op === OpCode.CONST) {
                const cIdx = instr & 0xFFFFFF;
                if (newConsts[i] !== undefined) {
                    buffer.constBuffer[cBase + cIdx] = newConsts[i];
                }
            }
            buffer.geneBuffer[baseOffset + start + i] = instr;
        }

        buffer.lengths[genomeIdx] = newLen;
    }

    private static generateRandomSubtree(
        tree: number[],
        consts: number[],
        availableOps: OpCode[],
        depth: number,
        varCount: number
    ): void {
        const isBooleanDomain = availableOps.length > 0
            && availableOps.every(op => op === OpCode.AND || op === OpCode.OR || op === OpCode.NAND || op === OpCode.NOR || op === OpCode.XOR || op === OpCode.ITE);
        const isMonotoneBooleanDomain = availableOps.length > 0
            && availableOps.every(op => op === OpCode.AND || op === OpCode.OR);
        if (depth <= 0 || random() < 0.3) {
            // TERMINAL
            const variableBias = isMonotoneBooleanDomain ? 1.0 : (isBooleanDomain ? 0.85 : 0.5);
            if (random() < variableBias) {
                const safeVarCount = Math.max(1, varCount);
                const varIdx = Math.floor(random() * safeVarCount);
                tree.push(PopulationBuffer.pack(OpCode.XN, varIdx));
                consts.push(0);
            } else {
                const val = isBooleanDomain ? (random() < 0.5 ? 0 : 1) : (random() * 20) - 10;
                const cIdx = tree.length; // use index as slot
                tree.push(PopulationBuffer.pack(OpCode.CONST, cIdx));
                consts.push(val);
            }
        } else {
            const ops = availableOps.filter(op => this.ArityTable[op] > 0);
            const op = ops.length > 0 ? ops[Math.floor(random() * ops.length)] : OpCode.ADD;
            const arity = this.ArityTable[op];

            for (let i = 0; i < arity; i++) {
                this.generateRandomSubtree(tree, consts, availableOps, depth - 1, varCount);
            }
            tree.push(PopulationBuffer.pack(op, 0));
            consts.push(0);
        }
    }
}
