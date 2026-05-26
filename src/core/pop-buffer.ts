/**
 * Genetic Code Lab - Population Buffer (DOGP)
 * Monolithic memory management for linearized genomes.
 */

import { random } from '../utils/helpers';

export enum OpCode {
    NONE = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    SIN = 5,
    COS = 6,
    EXP = 7,
    LOG = 8,
    SQRT = 9,
    XN = 10,     // Data: variable index
    CONST = 11,  // Data: constant index
    AND = 12,
    OR = 13,
    NAND = 14,
    NOR = 15,
    XOR = 16,
    ITE = 17,
    TERMINATOR = 255
}

export class PopulationBuffer {
    public geneBuffer: Uint32Array;
    public constBuffer: Float64Array;
    public offsets: Uint32Array;
    public lengths: Uint32Array;
    public fitness: Float64Array;
    public searchFitness: Float64Array;
    public spea2Score: Float64Array;
    public caseErrors: Float64Array;
    public caseWeights: Float64Array; // Phase 55: Importance weights
    public searchCaseWeights: Float64Array;
    public ancestorMap: Uint8Array;
    public generationMap: Uint32Array; // generation when genome was born/last modified

    public readonly maxGenomes: number;
    private maxNodesPerGenome: number;
    private maxConstants: number;
    public maxTestCases: number;
    public nicheMap: Uint32Array;

    constructor(
        maxGenomes: number,
        maxNodesPerGenome: number = 128,
        maxConstantsPerGenome: number = 10,
        maxTestCases: number = 100
    ) {
        this.maxGenomes = maxGenomes;
        this.maxNodesPerGenome = maxNodesPerGenome;
        this.maxConstants = maxGenomes * maxConstantsPerGenome;
        this.maxTestCases = maxTestCases;

        this.geneBuffer = new Uint32Array(maxGenomes * maxNodesPerGenome);
        this.constBuffer = new Float64Array(this.maxConstants);
        this.offsets = new Uint32Array(maxGenomes);
        for (let i = 0; i < maxGenomes; i++) {
            this.offsets[i] = i * maxNodesPerGenome;
        }
        this.lengths = new Uint32Array(maxGenomes);
        this.fitness = new Float64Array(maxGenomes).fill(-Infinity);
        this.searchFitness = new Float64Array(maxGenomes).fill(-Infinity);
        this.spea2Score = new Float64Array(maxGenomes).fill(Infinity);
        this.caseErrors = new Float64Array(maxGenomes * maxTestCases).fill(0);
        this.caseWeights = new Float64Array(maxTestCases).fill(1.0); // Default importance 1.0
        this.searchCaseWeights = new Float64Array(maxTestCases).fill(1.0);
        this.ancestorMap = new Uint8Array(maxGenomes).fill(0);
        this.generationMap = new Uint32Array(maxGenomes).fill(0);
        this.nicheMap = new Uint32Array(1024).fill(0);

        for (let i = 0; i < maxGenomes; i++) this.offsets[i] = i * maxNodesPerGenome;
    }

    public static pack(op: OpCode, data: number = 0): number {
        return (op << 24) | (data & 0xFFFFFF);
    }

    public static unpack(instruction: number): { op: OpCode, data: number } {
        return { op: (instruction >>> 24) as OpCode, data: instruction & 0xFFFFFF };
    }

    public randomize(
        varCount: number,
        availableOps: OpCode[],
        _minNodes: number = 3,
        maxNodes: number = 16,
        maxTreeDepth: number = 4,
        quotientMode: boolean = false
    ): void {
        const cPerG = 10;
        const maxDepth = quotientMode
            ? Math.min(Math.max(2, maxTreeDepth), 4)
            : Math.max(2, maxTreeDepth);

        for (let i = 0; i < this.maxGenomes; i++) {
            const genomeOffset = this.offsets[i];
            const cBase = i * cPerG;
            
            const method = i % 2 === 0 ? 'grow' : 'full';
            const depthSpread = Math.max(1, maxDepth - 1);
            const currentMaxDepth = Math.max(2, Math.min(maxDepth, 2 + (i % depthSpread)));
            
            let tree: number[] = [];
            let constants: number[] = [];
            let attempts = 0;
            
            do {
                tree = [];
                constants = [];
                if (quotientMode) {
                    const numTree: number[] = [];
                    const denTree: number[] = [];
                    this.buildRandomTreeRecursive(numTree, constants, method, currentMaxDepth, varCount, availableOps);
                    this.buildRandomTreeRecursive(denTree, constants, method, currentMaxDepth, varCount, availableOps);
                    tree = [...numTree, ...denTree, PopulationBuffer.pack(OpCode.DIV, 0)];
                } else {
                    this.buildRandomTreeRecursive(tree, constants, method, currentMaxDepth, varCount, availableOps);
                }
                attempts++;
            } while (tree.length > this.maxNodesPerGenome && attempts < 10);

            // Fallback terminal if we couldn't fit it into constraints
            if (tree.length > this.maxNodesPerGenome) {
                if (quotientMode) {
                    tree = [
                        PopulationBuffer.pack(OpCode.XN, Math.floor(random() * varCount)),
                        PopulationBuffer.pack(OpCode.CONST, 0),
                        PopulationBuffer.pack(OpCode.DIV, 0)
                    ];
                    constants = [1.0]; // Safe fallback denominator
                } else {
                    tree = [PopulationBuffer.pack(OpCode.XN, Math.floor(random() * varCount))];
                    constants = [];
                }
            }
            
            this.geneBuffer.set(tree, genomeOffset);
            this.lengths[i] = tree.length;
            
            // Fill constants
            for (let c = 0; c < constants.length && c < cPerG; c++) {
                this.constBuffer[cBase + c] = constants[c];
            }

            this.fitness[i] = -Infinity;
            this.searchFitness[i] = -Infinity;
            this.ancestorMap[i] = 0;
            this.generationMap[i] = 0;
        }
    }

    private isBooleanOperatorSet(availableOps: OpCode[]): boolean {
        if (availableOps.length === 0) return false;
        return availableOps.every(op => op === OpCode.AND || op === OpCode.OR || op === OpCode.NAND || op === OpCode.NOR || op === OpCode.XOR || op === OpCode.ITE);
    }

    private isMonotoneBooleanOperatorSet(availableOps: OpCode[]): boolean {
        if (availableOps.length === 0) return false;
        return availableOps.every(op => op === OpCode.AND || op === OpCode.OR);
    }

    private buildRandomTreeRecursive(
        tree: number[], 
        constants: number[], 
        method: 'grow' | 'full', 
        depth: number, 
        varCount: number, 
        availableOps: OpCode[]
    ): void {
        const ArityTable: Record<number, number> = {
            [OpCode.ADD]: 2, [OpCode.SUB]: 2, [OpCode.MUL]: 2, [OpCode.DIV]: 2,
            [OpCode.SIN]: 1, [OpCode.COS]: 1, [OpCode.EXP]: 1, [OpCode.LOG]: 1, [OpCode.SQRT]: 1,
            [OpCode.AND]: 2, [OpCode.OR]: 2, [OpCode.NAND]: 2, [OpCode.NOR]: 2, [OpCode.XOR]: 2, [OpCode.ITE]: 3,
            [OpCode.XN]: 0, [OpCode.CONST]: 0
        };

        const isTerminal = depth <= 0 || (method === 'grow' && random() < 0.3);
        const isBooleanDomain = this.isBooleanOperatorSet(availableOps);
        const isMonotoneBooleanDomain = this.isMonotoneBooleanOperatorSet(availableOps);

        if (isTerminal) {
            const variableBias = isMonotoneBooleanDomain ? 1.0 : (isBooleanDomain ? 0.85 : 0.7);
            if (random() < variableBias) {
                tree.push(PopulationBuffer.pack(OpCode.XN, Math.floor(random() * varCount)));
            } else {
                const val = isBooleanDomain ? (random() < 0.5 ? 0 : 1) : (random() * 20) - 10;
                tree.push(PopulationBuffer.pack(OpCode.CONST, constants.length % 10));
                constants.push(val);
            }
        } else {
            const ops = availableOps.filter(op => ArityTable[op] > 0);
            const op = ops[Math.floor(random() * ops.length)] || OpCode.ADD;
            const arity = ArityTable[op];

            for (let a = 0; a < arity; a++) {
                this.buildRandomTreeRecursive(tree, constants, method, depth - 1, varCount, availableOps);
            }
            tree.push(PopulationBuffer.pack(op, 0));
        }
    }

    public copyGenome(sourceIdx: number, targetIdx: number, sourceBuffer: PopulationBuffer = this): void {
        const sOff = sourceBuffer.offsets[sourceIdx];
        const tOff = this.offsets[targetIdx];
        const len = sourceBuffer.lengths[sourceIdx];
        this.geneBuffer.set(sourceBuffer.geneBuffer.subarray(sOff, sOff + len), tOff);
        this.lengths[targetIdx] = len;
        this.fitness[targetIdx] = sourceBuffer.fitness[sourceIdx];
        this.searchFitness[targetIdx] = sourceBuffer.searchFitness[sourceIdx];
        const cPerG = 10;
        this.constBuffer.set(sourceBuffer.constBuffer.subarray(sourceIdx * cPerG, (sourceIdx + 1) * cPerG), targetIdx * cPerG);
        this.ancestorMap[targetIdx] = sourceBuffer.ancestorMap[sourceIdx];
        this.generationMap[targetIdx] = sourceBuffer.generationMap[sourceIdx];
    }

    public writeGenome(targetIdx: number, opcodes: Uint32Array | number[], data: Uint32Array | number[], constants: Float64Array | number[]): void {
        const tOff = this.offsets[targetIdx];
        const opcodeCount = Math.min(opcodes.length, this.maxNodesPerGenome);
        this.geneBuffer
            .subarray(tOff, tOff + this.maxNodesPerGenome)
            .fill(PopulationBuffer.pack(OpCode.TERMINATOR, 0));
        this.geneBuffer.set(Array.from(opcodes).slice(0, opcodeCount), tOff);
        this.lengths[targetIdx] = opcodeCount;
        const cPerG = 10;
        const constOffset = targetIdx * cPerG;
        this.constBuffer.subarray(constOffset, constOffset + cPerG).fill(0);
        this.constBuffer.set(Array.from(constants).slice(0, cPerG), constOffset);
        this.fitness[targetIdx] = -Infinity;
        this.searchFitness[targetIdx] = -Infinity;
        this.ancestorMap[targetIdx] = 1;
    }

    public getOpcodes(idx: number): Uint32Array {
        return this.geneBuffer.subarray(this.offsets[idx], this.offsets[idx] + this.lengths[idx]);
    }

    public getConstants(idx: number): Float64Array {
        return this.constBuffer.subarray(idx * 10, (idx + 1) * 10);
    }

    public getMaxNodesPerGenome(): number {
        return this.maxNodesPerGenome;
    }

    public calculateEntropy(sampleSize: number = 2000): number {
        this.nicheMap.fill(0);
        const sample = Math.min(this.maxGenomes, sampleSize);
        const stride = Math.floor(this.maxGenomes / sample);
        for (let i = 0; i < sample; i++) {
            const idx = i * stride;
            const offset = this.offsets[idx];
            const len = this.lengths[idx];
            let hash = 2166136261;
            for (let n = 0; n < len; n++) { hash ^= this.geneBuffer[offset + n]; hash = Math.imul(hash, 0x01000193); }
            this.nicheMap[(hash >>> 0) % 1024]++;
        }
        let entropy = 0;
        for (let j = 0; j < 1024; j++) { if (this.nicheMap[j] > 0) { const p = this.nicheMap[j] / sample; entropy -= p * Math.log2(p); } }
        return entropy / 10;
    }
}
