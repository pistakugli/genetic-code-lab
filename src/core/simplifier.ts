import { PopulationBuffer, OpCode } from './pop-buffer';
import { PostfixBridge } from './postfix-bridge';
import { Genome, Gene } from './types';

/**
 * 🚀 Genetic Code Lab - Advanced Symbolic Simplifier (Phase 68)
 * Implements recursive algebraic reduction to eliminate bloat and expose true formulas.
 */
export class SymbolicSimplifier {
    
    /**
     * Simplifies a genome in the buffer by converting to AST, 
     * applying recursive rules, and baking back.
     */
    public static simplify(
        buffer: PopulationBuffer,
        genomeIdx: number,
        variableMap: string[]
    ): void {
        // 1. Unbake from buffer to AST
        const binary = buffer.getOpcodes(genomeIdx);
        const opcodes: OpCode[] = [];
        const data: number[] = [];
        for (let i = 0; i < binary.length; i++) {
            opcodes.push((binary[i] >>> 24) as OpCode);
            data.push(binary[i] & 0xFFFFFF);
        }
        const constants = Array.from(buffer.getConstants(genomeIdx));
        
        const genome = PostfixBridge.unbake(opcodes, data, constants, variableMap);
        
        // 2. Recursive Simplification
        const initialSize = genome.genes.length;
        this.simplifyAST(genome);
        
        if (genome.genes.length < initialSize) {
            // 3. Bake back to buffer if simplified
            const baked = PostfixBridge.bake(genome, variableMap);
            const packedInstructions = new Uint32Array(baked.opcodes.length);
            for (let i = 0; i < baked.opcodes.length; i++) {
                packedInstructions[i] = (baked.opcodes[i] << 24) | (baked.data[i] & 0xFFFFFF);
            }
            buffer.writeGenome(genomeIdx, packedInstructions, baked.data, baked.constants);
        }
    }

    public static simplifyAST(genome: Genome, protectRoot: boolean = false): void {
        if (genome.genes.length === 0) return;
        
        let changed = true;
        while (changed) {
            const sizeBefore = genome.genes.length;
            const rootId = genome.genes[0].id;
            let newRootId;
            
            if (protectRoot) {
                const rootGene = genome.genes.find(g => g.id === rootId);
                if (rootGene && rootGene.type === 'function') {
                    rootGene.operands = (rootGene.operands as string[]).map(id => this.simplifyRecursive(genome, id));
                }
                newRootId = rootId;
            } else {
                newRootId = this.simplifyRecursive(genome, rootId);
            }
            
            // Cleanup unreachable genes
            this.garbageCollect(genome, newRootId);
            changed = genome.genes.length < sizeBefore;
        }
    }

    private static simplifyRecursive(genome: Genome, geneId: string): string {
        const gene = genome.genes.find(g => g.id === geneId);
        if (!gene || gene.type === 'terminal') return geneId;

        // Simplify children first
        gene.operands = (gene.operands as string[]).map(id => this.simplifyRecursive(genome, id));

        const op = gene.operation.toLowerCase();
        const leftId = gene.operands[0] as string;
        const rightId = gene.operands[1] as string;
        const thirdId = gene.operands[2] as string | undefined;
        
        const left = genome.genes.find(g => g.id === leftId);
        const right = rightId ? genome.genes.find(g => g.id === rightId) : null;
        const third = thirdId ? genome.genes.find(g => g.id === thirdId) : null;

        // --- ALGEBRAIC RULES ---
        
        // 1. Constant Folding
        if (left?.operation === 'const' && (!rightId || (right && right.operation === 'const')) && (!thirdId || (third && third.operation === 'const'))) {
            const v1 = left.operands[0] as number;
            const v2 = right ? (right.operands[0] as number) : 0;
            const v3 = third ? (third.operands[0] as number) : 0;
            let result = 0;
            switch(op) {
                case 'add': result = v1 + v2; break;
                case 'sub': result = v1 - v2; break;
                case 'mul': result = v1 * v2; break;
                case 'div': result = v2 === 0 ? 1 : v1 / v2; break;
                case 'sin': result = Math.sin(v1); break;
                case 'cos': result = Math.cos(v1); break;
                case 'and': result = (v1 > 0.5 && v2 > 0.5) ? 1 : 0; break;
                case 'or': result = (v1 > 0.5 || v2 > 0.5) ? 1 : 0; break;
                case 'nand': result = (v1 > 0.5 && v2 > 0.5) ? 0 : 1; break;
                case 'nor': result = (v1 > 0.5 || v2 > 0.5) ? 0 : 1; break;
                case 'xor': result = ((v1 > 0.5) !== (v2 > 0.5)) ? 1 : 0; break;
                case 'if_then_else': result = v1 > 0.5 ? v2 : v3; break;
            }
            return this.createConst(genome, result);
        }

        // 2. Identity Laws
        if (op === 'add') {
            if (this.isConst(left, 0)) return rightId;
            if (this.isConst(right, 0)) return leftId;
        }
        if (op === 'sub') {
            if (this.isConst(right, 0)) return leftId;
            if (this.isStructurallyEqual(genome, leftId, rightId)) return this.createConst(genome, 0);
        }
        if (op === 'mul') {
            if (this.isConst(left, 1)) return rightId;
            if (this.isConst(right, 1)) return leftId;
            if (this.isConst(left, 0) || this.isConst(right, 0)) return this.createConst(genome, 0);
        }
        if (op === 'div') {
            if (this.isConst(right, 1)) return leftId;
            if (this.isStructurallyEqual(genome, leftId, rightId)) return this.createConst(genome, 1);
        }
        if (op === 'and') {
            if (this.isBooleanConst(left, 0) || this.isBooleanConst(right, 0)) return this.createConst(genome, 0);
            if (this.isBooleanConst(left, 1)) return rightId;
            if (this.isBooleanConst(right, 1)) return leftId;
            if (this.isStructurallyEqual(genome, leftId, rightId)) return leftId;
            const absorbedLeft = this.applyAbsorption(genome, leftId, rightId, 'or');
            if (absorbedLeft) return absorbedLeft;
            const absorbedRight = this.applyAbsorption(genome, rightId, leftId, 'or');
            if (absorbedRight) return absorbedRight;
        }
        if (op === 'or') {
            if (this.isBooleanConst(left, 1) || this.isBooleanConst(right, 1)) return this.createConst(genome, 1);
            if (this.isBooleanConst(left, 0)) return rightId;
            if (this.isBooleanConst(right, 0)) return leftId;
            if (this.isStructurallyEqual(genome, leftId, rightId)) return leftId;
            const absorbedLeft = this.applyAbsorption(genome, leftId, rightId, 'and');
            if (absorbedLeft) return absorbedLeft;
            const absorbedRight = this.applyAbsorption(genome, rightId, leftId, 'and');
            if (absorbedRight) return absorbedRight;
        }
        if (op === 'xor') {
            if (this.isBooleanConst(left, 0)) return rightId;
            if (this.isBooleanConst(right, 0)) return leftId;
            if (this.isStructurallyEqual(genome, leftId, rightId)) return this.createConst(genome, 0);
        }
        if (op === 'if_then_else') {
            if (rightId && thirdId && this.isStructurallyEqual(genome, rightId, thirdId)) return rightId;
            if (this.isBooleanConst(left, 1)) return rightId;
            if (this.isBooleanConst(left, 0) && thirdId) return thirdId;
        }

        return geneId;
    }

    private static isStructurallyEqual(genome: Genome, id1: string | undefined | null, id2: string | undefined | null): boolean {
        if (!id1 || !id2) return false;
        if (id1 === id2) return true;
        const g1 = genome.genes.find(g => g.id === id1);
        const g2 = genome.genes.find(g => g.id === id2);
        if (!g1 || !g2) return false;
        
        if (g1.type !== g2.type || g1.operation !== g2.operation) return false;
        if (g1.operands.length !== g2.operands.length) return false;
        
        if (g1.type === 'terminal') {
            return g1.operands[0] === g2.operands[0];
        }
        
        for (let i = 0; i < g1.operands.length; i++) {
            if (!this.isStructurallyEqual(genome, g1.operands[i] as string, g2.operands[i] as string)) {
                // Check symmetric operators
                if (g1.operands.length === 2 && (g1.operation === 'add' || g1.operation === 'mul' || g1.operation === 'and' || g1.operation === 'or' || g1.operation === 'xor' || g1.operation === 'nand' || g1.operation === 'nor')) {
                     if (this.isStructurallyEqual(genome, g1.operands[0] as string, g2.operands[1] as string) &&
                         this.isStructurallyEqual(genome, g1.operands[1] as string, g2.operands[0] as string)) {
                         return true;
                     }
                }
                return false;
            }
        }
        return true;
    }

    private static isConst(gene: Gene | undefined | null, val: number): boolean {
        return !!gene && gene.operation === 'const' && Math.abs((gene.operands[0] as number) - val) < 1e-9;
    }

    private static isBooleanConst(gene: Gene | undefined | null, val: 0 | 1): boolean {
        if (!gene || gene.operation !== 'const') return false;
        const boolVal = (gene.operands[0] as number) > 0.5 ? 1 : 0;
        return boolVal === val;
    }

    private static applyAbsorption(
        genome: Genome,
        anchorId: string | undefined | null,
        candidateId: string | undefined | null,
        candidateOp: 'and' | 'or'
    ): string | null {
        if (!anchorId || !candidateId) return null;
        const candidate = genome.genes.find(g => g.id === candidateId);
        if (!candidate || candidate.type !== 'function' || candidate.operation !== candidateOp) return null;

        const left = candidate.operands[0] as string | undefined;
        const right = candidate.operands[1] as string | undefined;
        if (left && this.isStructurallyEqual(genome, anchorId, left)) return anchorId;
        if (right && this.isStructurallyEqual(genome, anchorId, right)) return anchorId;
        return null;
    }

    private static createConst(genome: Genome, val: number): string {
        const id = `c_${Math.random().toString(36).substr(2, 5)}`;
        genome.genes.push({
            id,
            type: 'terminal',
            operation: 'const',
            operands: [val]
        });
        return id;
    }

    private static garbageCollect(genome: Genome, rootId: string): void {
        const reachable = new Set<string>();
        const walk = (id: string) => {
            if (reachable.has(id)) return;
            reachable.add(id);
            const gene = genome.genes.find(g => g.id === id);
            if (gene && gene.type === 'function') {
                (gene.operands as string[]).forEach(walk);
            }
        };
        walk(rootId);
        genome.genes = genome.genes.filter(g => reachable.has(g.id));
        
        // Ensure root is at index 0
        const rootIdx = genome.genes.findIndex(g => g.id === rootId);
        if (rootIdx > 0) {
            const root = genome.genes.splice(rootIdx, 1)[0];
            genome.genes.unshift(root);
        } else if (rootIdx === -1) {
            // Should not happen if logic is correct
        }
    }
}
