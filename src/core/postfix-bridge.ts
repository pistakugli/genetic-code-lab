import { Genome, Gene } from './types';
import { OpCode, PopulationBuffer } from './pop-buffer';

/**
 * Genetic Code Lab - Postfix Bridge (DOGP)
 * Converts between AST-based Genomes and Binary Postfix.
 */
export class PostfixBridge {

    private static OpMap: Record<string, OpCode> = {
        'add': OpCode.ADD,
        'sub': OpCode.SUB,
        'mul': OpCode.MUL,
        'div': OpCode.DIV,
        'sin': OpCode.SIN,
        'cos': OpCode.COS,
        'exp': OpCode.EXP,
        'log': OpCode.LOG,
        'sqrt': OpCode.SQRT,
        'x': OpCode.XN,
        'const': OpCode.CONST,
        'and': OpCode.AND,
        'or': OpCode.OR,
        'nand': OpCode.NAND,
        'nor': OpCode.NOR,
        'xor': OpCode.XOR,
        'if_then_else': OpCode.ITE
    };

    /**
     * Converts an AST Genome to its Postfix components.
     */
    public static bake(genome: Genome, variableMap: string[]): { opcodes: OpCode[], data: number[], constants: number[] } {
        const opcodes: OpCode[] = [];
        const data: number[] = [];
        const constants: number[] = [];

        // Identify the root (assuming tree structure stored in genes)
        // Usually, genes represent the flattened tree. We need to traverse recursively.
        // For simplicity, we'll traverse starting from the root gene.

        const traverse = (gene: Gene) => {
            if (gene.type === 'terminal') {
                if (gene.operation === 'const') {
                    opcodes.push(OpCode.CONST);
                    data.push(constants.length);
                    constants.push(Number(gene.operands[0]));
                } else {
                    const vIdx = variableMap.indexOf(gene.operation);
                    opcodes.push(OpCode.XN);
                    data.push(vIdx !== -1 ? vIdx : 0);
                }
                return;
            }

            // Function: Traverse children first (Postfix)
            for (const operand of gene.operands) {
                let child: Gene | undefined;
                if (typeof operand === 'string') {
                    child = genome.genes.find(g => g.id === operand);
                } else if (typeof operand === 'object' && operand !== null) {
                    child = operand as Gene;
                }
                if (child) traverse(child);
            }

            const opName = gene.operation.toLowerCase();
            const op = PostfixBridge.OpMap[opName];
            opcodes.push(op || OpCode.NONE);
            data.push(0);
        };

        const root = genome.genes[0]; // Convention: first gene is the root
        if (root) traverse(root);

        return { opcodes, data, constants };
    }

    /**
     * Reconstructs an AST Genome from Postfix components.
     */
    public static unbake(
        opcodes: OpCode[],
        data: number[],
        constants: number[],
        variableMap: string[]
    ): Genome {
        const genes: Gene[] = [];
        const stack: string[] = []; // stores gene IDs

        const getOpName = (op: OpCode): string => {
            return Object.keys(this.OpMap).find(key => this.OpMap[key] === op) || 'unknown';
        };

        for (let i = 0; i < opcodes.length; i++) {
            const op = opcodes[i];
            const d = data[i];
            const id = `g_${i}_${Math.random().toString(36).substr(2, 4)}`;

            if (op === OpCode.XN) {
                genes.push({
                    id,
                    type: 'terminal',
                    operation: variableMap[d] || 'x0',
                    operands: []
                });
                stack.push(id);
            } else if (op === OpCode.CONST) {
                genes.push({
                    id,
                    type: 'terminal',
                    operation: 'const',
                    operands: [constants[d]]
                });
                stack.push(id);
            } else {
                // Function
                const arity = 2;
                const isUnary = op === OpCode.SIN || op === OpCode.COS || op === OpCode.EXP || op === OpCode.LOG || op === OpCode.SQRT;
                const isTernary = op === OpCode.ITE;
                const actualArity = isTernary ? 3 : (isUnary ? 1 : arity);

                const operands: string[] = [];
                for (let j = 0; j < actualArity; j++) {
                    const operandId = stack.pop();
                    if (operandId) operands.unshift(operandId);
                }

                genes.push({
                    id,
                    type: 'function',
                    operation: getOpName(op),
                    operands: operands
                });
                stack.push(id);
            }
        }

        const rootId = stack.pop();
        const rootIdx = genes.findIndex(g => g.id === rootId);
        if (rootIdx !== -1 && rootIdx !== 0) {
            const root = genes.splice(rootIdx, 1)[0];
            genes.unshift(root);
        }

        return {
            id: `genome_${Math.random().toString(36).substr(2, 4)}`,
            genes,
            fitness: -Infinity,
            generation: 0,
            age: 0
        };
    }
}
