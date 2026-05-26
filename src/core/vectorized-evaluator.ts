import { Genome, Gene, TestCase } from './types';

/**
 * 🚀 Genetic Code Lab - Vectorized Master JIT v2 (Structural Batching)
 * High-throughput infrastructure for "Beyond SOTA" scaling.
 * Groups population by AST structure to reuse JIT-compiled kernels.
 */
export class VectorizedEvaluator {
    private static kernelCache = new Map<string, Function>();

    /**
     * Evaluates a massive batch of genomes by grouping them by structural similarity.
     * This is the "Beyond SOTA" throughput engine.
     */
    public static async evaluateMassiveBatch(
        genomes: Genome[],
        testX: Float64Array,
        testY: Float64Array,
        inputSize: number
    ): Promise<number[]> {
        const numTests = testY.length;
        if (genomes.length === 0 || numTests === 0) return [];

        const results = new Array(genomes.length).fill(0);
        
        // 1. Group genomes by structural hash
        const groups = new Map<string, { genomes: Genome[], originalIndices: number[] }>();
        for (let i = 0; i < genomes.length; i++) {
            const hash = this.getStructuralHash(genomes[i]);
            if (!groups.has(hash)) {
                groups.set(hash, { genomes: [], originalIndices: [] });
            }
            groups.get(hash)!.genomes.push(genomes[i]);
            groups.get(hash)!.originalIndices.push(i);
        }

        // 2. Evaluate each structural group using a Vectorized Kernel
        for (const [hash, group] of groups.entries()) {
            let kernel = this.kernelCache.get(hash);
            
            // Extract all constants for this group into a single buffer
            const allConstants: number[] = [];
            let constsPerGenome = 0;
            
            for (const genome of group.genomes) {
                const c = this.extractConstants(genome);
                allConstants.push(...c);
                constsPerGenome = c.length; // Same for all in group
            }

            if (!kernel) {
                kernel = this.compileStructuralKernel(group.genomes[0], constsPerGenome);
                this.kernelCache.set(hash, kernel);
            }

            const cArray = new Float64Array(allConstants);
            const groupResults = new Float64Array(group.genomes.length);

            // Execute the Batch Kernel (One call for the whole group!)
            kernel(testX, testY, cArray, groupResults, numTests, inputSize, constsPerGenome);

            // Map results back to original indices
            for (let i = 0; i < group.originalIndices.length; i++) {
                results[group.originalIndices[i]] = groupResults[i];
            }
        }

        return results;
    }

    /**
     * Generates a stable hash for the AST structure (ignoring constant values).
     */
    public static getStructuralHash(genome: Genome): string {
        if (!genome.genes.length) return '';
        const geneMap = new Map<string, Gene>();
        genome.genes.forEach(g => geneMap.set(g.id, g));

        function buildHash(gene: Gene | undefined): string {
            if (!gene) return 'NULL';
            if (gene.type === 'terminal') {
                return gene.operation === 'const' ? '@c' : `@v:${gene.operation}`;
            }
            const args = gene.operands.map(id => {
                if (typeof id === 'string') return buildHash(geneMap.get(id));
                return '?';
            });
            return `${gene.operation}(${args.join(',')})`;
        }

        return buildHash(genome.genes[0]);
    }

    /**
     * Extracts constants in the exact order they appear in the AST.
     */
    private static extractConstants(genome: Genome): number[] {
        const geneMap = new Map<string, Gene>();
        genome.genes.forEach(g => geneMap.set(g.id, g));
        const constants: number[] = [];

        function walk(gene: Gene | undefined) {
            if (!gene) return;
            if (gene.type === 'terminal' && gene.operation === 'const') {
                constants.push(Number(gene.operands[0]));
            } else if (gene.operands) {
                gene.operands.forEach(id => {
                    if (typeof id === 'string') walk(geneMap.get(id));
                });
            }
        }
        walk(genome.genes[0]);
        return constants;
    }

    /**
     * Compiles a Multivariate Kernel for a specific structure.
     * Processes multiple individuals of the SAME structure in a single call.
     */
    private static compileStructuralKernel(template: Genome, constsPerGenome: number): Function {
        const geneMap = new Map<string, Gene>();
        template.genes.forEach(g => geneMap.set(g.id, g));
        let cIdx = 0;

        function buildCode(gene: Gene | undefined): string {
            if (!gene) return '0.0';
            if (gene.type === 'terminal') {
                if (gene.operation === 'const') {
                    const val = Number(gene.operands[0]);
                    const currentCIdx = cIdx++;
                    return `(constsPerGenome > 0 ? c[genome_const_offset + ${currentCIdx}] : ${val})`;
                } else {
                    const match = gene.operation.match(/\d+/);
                    const idx = match ? parseInt(match[0]) : 0;
                    return `testX[t * inputSize + ${idx}]`;
                }
            }
            const args = gene.operands.map(id => {
                if (typeof id === 'string') return buildCode(geneMap.get(id));
                return '0.0';
            });
            switch (gene.operation) {
                case 'add': return `(${args[0]} + ${args[1]})`;
                case 'sub': return `(${args[0]} - ${args[1]})`;
                case 'mul': return `(${args[0]} * ${args[1]})`;
                case 'div': return `(Math.abs(${args[1]}) > 1e-9 ? ${args[0]} / ${args[1]} : 0.0)`;
                case 'abs': return `Math.abs(${args[0]})`;
                case 'sin': return `Math.sin(${args[0]})`;
                case 'cos': return `Math.cos(${args[0]})`;
                case 'exp': return `Math.exp(${args[0]})`;
                case 'log1p': return `Math.log1p(Math.abs(${args[0]}))`;
                case 'neg': return `(-${args[0]})`;
                case 'sqr': return `(${args[0]} * ${args[0]})`;
                case 'cube': return `(${args[0]} * ${args[0]} * ${args[0]})`;
                case 'tanh': return `Math.tanh(${args[0]})`;
                case 'sqrt': return `Math.sqrt(Math.abs(${args[0]}))`;
                default: return '0.0';
            }
        }

        const expression = buildCode(template.genes[0]);

        const kernelSource = `
            return function(testX, testY, c, results, numTests, inputSize, constsPerGenome) {
                const numGenomes = results.length;
                for (let g = 0; g < numGenomes; g++) {
                    const genome_const_offset = g * constsPerGenome;
                    let errorSum = 0.0;
                    for (let t = 0; t < numTests; t++) {
                        let prediction = ${expression};
                        if (isNaN(prediction) || !isFinite(prediction)) prediction = 0.0;
                        const diff = prediction - testY[t];
                        errorSum += diff * diff;
                    }
                    const fitness = 1.0 / (1.0 + (errorSum / numTests));
                    results[g] = isNaN(fitness) ? 0.0 : fitness;
                }
            }
        `;

        return new Function(kernelSource)();
    }
}
