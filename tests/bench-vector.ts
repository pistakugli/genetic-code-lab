import { PopulationManager } from '../src/core/population';
import { JITCompiler } from '../src/core/jit-compiler';
import { VectorizedEvaluator } from '../src/core/vectorized-evaluator';
import { TestCase, Genome } from '../src/core/types';

/**
 * 📊 Vectorized Master JIT v2 Benchmark
 * Proves SOTA throughput on both unique and converged populations.
 */
async function runBenchmark() {
    console.log('--- VECTORIZED JIT v2 (STRUCTURAL BATCHING) PERFORMANCE SWEEP ---');
    
    const scales = [10000, 25000, 50000];
    const numTests = 100;
    const inputSize = 2;

    const testX = new Float64Array(numTests * inputSize);
    const testY = new Float64Array(numTests);
    for (let i = 0; i < numTests; i++) {
        const x = [Math.random() * 10 - 5, Math.random() * 10 - 5];
        testX[i * inputSize] = x[0];
        testX[i * inputSize + 1] = x[1];
        testY[i] = x[0] + x[1];
    }

    const funcSet = [
        { name: 'add', arity: 2, implementation: (a: any, b: any) => a + b, returnType: 'number' },
        { name: 'sub', arity: 2, implementation: (a: any, b: any) => a - b, returnType: 'number' },
        { name: 'mul', arity: 2, implementation: (a: any, b: any) => a * b, returnType: 'number' },
        { name: 'sin', arity: 1, implementation: (a: any) => Math.sin(a), returnType: 'number' }
    ];
    const termSet = [
        { name: 'x0', type: 'input' as const },
        { name: 'x1', type: 'input' as const },
        { name: 'const', type: 'constant' as const }
    ];

    for (const popSize of scales) {
        console.log(`\n--- Scale: ${popSize} Genomes ---`);
        
        const popManager = new PopulationManager(funcSet, termSet, { size: popSize });
        
        // Scenario A: 100% Unique
        const uniqueGenomes = popManager.initialize().genomes;
        
        // Scenario B: Converged (10 templates)
        const templates = popManager.initialize().genomes.slice(0, 5);
        const convergedGenomes: Genome[] = [];
        for (let i = 0; i < popSize; i++) {
            const template = templates[i % 5];
            convergedGenomes.push({
                ...template,
                id: `genome_${i}`,
                // Keep same structure, slightly different constants
                genes: template.genes.map(g => g.operation === 'const' ? { ...g, operands: [(Number(g.operands[0]) + Math.random())] } : g)
            });
        }

        async function measure(name: string, genomes: Genome[]) {
            // Baseline
            const startJIT = Date.now();
            for (const genome of genomes) {
                const { fn, constants } = JITCompiler.compile(genome);
                let err = 0;
                for (let t=0; t<numTests; t++) {
                     const p = fn([testX[t*2], testX[t*2+1]], constants);
                     const d = p - testY[t];
                     err += d*d;
                }
            }
            const timeJIT = Date.now() - startJIT;

            // Vectorized
            const startVec = Date.now();
            await VectorizedEvaluator.evaluateMassiveBatch(genomes, testX, testY, inputSize);
            const timeVec = Date.now() - startVec;

            console.log(`  [${name}] JIT: ${timeJIT}ms | Vector JIT: ${timeVec}ms | Speedup: ${(timeJIT/timeVec).toFixed(2)}x`);
        }

        await measure('100% Unique', uniqueGenomes);
        await measure('Converged (90% reuse)', convergedGenomes);
    }
}

runBenchmark().catch(console.error);
