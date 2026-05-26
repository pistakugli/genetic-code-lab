import { EvolutionEngine } from '../src/core/evolution';
import { JITCompiler } from '../src/core/jit-compiler';
import { VectorizedEvaluator } from '../src/core/vectorized-evaluator';
import { TestCase, Genome } from '../src/core/types';

/**
 * 🧪 Vectorized Master JIT Parity Test
 * Verifies that the new chunked evaluator matches the original JIT semantics.
 */
async function runParityTest() {
    console.log('--- VECTORIZED JIT PARITY TEST ---');
    
    // 1. Setup Data
    const numGenomes = 100;
    const numTests = 50;
    const inputSize = 2; // x, y

    const testCases: TestCase<number[], number>[] = [];
    const testX = new Float64Array(numTests * inputSize);
    const testY = new Float64Array(numTests);

    for (let i = 0; i < numTests; i++) {
        const x = [Math.random() * 10 - 5, Math.random() * 10 - 5];
        const y = x[0] * Math.sin(x[1]) + x[0] * x[0] - x[1]; // Known function
        testCases.push({ input: x, expectedOutput: y });
        testX[i * inputSize] = x[0];
        testX[i * inputSize + 1] = x[1];
        testY[i] = y;
    }

    // 2. Setup Population Manager
    const popConfig = {
        size: numGenomes,
        eliteCount: 2,
        crossoverRate: 0.8,
        mutationRate: 0.2,
        maxGenomeDepth: 5,
        maxGenomeSize: 50,
        tournamentSize: 3
    };

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

    const { PopulationManager } = require('../src/core/population');
    const popManager = new PopulationManager(funcSet, termSet, popConfig);
    
    // Initialize population to get random genomes
    const population = popManager.initialize();
    const genomes: Genome[] = population.genomes;

    console.log(`Created ${genomes.length} random genomes.`);

    // 3. Evaluate with Old JIT (Baseline)
    console.log('Evaluating with Individual JIT (Baseline)...');
    const baselineResults: number[] = [];
    const startTimeJIT = Date.now();
    
    for (const genome of genomes) {
        const { fn, constants } = JITCompiler.compile(genome);
        let errorSum = 0;
        for (const tc of testCases) {
            const pred = fn(tc.input, constants);
            const diff = pred - tc.expectedOutput;
            errorSum += diff * diff;
        }
        baselineResults.push(1.0 / (1.0 + (errorSum / numTests)));
    }
    const endTimeJIT = Date.now();
    console.log(`Individual JIT took: ${endTimeJIT - startTimeJIT}ms`);

    // 4. Evaluate with New Vectorized JIT
    console.log('Evaluating with Vectorized Master JIT (v2 Structural Batching)...');
    const startTimeVec = Date.now();
    const vectorizedResults = await VectorizedEvaluator.evaluateMassiveBatch(genomes, testX, testY, inputSize);
    const endTimeVec = Date.now();
    console.log(`Vectorized JIT took: ${endTimeVec - startTimeVec}ms`);

    // 5. Compare Results
    let maxError = 0;
    let failCount = 0;
    const epsilon = 1e-10;

    for (let i = 0; i < numGenomes; i++) {
        const diff = Math.abs(baselineResults[i] - vectorizedResults[i]);
        if (diff > maxError) maxError = diff;
        if (diff > epsilon) {
            failCount++;
            // console.log(`Genome ${i} mismatch: ${baselineResults[i]} vs ${vectorizedResults[i]} (diff: ${diff})`);
        }
    }

    console.log('--- RESULTS ---');
    console.log(`Max Error: ${maxError}`);
    console.log(`Failures (>1e-10): ${failCount}`);
    
    if (failCount === 0) {
        console.log('✅ PASS: Semantic Equivalence Confirmed.');
    } else {
        console.error('❌ FAIL: Numerical mismatch detected.');
        process.exit(1);
    }
}

runParityTest().catch(console.error);
