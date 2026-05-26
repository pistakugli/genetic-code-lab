import { NumericalOptimizer } from '../src/core/optimizer';
import { Gene, Genome } from '../src/core/types';
import { verifyF3 } from './canonical-f3-verifier';

/**
 * STEP 3: GRAMMAR & OPTIMIZER COMPLETENESS PROOF
 */

async function runStep3() {
    console.log("--- PROOF GRAMMAR & OPTIMIZER (STEP 3) ---");

    // 1. Numerator Proof: x^2 + c where c=0.9
    const numGenes: Gene[] = [
        { id: '1', type: 'function', operation: 'add', operands: ['2', '3'] },
        { id: '2', type: 'function', operation: 'mul', operands: ['4', '4'] },
        { id: '3', type: 'terminal', operation: 'constant', operands: [0.9] },
        { id: '4', type: 'terminal', operation: 'x', operands: [] }
    ];
    const numGenome: Genome = { id: 'num', genes: numGenes, fitness: 0, generation: 0, age: 0 };
    
    // 2. Denominator Proof: x - c where c=1.9
    const denGenes: Gene[] = [
        { id: '1', type: 'function', operation: 'sub', operands: ['2', '3'] },
        { id: '2', type: 'terminal', operation: 'x', operands: [] },
        { id: '3', type: 'terminal', operation: 'constant', operands: [1.9] }
    ];
    const denGenome: Genome = { id: 'den', genes: denGenes, fitness: 0, generation: 0, age: 0 };

    // 3. Prepare Test Data for F3-Rational (1000 point high-res grid)
    const testPoints = 1000;
    const testX = new Float64Array(testPoints);
    const testY = new Float64Array(testPoints);
    const start = -5, end = 5, step = (end - start) / (testPoints - 1);
    for (let i = 0; i < testPoints; i++) {
        const x = start + i * step;
        testX[i] = x;
        const den = x - 2;
        testY[i] = Math.abs(den) < 1e-12 ? 1e12 : (x * x + 1) / den;
    }

    // 4. Forensic Sweep: check c2 around 2.0
    console.log("\n--- FORENSIC SWEEP (c2 around 2.0) ---");
    const sweepC1 = 1.0;
    for (let sweepC2 = 1.95; sweepC2 <= 2.05; sweepC2 += 0.01) {
        const sweepEval = (x: number) => (x * x + sweepC1) / (x - sweepC2);
        const p = verifyF3(sweepEval, testPoints);
        console.log(`c2=${sweepC2.toFixed(3)} | MAE=${p.mae.toFixed(4)} | HonestFit=${p.honestFitness.toFixed(6)}`);
    }

    // 5. Build and Optimize
    const fullGenes: Gene[] = [
        { id: 'root', type: 'function', operation: 'div', operands: ['num', 'den'] },
        { id: 'num', type: 'function', operation: 'add', operands: ['x2', 'c1'] },
        { id: 'x2', type: 'function', operation: 'mul', operands: ['x', 'x'] },
        { id: 'x', type: 'terminal', operation: 'x', operands: [] },
        { id: 'c1', type: 'terminal', operation: 'const', operands: [0.9] }, // Should be 1.0
        { id: 'den', type: 'function', operation: 'sub', operands: ['x', 'c2'] },
        { id: 'c2', type: 'terminal', operation: 'const', operands: [1.9] }, // Should be 2.0
    ];
    const fullGenome: Genome = { id: 'full', genes: fullGenes, fitness: 0, generation: 0, age: 0 };

    console.log("\nStarting Deep Optimization (2000 iter)...");
    NumericalOptimizer.refineIndividual(fullGenome, testX, testY, 1, 2000); 

    console.log(`\nFinal Fitness: ${fullGenome.fitness.toFixed(15)}`);
    const c1 = fullGenome.genes[4].operands[0] as number;
    const c2 = fullGenome.genes[6].operands[0] as number;
    console.log(`Refined Constants: c1=${c1.toFixed(6)}, c2=${c2.toFixed(6)}`);

    const status = fullGenome.fitness > 0.999 ? "PASS" : "FAIL";
    console.log(`\nSTATUS: ${status}`);
}

runStep3().catch(console.error);
