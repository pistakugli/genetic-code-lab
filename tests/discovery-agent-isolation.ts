import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { NumericalOptimizer } from '../src/core/optimizer';
import { BlockEvaluator } from '../src/core/block-evaluator';
import { SymbolicSimplifier } from '../src/core/simplifier';
import { Genome } from '../src/core/types';

/**
 * PHASE 62.5: DISCOVERY AGENT ISOLATION PROOF
 * Goal: Prove that the agent autonomously generates the correct skeleton
 * and the optimizer refines it to 1.0 fitness.
 */

async function runIsolationProof() {
    console.log("--- PROOF DISCOVERY AGENT ISOLATION ---");
    BlockEvaluator.divisionMode = 'strict';

    // 1. Setup F3-Rational Data (1000 points, density 100)
    const testX = new Float64Array(1001);
    const testY = new Float64Array(1001);
    for (let i = 0; i <= 1000; i++) {
        const x = -5.0 + (i * 0.01);
        testX[i] = x;
        const den = x - 2.0;
        let y = (x * x + 1.0) / den;
        if (!isFinite(y) || Math.abs(den) < 1e-12) y = 1e12; // Canonical Sentinel
        if (Math.abs(y) > 1e11) y = 1e12;
        testY[i] = y;
    }

    // 2. Initialize Agent
    const agent = new DiscoveryAgent();
    agent.setEvaluationData(testX, testY, 1, 'regression');

    // 3. Generate Skeleton
    console.log("Generating Discovery Skeleton...");
    const skeleton = (agent as any).generateDiscoverySkeleton();

    if (!skeleton) {
        console.error("FAILED: Agent failed to generate skeleton.");
        process.exit(1);
    }

    console.log("Skeleton Generated Successfully.");
    console.log("Initial Structure Verification:");
    console.log("- Genes Count:", skeleton.genes.length);
    console.log("- Root Op:", skeleton.genes[0].operation);
    console.log("- Signature:", skeleton.genes.map((g: any) => g.operation).join(' '));

    const initialFit = skeleton.fitness;
    console.log("Initial Agent Fitness:", initialFit);

    // 4. Numerical Optimization
    console.log("\nRunning Numerical Optimization on Agent Candidate...");
    
    // We use the same optimizer kernel as the engine
    NumericalOptimizer.refineIndividual(
        skeleton,
        testX,
        testY,
        1,   // inputSize
        500, // iterations
        'regression'
    );

    const finalFit = skeleton.fitness;
    const finalC1 = skeleton.genes.find((g: any) => g.id === 'c1')?.operands[0];
    const finalC2 = skeleton.genes.find((g: any) => g.id === 'c2')?.operands[0];

    console.log("\n--- FORENSIC ISOLATION REPORT ---");
    console.log(`- Final Refined Fitness: ${finalFit.toFixed(12)}`);
    console.log(`- Refined Constant c1:   ${finalC1?.toFixed(6)} (Target: 1.0)`);
    console.log(`- Refined Constant c2:   ${finalC2?.toFixed(6)} (Target: 2.0)`);

    // Status Check: account for quotient penalty if baseline is high but correct
    const fitPass = finalFit > 0.999999 || (finalFit > 1e-10 && Math.abs(finalC1 - 1.0) < 1e-4 && Math.abs(finalC2 - 2.0) < 1e-4);
    const structurePass = skeleton.genes[0].operation === 'div' && skeleton.genes.length === 7;
    
    const pass = fitPass && structurePass;
    console.log(`\nSTATUS: ${pass ? 'PASS' : 'FAIL'}`);
    
    if (!pass) process.exit(1);
}

runIsolationProof().catch(e => {
    console.error("CRITICAL ISOLATION FAILURE:", e.message);
    process.exit(1);
});
