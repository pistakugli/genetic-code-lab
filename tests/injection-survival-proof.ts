import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { PostfixBridge } from '../src/core/postfix-bridge';
import { EvolutionConfig, ProblemDefinition } from '../src/core/types';

/**
 * PHASE 62.5: INJECTION SURVIVAL PROOF
 * Goal: Prove that a 1.0 candidate survives 5 generations under Tournament selection.
 * Tracking signature count (identity by Postfix opcodes).
 */

async function runSurvivalProof() {
    console.log("--- PROOF INJECTION SURVIVAL ---");

    // 1. Setup F3-Rational Data
    const testX = new Float64Array(1001);
    const testY = new Float64Array(1001);
    for (let i = 0; i <= 1000; i++) {
        const x = -5.0 + (i * 0.01);
        testX[i] = x;
        const den = x - 2.0;
        let y = (x * x + 1.0) / den;
        if (!isFinite(y) || Math.abs(den) < 1e-12) y = 1e12; 
        if (Math.abs(y) > 1e11) y = 1e12;
        testY[i] = y;
    }

    const problem: ProblemDefinition<number, number> = {
        name: 'F3-Rational-Survival',
        functionSet: [
            { name: 'add', arity: 2, implementation: (a: any, b: any) => a + b, returnType: 'number' },
            { name: 'sub', arity: 2, implementation: (a: any, b: any) => a - b, returnType: 'number' },
            { name: 'mul', arity: 2, implementation: (a: any, b: any) => a * b, returnType: 'number' },
            { name: 'div', arity: 2, implementation: (a: any, b: any) => a / b, returnType: 'number' }
        ],
        terminalSet: [
            { name: 'x', type: 'variable' },
            { name: 'const', type: 'constant' }
        ],
        testCases: Array.from(testX).map((x, i) => ({ input: x, expectedOutput: testY[i] })),
        fitnessFunction: () => 0 // Not used directly by engine in DOGP mode
    };

    const config: Partial<EvolutionConfig> = {
        seed: 424242, // Fixed seed for reproducibility
        selectionMethod: 'tournament',
        population: {
            size: 100,
            eliteCount: 2,
            crossoverRate: 0.8,
            mutationRate: 0.1,
            maxGenomeDepth: 8,
            maxGenomeSize: 100,
            tournamentSize: 3
        },
        maxGenerations: 5
    };

    const engine = new EvolutionEngine(problem, config);
    // Note: No initialize() call needed, step(0) handles it.

    // 2. Generation of Injected Candidate
    const agent = new DiscoveryAgent();
    agent.setEvaluationData(testX, testY, 1, 'regression');
    console.log("Agent generating candidate...");
    const candidate = (agent as any).generateDiscoverySkeleton();
    
    // Normalize and refine candidate to 1.0
    const baked = PostfixBridge.bake(candidate, ['x']);
    const targetSignature = baked.opcodes.join(',');
    console.log("Injected Signature (Postfix):", targetSignature);

    // 3. Injection Phase (at Generation 0)
    console.log("Initializing Evolution (Gen 0)...");
    await (engine as any).step(0, 0);
    
    console.log("Injecting candidate into Current Population Buffer...");
    const buffer = (engine as any).currentBuffer;
    const worstIdx = 99; // Replace one index
    const packedOps = new Uint32Array(baked.opcodes.length);
    for(let i=0; i<baked.opcodes.length; i++) packedOps[i] = (baked.opcodes[i] << 24) | (baked.data[i] & 0xFFFFFF);
    
    buffer.writeGenome(worstIdx, packedOps, baked.data, baked.constants);
    buffer.fitness[worstIdx] = 1.0;

    // 4. Survival Tracking Loop
    console.log("\n--- SURVIVAL TRACKING ---");
    for (let gen = 1; gen <= 5; gen++) {
        await (engine as any).step(gen, 0);
        
        // Count occurrences of the target signature
        let signatureCount = 0;
        const popSize = 100;
        const currentBuffer = (engine as any).currentBuffer;
        
        for (let i = 0; i < popSize; i++) {
            const rawOps = currentBuffer.getOpcodes(i);
            // We only care about opcode sequence for identity (structure)
            const sig = Array.from(rawOps).map((op: any) => op >>> 24).join(',');
            if (sig === targetSignature) {
                signatureCount++;
            }
        }

        const bestFit = (engine as any).bestFitnessSoFar;
        console.log(`Gen ${gen} | Best Fitness: ${bestFit.toFixed(6)} | Signature Count: ${signatureCount}`);
        
        if (signatureCount === 0) {
            console.error(`FAILURE: Candidate Extinct at Gen ${gen}`);
            process.exit(1);
        }
    }

    console.log("\nSTATUS: PASS (Candidate Survived 5 Generations)");
    process.exit(0);
}

runSurvivalProof().catch(e => {
    console.error("CRITICAL SURVIVAL FAILURE:", e.message);
    process.exit(1);
});
