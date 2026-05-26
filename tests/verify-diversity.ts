import { EvolutionEngine } from '../src/core/evolution';
import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';
import { DiscoveryAgent } from '../src/agents/discovery-agent';

async function verifyDiversity() {
    console.log("--- PHASE 35: SURGICAL DIVERSITY VERIFIER ---");
    const size = 1000;
    const engine = new EvolutionEngine({
        functionSet: [{ name: 'add', arity: 2, implementation: (a: any, b: any) => a + b, returnType: 'number' }],
        terminalSet: [{ name: 'x', type: 'variable' }],
        fitnessFunction: () => Math.random(), 
        testCases: [{ input: [1], expectedOutput: 2 }]
    }, {
        population: { 
            size, 
            eliteCount: 1, 
            crossoverRate: 1.0, 
            mutationRate: 0.1, 
            maxGenomeDepth: 8, 
            maxGenomeSize: 128, 
            tournamentSize: 3, 
            parsimonyCoefficient: 0.002 
        }
    });

    const agent = new DiscoveryAgent();
    engine.setDiscoveryAgent(agent);

    (engine as any).isBinaryMode = true;
    (engine as any).currentBuffer = new PopulationBuffer(size, 128, 10, 1);
    (engine as any).nextBuffer = new PopulationBuffer(size, 128, 10, 1);
    
    // 1. Force a "Low Diversity" state (All individuals identical)
    console.log("[Test] Seeding Low Diversity state (Identical Genomes)...");
    const baked = { opcodes: [OpCode.ADD, OpCode.ADD], data: [0, 0], constants: [1, 1] };
    for (let i = 0; i < size; i++) {
        (engine as any).currentBuffer.writeGenome(i, baked.opcodes, baked.data, baked.constants);
    }
    
    const initialEntropy = (engine as any).currentBuffer.calculateEntropy(1000);
    console.log(`[Test] Initial Entropy (Identical): ${initialEntropy.toFixed(4)}`);

    // 2. Force High Diversity Bias in the agent
    // We'll override the analyzePopulationDynamics to return the expected structure
    agent.analyzePopulationDynamics = () => ({ bias: 1.0, intensity: 1.0 }); 
    (engine as any).agentBridge.pushPopulationAnalytics(initialEntropy, (engine as any).currentBuffer.nicheMap);
    console.log(`[Test] Forced Diversity Bias: ${(engine as any).agentBridge.getDiversityBias()}`);

    // 3. Run one generation
    console.log("[Test] Running one generation with 100% Crossover and 1.0 Diversity Bias...");
    // Injecting manually since evolve() is complex
    for (let i = 1; i < size; i++) {
        const p1 = (engine as any).binaryTournamentSelection((engine as any).currentBuffer, 7);
        const p2 = (engine as any).binaryTournamentSelection((engine as any).currentBuffer, 7);
        // Note: Crossover will generate 0.777 constants or varied structures if we use standard operators
        // But the key is: will it pick novel candidates?
        // In an identical population, novelty is impossible, BUT the reproduction logic 
        // with crossover should create variety.
    }

    // Actually, evolve() is better to test the whole loop.
    // We'll just run 2 generations.
    await engine.evolve(); // gen 1
    
    const postEntropy = (engine as any).currentBuffer.calculateEntropy(1000);
    console.log(`[Test] Post-Evolution Entropy: ${postEntropy.toFixed(4)}`);

    if (postEntropy > initialEntropy) {
        console.log("--- DIVERSITY RECOVERY VERIFIED ---");
    } else {
        console.error("[FAIL] Entropy did not recover despite Bias.");
        process.exit(1);
    }
}

verifyDiversity().catch(console.error);
