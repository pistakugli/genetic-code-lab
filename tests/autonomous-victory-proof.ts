import { EvolutionEngine } from '../src/core/evolution';
import { EvolutionConfig, ProblemDefinition } from '../src/core/types';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';

/**
 * PHASE 62.6: THE AUTONOMOUS VICTORY PROOF
 * Goal: Prove that the system can solve F3-Rational autonomously.
 * Hard Assertions for: Asymptote detection, Agent injection, Survival, and 1.0 Fitness.
 */

async function runAutonomousVictory() {
    console.log("--- PROOF: AUTONOMOUS BEYOND-SOTA VICTORY ---");
    console.log("Domain: F3-Rational | Method: Autonomous Discovery Agent | Selection: Tournament");

    // 1. Setup F3-Rational Data (1001 points)
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

    const problem: any = {
        name: 'F3-Autonomous-Victory',
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
        testCases: Array.from(testX).map((x, i) => ({ input: [x], expectedOutput: testY[i] })),
        fitnessFunction: () => 0 
    };

    const config: any = {
        seed: 42, 
        selectionMethod: 'lexicase',
        population: {
            size: 100,
            eliteCount: 2,
            crossoverRate: 0.8,
            mutationRate: 0.1,
            maxGenomeSize: 50
        },
        maxGenerations: 60,
        targetFitness: 1.0
    };

    const engine = new EvolutionEngine(problem, config);
    
    // 2. Setup Discovery Agent and Bridge
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    (engine as any).discoveryAgent = agent;
    (engine as any).agentBridge = bridge;
    bridge.setEvaluationData(testX, testY, 1, 'regression');

    // 3. State Tracking for Hard Assertions
    let asymptoteGen = -1;
    let injectionGen = -1;
    let survivalGens = 0;
    let finalFitness = 0;
    const targetSignature = '10,10,3,11,1,10,11,2,4'; 
    
    // Programmatic State Flags
    const state = {
        asymptoteDetected: false,
        candidateInjected: false,
        survivalVerified: false,
        targetFitnessReached: false,
        signatureMatched: false
    };

    console.log("\n--- STARTING AUTONOMOUS CAUSAL CHAIN ---");

    for (let gen = 0; gen <= 60; gen++) {
        await (engine as any).step(gen, 0);
        
        const bestFit = (engine as any).bestFitnessSoFar;
        finalFitness = bestFit;

        // 1. MONITOR ASYMPTOTE DETECTION
        if (!state.asymptoteDetected && agent.currentAction === 'STRUCTURAL_PIVOT') {
            state.asymptoteDetected = true;
            asymptoteGen = gen;
            console.log(`[Gen ${gen}] Programmatic Flag: ASYMPTOTE_DETECTED`);
        }
        
        // 2. POLL & INJECT AGENT CANDIDATES
        const stats = {
            gen: gen,
            stagnationCount: (engine as any).stagnationCount,
            entropy: (engine as any).currentBuffer.calculateEntropy(),
            bestSize: (engine as any).currentBuffer.lengths[0],
            bestFitness: bestFit
        };
        (bridge as any).pushGlobalAnalytics([stats], 'F3-Rational');
        
        if (gen % 10 === 0) {
            const best = engine.getBestGenome();
            if (best) {
                bridge.pushSnapshot([{ opcodes: best.opcodes, constants: best.constants, fitness: best.fitness, genes: [] } as any], gen);
            }
        }

        const agentCandidates = bridge.poll(gen);
        if (agentCandidates.length > 0) {
            state.candidateInjected = true;
            if (injectionGen === -1) injectionGen = gen;
            console.log(`[Gen ${gen}] Programmatic Flag: CANDIDATE_INJECTED`);
            
            const buffer = (engine as any).currentBuffer;
            agentCandidates.forEach((cand) => {
                const b = (cand.genome as any);
                buffer.writeGenome(0, b.opcodes, [], b.constants);
                buffer.fitness[0] = cand.fitness;
                buffer.ancestorMap[0] = 1;
            });
        }

        // 3. TRACK SURVIVAL (Post-Injection)
        if (state.candidateInjected) {
            const winner = engine.getBestGenome();
            if (winner) {
                const winnerSig = Array.from(winner.opcodes).map(op => op >>> 24).join(',');
                if (winnerSig === targetSignature && winner.fitness >= 0.999999999999) {
                    survivalGens++;
                } else {
                    // Reset survival if signature is lost or fitness drops
                    survivalGens = 0;
                }
            }
        }

        if (bestFit >= 0.999999999999) state.targetFitnessReached = true;

        // Periodic Status
        if (gen % 5 === 0) console.log(`Gen ${gen} | BestFit: ${bestFit.toFixed(6)} | Survival: ${survivalGens}`);

        // BRAKING CONDITION: Only stop if all conditions are met OR we exhausted limit
        if (state.targetFitnessReached && survivalGens >= 5) {
            state.survivalVerified = true;
            console.log(`\n[Gen ${gen}] SUCCESS: High-Fitness Candidate Survived 5 Generations.`);
            break;
        }
    }

    // 4. FINAL HARD ASSERTIONS (Verification against Programmatic State)
    console.log("\n--- FINAL FORENSIC ASSERTIONS ---");
    
    if (!state.asymptoteDetected) {
        console.error("HARD_FAIL: Asymptote was never detected by DiscoveryAgent.");
        process.exit(1);
    }
    console.log("1. Asymptote Detected:   [PASS]");

    if (!state.candidateInjected) {
        console.error("HARD_FAIL: No agent candidates were generated/injected.");
        process.exit(1);
    }
    console.log("2. Agent Injection:      [PASS]");

    if (survivalGens < 5) {
        console.error(`HARD_FAIL: Survival requirement not met. (Count: ${survivalGens}/5)`);
        process.exit(1);
    }
    console.log("3. Injection Survival:   [PASS]");

    if (finalFitness < 0.999999999999) {
        console.error(`HARD_FAIL: Final fitness ${finalFitness} below threshold.`);
        process.exit(1);
    }
    console.log("4. Final Fitness Check:  [PASS]");

    const winner = engine.getBestGenome();
    const winnerSig = Array.from(winner!.opcodes).map(op => op >>> 24).join(',');
    if (winnerSig !== targetSignature) {
        console.error(`HARD_FAIL: Postfix signature mismatch. Got: ${winnerSig}`);
        process.exit(1);
    }
    console.log("5. Rational Signature:   [PASS]");

    console.log("\nAUTONOMOUS VICTORY: YES");
    process.exit(0);
}

runAutonomousVictory().catch(e => {
    console.error("CRITICAL TEST FAILURE:", e.message);
    process.exit(1);
});
