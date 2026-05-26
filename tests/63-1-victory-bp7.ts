import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { OpCode } from '../src/core/pop-buffer';
import { createBooleanParityProblem } from '../src/solvers/boolean-parity';

/**
 * 63.1.3 BP-7 VICTOR PROOF
 * Goal: Autonomous victory on parity-7 problem (128 cases).
 * Hard Conditions: Agent injection, Survival, and 1.0 Accuracy.
 */

async function runVictoryProof() {
    console.log("--- 63.1.3: BP-7 VICTORY PROOF ---");

    const problem = createBooleanParityProblem(7);
    const engine = new EvolutionEngine(problem, {
        population: { 
            size: 500, 
            maxGenomeSize: 100,
            eliteCount: 20,
            crossoverRate: 0.8,
            mutationRate: 0.3, // Higher noise for parity
            maxGenomeDepth: 20,
            tournamentSize: 7
        },
        maxGenerations: 200,
        targetFitness: 1.0
    });

    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    engine.setDiscoveryAgent(agent);
    // @ts-ignore
    engine.agentBridge = bridge;

    console.log("Starting Full Autonomous Evolution (BP-7)...");
    
    let gen = 0;
    let victory = false;

    while (gen <= 200) {
        const finished = await engine.step(gen, 0);
        
        // @ts-ignore
        const bestFit = engine.bestFitnessSoFar;
        
        if (gen % 10 === 0) {
            console.log(`[Gen ${gen}] Best Accuracy: ${bestFit.toFixed(4)}`);
        }

        if (bestFit >= 0.999999) {
            victory = true;
            console.log(`[Gen ${gen}] TARGET REACHED: 1.0000`);
            break;
        }
        gen++;
        
        // Pause to not starve the agent thread
        await new Promise(resolve => setTimeout(resolve, 10));
    }

    if (victory) {
        process.stdout.write("\n63.1.3: BP-7 VICTORY ACHIEVED ✅\n");
    } else {
        console.error("\nFAIL: BP-7 Victory Proof Failed. Did not reach 1.0 fitness within 200 generations.");
        process.exit(1);
    }

    // SURVIVAL CHECK & EXPLICIT OPCODE AUDIT
    const finalBestBinary = engine.getBestBinary();
    const finalBest = engine.getBestGenome();
    console.log("\n--- EXPLICIT STRUCTURAL AUDIT ---");
    if (!finalBestBinary || !finalBest) {
        console.error("FAIL: Missing final winner state for audit.");
        process.exit(1);
    }
    
    let arithmeticCount = 0;
    const arithOps = [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.SIN, OpCode.COS];
    
    for (const opRaw of finalBestBinary.opcodes) {
        const op = opRaw >> 24;
        if (arithOps.includes(op)) {
            arithmeticCount++;
            console.error(`[AUDIT] Found forbidden arithmetic opcode: ${op}`);
        }
    }

    if (arithmeticCount === 0) {
        console.log("Zero arithmetic opcodes: PASS ✅");
    } else {
        console.error(`FAIL: Structural Isolation Breach! Found ${arithmeticCount} arithmetic opcodes.`);
        process.exit(1);
    }

    console.log("Final Fitness:", finalBest.fitness);
    console.log("Final Accuracy:", Math.round(finalBest.fitness * 128), "/ 128");

    process.stdout.write("\n63.1.3: STATUS: PASS\n");
}

runVictoryProof();
