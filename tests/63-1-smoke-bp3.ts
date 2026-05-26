import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { createBooleanParityProblem } from '../src/solvers/boolean-parity';

/**
 * 63.1.2 BP-3 SMOKE TEST
 * Goal: Autonomous victory on parity-3 problem (8 cases).
 */

async function runSmokeTest() {
    console.log("--- 63.1.2: BP-3 SMOKE TEST ---");

    const problem = createBooleanParityProblem(3);
    const engine = new EvolutionEngine(problem, {
        population: { 
            size: 100, 
            maxGenomeSize: 30,
            eliteCount: 10,
            crossoverRate: 0.8,
            mutationRate: 0.2,
            maxGenomeDepth: 12,
            tournamentSize: 7
        },
        maxGenerations: 20,
        targetFitness: 1.0
    });

    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    engine.setDiscoveryAgent(agent);
    // @ts-ignore
    engine.agentBridge = bridge;

    console.log("Starting Autonomous Evolution (BP-3)...");
    
    let gen = 0;
    let victory = false;

    while (gen <= 20) {
        const finished = await engine.step(gen, 0);
        
        // @ts-ignore
        const bestFit = engine.bestFitnessSoFar;
        console.log(`[Gen ${gen}] Best Fitness: ${bestFit.toFixed(4)}`);

        if (bestFit >= 0.999999) {
            victory = true;
            break;
        }
        gen++;
        
        // Brief pause for agent async processing
        await new Promise(resolve => setTimeout(resolve, 50));
    }

    if (victory) {
        process.stdout.write("BP-3 SMOKE TEST: VICTORY ✅ (1.0 Fitness Achieved)\n");
    } else {
        console.error("FAIL: BP-3 Smoke Test did not reach 1.0 fitness within 20 generations.");
        process.exit(1);
    }

    // FINAL VALIDATION: Signature Check
    // @ts-ignore
    const bestInd = engine.getBestGenome();
    if (bestInd) {
        const hasArith = bestInd.opcodes.some((op: number) => {
            const opcode = op >> 24;
            return opcode >= 1 && opcode <= 6; // ADD, SUB, MUL, DIV, SIN, COS
        });

        if (hasArith) {
            console.error("FAIL: Winning signature contains arithmetic opcodes (Structural Seal Broken).");
            process.exit(1);
        } else {
            console.log("Structural Integrity: PASS (Logic Only)");
        }
    }

    console.log("\n63.1.2: STATUS: PASS");
}

runSmokeTest();
