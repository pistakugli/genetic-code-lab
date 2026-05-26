import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { createBooleanParityProblem } from '../src/solvers/boolean-parity';

/**
 * 63.1.1 Integration Proof
 * Verifies that the Domain Switchboard correctly pivots to LOGIC mode.
 */

async function runIntegrationProof() {
    console.log("--- 63.1.1: INTEGRATION PROOF (BP-3) ---");

    // 1. Setup BP-3 Problem (Discrete)
    const problem = createBooleanParityProblem(3);
    const engine = new EvolutionEngine(problem, {
        population: { 
            size: 10, 
            maxGenomeSize: 20,
            eliteCount: 2,
            crossoverRate: 0.8,
            mutationRate: 0.1,
            maxGenomeDepth: 10,
            tournamentSize: 3
        },
        maxGenerations: 5
    });

    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    engine.setDiscoveryAgent(agent);
    // @ts-ignore
    engine.agentBridge = bridge;

    // 2. Initialize (this triggers DomainAnalyzer AND bridge sync)
    // @ts-ignore
    engine.ensureTestDataBuffers();
    
    // @ts-ignore
    const profile = engine.domainProfile;
    if (!profile) {
        console.error("FAIL: Domain analysis did not run.");
        process.exit(1);
    }
    console.log(`Detected Domain: ${profile.domainClass}`);

    if (profile.domainClass !== 'discrete') {
        console.error("FAIL: BP-3 misclassified.");
        process.exit(1);
    }

    // 3. Setup Agent Link
    engine.setDiscoveryAgent(agent);
    // @ts-ignore
    engine.agentBridge = bridge;
    bridge.setDomainProfile(profile);

    // 4. Run a few steps to trigger Agent logic
    console.log("Running steps to trigger Agent pivot...");
    await engine.step(0, 0);
    
    // Give background bridge a moment to process the snapshot
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Check if agent generated a logic pivot
    // @ts-ignore
    if (agent.currentAction === 'LOGIC_PIVOT') {
        process.stdout.write("Agent Status: LOGIC_PIVOT Detected ✅\n");
    } else {
        console.error("FAIL: Agent did not pivot to LOGIC strategy.");
        // @ts-ignore
        console.log("Current Action:", agent.currentAction);
        process.exit(1);
    }

    console.log("\n63.1.1: STATUS: PASS");
}

runIntegrationProof();
