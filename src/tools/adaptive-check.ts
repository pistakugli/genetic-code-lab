import { DistributedManager } from '../core/distributed-manager';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { SotaBenchmarks } from './quality-benchmarks';
import { DEFAULT_EVOLUTION_CONFIG } from '../core/evolution';

async function run() {
    console.log("=== PHASE 41: ADAPTIVE MUTATION TARGETED AUDIT ===");
    const problem = SotaBenchmarks.getF3();
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    
    const config = {
        ...DEFAULT_EVOLUTION_CONFIG,
        population: { ...DEFAULT_EVOLUTION_CONFIG.population, size: 2000 },
        maxGenerations: 60,
        parallelEvaluators: 1
    };

    const manager = new DistributedManager(5, config, bridge);
    await manager.start(problem, 42);

    // Wait for and monitor stagnation
    await new Promise(r => setTimeout(r, 20000));
    await manager.stop();
    process.exit(0);
}

run().catch(e => { console.error(e); process.exit(1); });
