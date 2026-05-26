import { DistributedManager } from '../core/distributed-manager';
import { EvolutionConfig } from '../core/types';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { SotaBenchmarks } from './quality-benchmarks';
import { resolvePerformanceParallelism } from '../core/runtime-lane';

/**
 * F3 FORENSIC PROBE
 * Isolated F3-Rational run with detailed per-generation logging
 * to identify where parsimony collapse occurs.
 */
async function runF3Probe() {
    console.log("=== F3-RATIONAL FORENSIC PROBE ===");

    const problem = SotaBenchmarks.getF3();
    const seed = 42;
    const parallelEvaluators = resolvePerformanceParallelism(1);

    const config: EvolutionConfig = {
        maxGenerations: 150,
        targetFitness: 1.0,
        stagnationLimit: 25,
        parallelEvaluators,
        seed,
        population: {
            size: 10000,
            eliteCount: 10,
            crossoverRate: 0.9,
            mutationRate: 0.1,
            maxGenomeSize: 128,
            maxGenomeDepth: 10,
            tournamentSize: 7,
            parsimonyCoefficient: 0.001
        }
    };

    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);

    // Single island — easier to trace
    const manager = new DistributedManager(1, config, bridge);
    console.log(`[Topology] 1 island x ${parallelEvaluators} evaluators`);
    await manager.start(problem, seed);

    // Log every 5 seconds for 30 seconds
    for (let i = 0; i < 6; i++) {
        await new Promise(r => setTimeout(r, 5000));
        const stats = manager.getGlobalAnalytics();
        const s = stats[0];
        console.log(`[T=${(i+1)*5}s] Gen ${s.gen} | Best Fit: ${s.bestFitness.toFixed(6)} | Best Size: ${s.bestSize} | Entropy: ${s.entropy.toFixed(4)}`);
    }

    const finalStats = manager.getGlobalAnalytics();
    await manager.stop();
    
    const bestFit = finalStats[0].bestFitness;
    const bestSize = finalStats[0].bestSize;
    console.log(`\n[FINAL] Best Fit: ${bestFit.toFixed(6)} | Best Size: ${bestSize}`);
}

runF3Probe().catch(console.error);
