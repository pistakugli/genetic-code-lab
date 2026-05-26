import { SotaBenchmarks } from './quality-benchmarks';
import { DistributedManager } from '../core/distributed-manager';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { EvolutionConfig } from '../core/types';

async function runSkeleton() {
    process.stderr.write(`=== SKELETON AUDIT START ===\n`);
    const problem = SotaBenchmarks.getF3();
    const config: EvolutionConfig = {
        population: { 
            size: 10, 
            eliteCount: 2, 
            crossoverRate: 0.8, 
            mutationRate: 0.2, 
            maxGenomeSize: 32,
            maxGenomeDepth: 5,
            tournamentSize: 3
        },
        maxGenerations: 5,
        targetFitness: 0.999,
        stagnationLimit: 5,
        parallelEvaluators: 1
    };

    const bridge = new AsyncAgentBridge(new DiscoveryAgent());
    const manager = new DistributedManager(1, config, bridge);
    
    process.stderr.write(`[Skeleton] Launching Manager...\n`);
    await manager.start(problem, 42);
    process.stderr.write(`[Skeleton] Manager Finished.\n`);
    const result = manager.getBestOfRun();
    process.stderr.write(`[Best Fit] ${result ? result.fitness : 'NONE'}\n`);
    await manager.stop();
}

runSkeleton().catch(e => process.stderr.write(`FATAL: ${e.message}\n`));
