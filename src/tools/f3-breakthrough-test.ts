import { SotaBenchmarks } from './quality-benchmarks';
import { DistributedManager } from '../core/distributed-manager';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { resolveDistributedIslands, resolvePerformanceParallelism } from '../core/runtime-lane';

/**
 * PHASE 55: PILOT BREAKTHROUGH TEST
 * Validating the 'Beyond SOTA' Hybrid Engine.
 */
async function runBreakthrough() {
    console.log('=== PHASE 55: F3 BREAKTHROUGH TEST ===');
    console.log('Target: 1.0 Fitness (Perfect Match)');
    
    const problem = SotaBenchmarks.getF3();
    const distributedIslands = resolveDistributedIslands(5);
    const islandParallelEvaluators = resolvePerformanceParallelism(1);
    const config = {
        population: {
            size: 5000,
            eliteCount: 50,
            crossoverRate: 0.8,
            mutationRate: 0.2,
            maxGenomeSize: 128,
            tournamentSize: 64
        },
        maxGenerations: 500, 
        targetFitness: 0.999999,
        stagnationLimit: 50,
        parallelEvaluators: islandParallelEvaluators,
        islandMode: true,
        solverType: 'math' as const
    };

    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    
    // Phase 56.1: Provide high-precision evaluation data with weights for pre-refinement
    const testX = new Float64Array(problem.testCases.length);
    const testY = new Float64Array(problem.testCases.length);
    const caseWeights = new Float64Array(problem.testCases.length);
    for (let i = 0; i < problem.testCases.length; i++) {
        testX[i] = problem.testCases[i].input[0];
        testY[i] = problem.testCases[i].expectedOutput;
        caseWeights[i] = problem.testCases[i].weight || 1.0;
    }
    bridge.setEvaluationData(testX, testY, 1, 'regression', caseWeights);

    const manager = new DistributedManager(distributedIslands, config as any, bridge);
    
    console.log('Launching Islands...');
    console.log(`Topology: ${distributedIslands} islands x ${islandParallelEvaluators} evaluators`);
    await manager.start(problem, 42);
    
    const result = manager.getBestOfRun();
    if (!result) {
        console.log('No result found.');
        return;
    }

    console.log('\n=== RESULTS ===');
    console.log(`Best Global Fitness (Internal): ${result.fitness.toFixed(6)}`);
    
    // HONEST GLOBAL VALIDATION
    // Phase 59: Expanding clip range to accommodate rational peaks
    const targetMin = -1e15;
    const targetMax = 1e15;
    
    const cleanedPreds = result.predictions.map((p: number) => {
        if (isNaN(p) || !isFinite(p)) return 0;
        return Math.max(targetMin, Math.min(targetMax, p));
    });

    const mae = cleanedPreds.reduce((sum: number, p: number, idx: number) => sum + Math.abs(problem.testCases[idx].expectedOutput - p), 0) / problem.testCases.length;
    const honestFit = 1.0 / (1.0 + mae);
    
    console.log(`Honest Global MAE: ${mae.toFixed(6)}`);
    console.log(`Honest Global Fitness: ${honestFit.toFixed(6)}`);
    
    if (honestFit > 0.95) {
        console.log('!!! SOTA BREAKTHROUGH ACHIEVED !!!');
    } else {
        console.log('Result stabilized at baseline. Further refinement needed.');
    }
}

runBreakthrough().catch(console.error);
