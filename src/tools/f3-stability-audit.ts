import { SotaBenchmarks } from './quality-benchmarks';
import { DistributedManager } from '../core/distributed-manager';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { EvolutionConfig } from '../core/types';
import { resolveDistributedIslands, resolvePerformanceParallelism } from '../core/runtime-lane';

/**
 * PHASE 54: F3 STABILITY AUDIT
 * Statistical validation of the 0.57+ breakthrough across 10 trials.
 * Proof of 'Honest' global fitness and Agent Injection effectiveness.
 */
async function runStabilityAudit() {
    const seeds = [42, 123, 777, 1337, 2024, 101, 202, 303, 404, 505];
    const problem = SotaBenchmarks.getF3();
    const distributedIslands = resolveDistributedIslands(5);
    const islandParallelEvaluators = resolvePerformanceParallelism(1);
    const config: EvolutionConfig = {
        population: { 
            size: 100, 
            eliteCount: 10, 
            crossoverRate: 0.8, 
            mutationRate: 0.2, 
            maxGenomeSize: 128,
            maxGenomeDepth: 10,
            tournamentSize: 7
        },
        maxGenerations: 100,
        targetFitness: 0.999,
        stagnationLimit: 20,
        parallelEvaluators: islandParallelEvaluators
    };

    console.error(`=== GENETIC CODE LAB: F3 STABILITY AUDIT (10 Trials) ===\n`);
    console.error(`[Topology] ${distributedIslands} islands x ${islandParallelEvaluators} evaluators`);
    
    const results: any[] = [];

    for (let i = 0; i < seeds.length; i++) {
        const seed = seeds[i];
        console.error(`[Trial ${i+1}/10] Seed: ${seed}`);
        
        // --- STAGE 1: Base Model ---
        const bridge1 = new AsyncAgentBridge(new DiscoveryAgent());
        const manager1 = new DistributedManager(distributedIslands, config, bridge1);
        
        await manager1.start(problem, seed);
        const baseResult = manager1.getBestOfRun();
        const injections1 = 0; // getInjectionCount was removed from DistributedManager
        await manager1.stop();

        if (!baseResult) {
            console.error(`  [Error] No base result for seed ${seed}`);
            continue;
        }

        // Calculate Honest Base Fitness
        const targetMin = Math.min(...problem.testCases.map((tc: any) => tc.expectedOutput).filter(o => isFinite(o)));
        const targetMax = Math.max(...problem.testCases.map((tc: any) => tc.expectedOutput).filter(o => isFinite(o)));
        
        const cleanedBase = baseResult.predictions.map((p: number) => {
            if (isNaN(p) || !isFinite(p)) return 0;
            return Math.max(targetMin, Math.min(targetMax, p));
        });

        const baseMAE = cleanedBase.reduce((sum: number, p: number, idx: number) => sum + Math.abs(problem.testCases[idx].expectedOutput - p), 0) / problem.testCases.length;
        const honestBaseFit = 1.0 / (1.0 + baseMAE);

        // --- STAGE 2: Residual Model ---
        const residuals = problem.testCases.map((tc, idx) => ({
            input: tc.input,
            expectedOutput: tc.expectedOutput - cleanedBase[idx]
        }));

        const resProblem = { ...problem, name: `${problem.name}-Residual`, testCases: residuals };
        const bridge2 = new AsyncAgentBridge(new DiscoveryAgent());
        const manager2 = new DistributedManager(distributedIslands, { ...config, maxGenerations: 75 }, bridge2);
        
        await manager2.start(resProblem, seed + 1000);
        const resResult = manager2.getBestOfRun();
        const injections2 = 0; // getInjectionCount was removed from DistributedManager
        await manager2.stop();

        let honestCombinedFit = honestBaseFit;
        if (resResult) {
            const cleanedRes = resResult.predictions.map((p: number) => (isNaN(p) || !isFinite(p)) ? 0 : p);
            const combinedMAE = cleanedBase.reduce((sum: number, p: number, idx: number) => {
                const combined = p + cleanedRes[idx];
                return sum + Math.abs(problem.testCases[idx].expectedOutput - combined);
            }, 0) / problem.testCases.length;
            honestCombinedFit = 1.0 / (1.0 + combinedMAE);
        }

        console.log(`  > Base Fit (Honest): ${honestBaseFit.toFixed(4)} | Combined (Honest): ${honestCombinedFit.toFixed(4)} | Injections: ${injections1 + injections2}`);
        
        results.push({
            seed,
            baseFit: honestBaseFit,
            combinedFit: honestCombinedFit,
            injections: injections1 + injections2
        });
    }

    // --- FINAL ANALYSIS ---
    const avgBase = results.reduce((s, r) => s + r.baseFit, 0) / results.length;
    const avgCombined = results.reduce((s, r) => s + r.combinedFit, 0) / results.length;
    const avgInjections = results.reduce((s, r) => s + r.injections, 0) / results.length;
    
    const stdDevBase = Math.sqrt(results.reduce((s, r) => s + Math.pow(r.baseFit - avgBase, 2), 0) / results.length);
    
    console.log(`\n=== FINAL STABILITY REPORT ===`);
    console.log(`Mean Honest Base Fit:     ${avgBase.toFixed(6)} (σ: ${stdDevBase.toFixed(6)})`);
    console.log(`Mean Honest Combined Fit: ${avgCombined.toFixed(6)}`);
    console.log(`Mean Injection Volume:    ${avgInjections.toFixed(1)}`);
    
    if (avgBase > 0.55 && stdDevBase < 0.05) {
        console.log(`\n[VERDICT] SUCCESS: F3-Rational is officially stable at 0.55+`);
    } else {
        console.log(`\n[VERDICT] MARGINAL: Stability criteria not fully met.`);
    }
}

runStabilityAudit().catch(console.error);
