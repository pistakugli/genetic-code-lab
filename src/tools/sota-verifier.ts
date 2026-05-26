import { DistributedManager } from '../core/distributed-manager';
import { EvolutionConfig } from '../core/types';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { AsyncAgentBridge } from '../core/async-agent-bridge';
import { SotaBenchmarks } from './quality-benchmarks';
import { resolveDistributedIslands, resolvePerformanceParallelism } from '../core/runtime-lane';

/**
 * PHASE 43: SOTA SCIENTIFIC VERIFIER (Statistical Fidelity Edition)
 * N=3 Trials with fixed seeds across full benchmark suite.
 */
async function runSotAVerifier() {
    console.log("=== GENETIC CODE LAB: BEYOND SOTA SCIENTIFIC VERIFIER (Phase 43) ===");

    const problems = [
        SotaBenchmarks.getF3()
    ];

    const TRIAL_COUNT = 3;
    const SEEDS = [42, 123, 777, 1337, 2024];

    const finalResults: any[] = [];

    for (const problem of problems) {
        console.log(`\n--- Auditing Problem: ${problem.name} ---`);
        
        const modes = [
            //{ islands: 1, pop: 10000, agent: false, label: "ISO_10K" },
            { islands: 5, pop: 10000, agent: true, label: "DIST_50K" },
        ];

        const problemStats: any = { problem: problem.name };

        for (const mode of modes) {
            const trials: any[] = [];
            console.log(`[Action] Starting ${mode.label} Matrix (${TRIAL_COUNT} Trials)...`);

            for (let i = 0; i < TRIAL_COUNT; i++) {
                const res = await runExperiment(problem, mode, SEEDS[i]);
                trials.push(res);
                console.log(`  > Trial ${i+1}/${TRIAL_COUNT} [Seed: ${SEEDS[i]}] | Best Fit: ${res.bestFit.toFixed(4)} | Gens: ${res.gens}`);
            }

            const meanFit = trials.reduce((a, b) => a + b.bestFit, 0) / TRIAL_COUNT;
            const variance = trials.reduce((a, b) => a + Math.pow(b.bestFit - meanFit, 2), 0) / TRIAL_COUNT;
            const stdDev = Math.sqrt(variance);
            const successRate = (trials.filter(t => t.bestFit >= 0.95).length / TRIAL_COUNT) * 100;
            const avgGens = trials.reduce((a, b) => a + b.gens, 0) / TRIAL_COUNT;

            problemStats[mode.label] = `${meanFit.toFixed(4)} ±${stdDev.toFixed(4)}`;
            problemStats[`${mode.label}_Succ`] = `${successRate.toFixed(0)}%`;
        }

        finalResults.push(problemStats);
    }

    console.log("\n=== PHASE 43 STATISTICAL SUMMARY (N=3) ===");
    console.table(finalResults);
}

async function runExperiment(problem: any, params: any, seed: number) {
    const isF3 = problem.name === 'F3-Rational';
    const baseGens = isF3 ? 100 : 150; // Give F3 a split budget
    const distributedIslands = resolveDistributedIslands(params.islands);
    const islandParallelEvaluators = resolvePerformanceParallelism(1);
    
    const config: EvolutionConfig = {
        maxGenerations: baseGens,
        targetFitness: 1.0,
        stagnationLimit: 25,
        parallelEvaluators: islandParallelEvaluators,
        seed: seed, 
        population: {
            size: params.pop, 
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
    
    // STAGE 1: BASE MODEL
    if (isF3) console.log(`    [Stage 1] Evolving Base Model (${baseGens} gens)...`);
    const manager = new DistributedManager(distributedIslands, config, bridge);
    await manager.start(problem, seed);
    
    // Controlled Evaluation Window (120s for DIST_50K to ensure generation depth)
    const waitTime = distributedIslands > 1 ? 120000 : 25000;
    await new Promise(r => setTimeout(r, waitTime));
    
    const baseResult = manager.getBestOfRun();
    const finalStats = manager.getGlobalAnalytics();
    await manager.stop();

    if (!isF3) {
        const bestFit = Math.max(...finalStats.map(s => s.bestFitness));
        return { bestFit: isNaN(bestFit) ? 0 : bestFit, gens: Math.max(...finalStats.map(s => s.gen)) };
    }

    // STAGE 2: RESIDUAL MODEL (Only for F3)
    if (!baseResult) {
        console.error(`    [Error] Stage 1 failed to produce a baseResult (fitness still null).`);
        return { bestFit: 0, gens: baseGens };
    }
    
    // Phase 46/53: Numerical Stability Guard (Residual Cleaning)
    const finiteTargets = problem.testCases.map((tc: any) => tc.expectedOutput).filter((o: any) => isFinite(o));
    const targetMin = Math.min(...finiteTargets);
    const targetMax = Math.max(...finiteTargets);
    const cleanedPredictions = baseResult.predictions.map((p: number) => {
        if (isNaN(p) || !isFinite(p)) return 0;
        return Math.max(targetMin, Math.min(targetMax, p));
    });

    console.log(`    [Stage 2] Starting Residual Evolution (Targeting Base Error: ${baseResult.fitness.toFixed(4)})...`);
    
    // Create Residual Problem
    const residualTestCases = problem.testCases.map((tc: any, i: number) => ({
        input: tc.input,
        expectedOutput: tc.expectedOutput - cleanedPredictions[i],
        weight: tc.weight
    }));

    const residualProblem = {
        ...problem,
        name: `F3-Residual`,
        testCases: residualTestCases
    };

    const resConfig = { ...config, maxGenerations: 75, seed: seed + 1000 };
    const resManager = new DistributedManager(distributedIslands, resConfig, bridge);
    await resManager.start(residualProblem, seed + 1000);
    
    await new Promise(r => setTimeout(r, waitTime));
    
    const resResult = resManager.getBestOfRun();
    const resStats = resManager.getGlobalAnalytics();
    await resManager.stop();

    // PHASE 50/52: HONEST COMBINED FITNESS (Sanitized)
    // Calculating real fitness (Base + Residual) on the Original Problem
    let finalFit = baseResult.fitness;
    if (resResult) {
        const combinedPredictions = new Float64Array(problem.testCases.length);
        let totalMAE = 0;
        
        // Phase 52: Sanitize Residuals (Numerical Stability Guard)
        const cleanedResiduals = resResult.predictions.map((p: number) => {
            if (isNaN(p) || !isFinite(p)) return 0;
            return p;
        });

        for (let i = 0; i < problem.testCases.length; i++) {
            // Model_Total = Model_Base + Model_Residual
            combinedPredictions[i] = cleanedPredictions[i] + cleanedResiduals[i];
            totalMAE += Math.abs(problem.testCases[i].expectedOutput - combinedPredictions[i]);
        }
        const avgMAE = totalMAE / problem.testCases.length;
        finalFit = 1.0 / (1.0 + avgMAE);
        
        console.log(`    [Forensic] Base Fit: ${baseResult.fitness.toFixed(4)} | Residual Local Fit: ${resResult.fitness.toFixed(4)} | HONEST COMBINED FIT: ${finalFit.toFixed(4)}`);
    }

    const totalGens = Math.max(...finalStats.map(s => s.gen)) + Math.max(...resStats.map(s => s.gen));

    return {
        bestFit: isNaN(finalFit) ? 0 : finalFit,
        gens: totalGens
    };
}

runSotAVerifier().catch(console.error);
