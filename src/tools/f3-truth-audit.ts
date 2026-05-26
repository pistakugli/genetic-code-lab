import { EvolutionConfig, ProblemDefinition } from '../core/types';
import { EvolutionEngine } from '../core/evolution';
import { SotaBenchmarks } from './quality-benchmarks';
import { DiscoveryAgent } from '../agents/discovery-agent';
import { AsyncAgentBridge } from '../core/async-agent-bridge';

/**
 * PHASE 51: F3 TRUTH AUDIT (The Forensic Proof)
 * Strictly re-evaluates the 0.5976 claim using honest global fitness math.
 */
async function runTruthAudit() {
    console.log("=== GENETIC CODE LAB: F3 TRUTH AUDIT (Seed 777) ===");

    const problem = SotaBenchmarks.getF3();
    const seed = 777; // The 'Gold' seed
    
    const config: EvolutionConfig = {
        maxGenerations: 100,
        targetFitness: 1.0,
        stagnationLimit: 25,
        parallelEvaluators: 0,
        seed: seed, 
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
    
    // ---------------------------------------------------------
    // STAGE 1: BASE MODEL
    // ---------------------------------------------------------
    console.log(`\n[Stage 1] Replaying Base Model (Seed: ${seed})...`);
    const engine = new EvolutionEngine(problem, config);
    engine.setDiscoveryAgent(agent);
    
    let bestBaseFound: any = null;
    let baseGens = 0;
    
    for (let gen = 0; gen < 100; gen++) {
        // Step evaluation manually to observe 'Injection' and 'Extraction'
        await (engine as any).step(gen, 0); 
        const stats = engine.getBestGenome();
        if (stats && (!bestBaseFound || stats.fitness > bestBaseFound.fitness)) {
            bestBaseFound = stats;
            baseGens = gen;
        }
        if (gen % 20 === 0) {
            const currentBest = engine.getBestGenome();
            console.log(`  Gen ${gen} | Fit: ${currentBest?.fitness.toFixed(6)} | Size: ${bestBaseFound?.genes?.length || 0}`);
        }
    }

    const baseResult = {
        fitness: bestBaseFound.fitness,
        predictions: engine.getBestPredictions()!
    };
    
    console.log(`[Stage 1 Result] Honest Fit: ${baseResult.fitness.toFixed(6)} | Target Error for Stage 2: ${(1/baseResult.fitness - 1).toFixed(6)} MAE`);

    // ---------------------------------------------------------
    // STAGE 2: RESIDUAL MODEL
    // ---------------------------------------------------------
    console.log(`\n[Stage 2] Starting Residual Evolution (Targeting Base Base Fit: ${baseResult.fitness.toFixed(4)})...`);
    
    // Create Residual Problem
    const residualTestCases = problem.testCases.map((tc: any, i: number) => ({
        input: tc.input,
        expectedOutput: tc.expectedOutput - baseResult.predictions[i],
        weight: tc.weight
    }));

    const residualProblem = {
        ...problem,
        name: `F3-Residual`,
        testCases: residualTestCases
    };

    const resConfig = { ...config, maxGenerations: 75, seed: seed + 1000 };
    const resEngine = new EvolutionEngine(residualProblem, resConfig);
    resEngine.setDiscoveryAgent(agent);
    
    let bestResidualFound: any = null;
    
    for (let gen = 0; gen < 75; gen++) {
        await (resEngine as any).step(gen, 0);
        const stats = resEngine.getBestGenome();
        if (stats && (!bestResidualFound || stats.fitness > bestResidualFound.fitness)) {
            bestResidualFound = stats;
        }
        if (gen % 20 === 0) {
            console.log(`  Gen ${gen} | Residual Local Fit: ${bestResidualFound?.fitness.toFixed(6)}`);
        }
    }

    const resResult = {
        fitness: bestResidualFound.fitness,
        predictions: resEngine.getBestPredictions()!
    };

    // ---------------------------------------------------------
    // THE HONEST FINAL VERDICT
    // ---------------------------------------------------------
    console.log("\n=== THE HONEST FINAL VERDICT (Base + Residual) ===");
    
    const combinedPredictions = new Float64Array(problem.testCases.length);
    let totalMAE = 0;
    let weightSum = 0;
    for (let i = 0; i < problem.testCases.length; i++) {
        // Model_Total = Model_Base + Model_Residual
        combinedPredictions[i] = baseResult.predictions[i] + resResult.predictions[i];
        const weight = problem.testCases[i].weight || 1.0;
        totalMAE += Math.abs(problem.testCases[i].expectedOutput - combinedPredictions[i]) * weight;
        weightSum += weight;
    }
    const avgMAE = totalMAE / weightSum;
    const globalHonestFit = 1.0 / (1.0 + avgMAE);
    
    console.log(`  [Data] Base Model Fitness: ${baseResult.fitness.toFixed(6)}`);
    console.log(`  [Data] Residual Model Local Fitness: ${resResult.fitness.toFixed(6)} (This was the '0.59' bubble)`);
    console.log(`  [Truth] HONEST GLOBAL FITNESS: ${globalHonestFit.toFixed(6)}`);
    console.log(`\nConclusion: ${globalHonestFit > baseResult.fitness ? 'SUCCESS: Residual Stage improved global fitness.' : 'FAIL: Residual Stage was illusory.'}`);
}

runTruthAudit().catch(console.error);
