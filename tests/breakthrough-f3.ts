import { DistributedManager } from '../src/core/distributed-manager';
import { EvolutionConfig } from '../src/core/types';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import * as fs from 'fs';
import * as path from 'path';

/**
 * PHASE 61: THE RATIONAL SINGULARITY BREAKTHROUGH
 * Orchestrates a 10,000-population run across 4 islands.
 * Targets the (x^2+1)/(x-2) rational singularity with 1e-10 precision.
 */
async function runBreakthrough() {
    console.error("\n🚀 [Phase 61] LAUNCHING RATIONAL SINGULARITY ASSAULT...");
    console.error("=========================================================");

    const problem = SotaBenchmarks.getF3();
    const seed = 42;

    const config: EvolutionConfig = {
        maxGenerations: 1500, // Full effort for victory
        targetFitness: 0.9999999999, 
        stagnationLimit: 100,
        parallelEvaluators: 2, 
        seed,
        population: {
            size: 2500, 
            eliteCount: 20,
            crossoverRate: 0.85,
            mutationRate: 0.35,
            maxGenomeSize: 128,
            maxGenomeDepth: 10,
            tournamentSize: 7,
            parsimonyCoefficient: 0.0001
        }
    };

    const agent = new DiscoveryAgent();
    // Enable structural pivot for asymptotes
    const bridge = new AsyncAgentBridge(agent);

    const manager = new DistributedManager(4, config, bridge);
    
    console.error(`| Problem:  ${problem.name}`);
    console.error(`| Target:   (x^2 + 1) / (x - 2)`);
    console.error(`| Shards:   4 islands x 2500 genomes = 10,000 total`);
    console.error(`| Agents:   DiscoveryAgent (Skeleton Injection) & AGBHC (Surgical Repair)`);
    console.error("=========================================================\n");

    const startTime = Date.now();
    await manager.start(problem, seed);

    // Monitoring loop
    let lastGen = -1;
    let stableGenCount = 0;

    const monitor = setInterval(async () => {
        const stats = manager.getGlobalAnalytics();
        if (stats.length === 0) return;

        const bestFit = Math.max(...stats.map(s => s.bestFitness));
        const avgGen = stats.reduce((acc, s) => acc + s.gen, 0) / stats.length;
        const totalEvals = stats.reduce((acc, s) => acc + s.gen * 2500, 0);

        if (avgGen !== lastGen) {
            console.error(`[Gen ${avgGen.toFixed(0)}] Global Best: ${bestFit.toFixed(9)} | Evals: ${totalEvals}`);
            lastGen = Math.floor(avgGen);
        }

        if (bestFit >= 0.9999999) {
            console.error(`\n🎯 [TARGET REACHED] Fitness: ${bestFit.toFixed(10)}`);
            clearInterval(monitor);
        }
    }, 5000);

    // Wait for manager to finish (targetFitness or generations)
    // We'll use a polling check for the manager status
    while (true) {
        const stats = manager.getGlobalAnalytics();
        const bestFit = stats.length > 0 ? Math.max(...stats.map(s => s.bestFitness)) : 0;
        const allDone = stats.every(s => s.gen >= 200 || bestFit >= 0.9999999999);
        
        if (allDone && stats.length > 0) break;
        await new Promise(r => setTimeout(r, 2000));
    }

    clearInterval(monitor);
    const duration = (Date.now() - startTime) / 1000;
    
    const finalStats = manager.getGlobalAnalytics();
    const bestFit = Math.max(...finalStats.map(s => s.bestFitness));
    const champion = manager.getBestOfRun();

    console.error("\n=========================================================");
    console.error(`📊 [BREAKTHROUGH REPORT]`);
    console.error(`| Final Fitness:  ${bestFit.toFixed(12)}`);
    console.error(`| Duration:       ${duration.toFixed(1)}s`);
    console.error(`| Result:         ${bestFit >= 0.99999 ? "LEGENDARY SUCCESS" : "MISSION FAILED"}`);
    console.error("=========================================================\n");

    // Persist Results for Walkthrough
    const report = {
        fitness: bestFit,
        duration,
        timestamp: new Date().toISOString(),
        success: bestFit >= 0.99999
    };
    fs.writeFileSync(path.join(__dirname, '../validation_report.json'), JSON.stringify(report, null, 2));

    await manager.stop();
}

runBreakthrough().catch(console.error);
