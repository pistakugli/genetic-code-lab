import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { DistributedManager } from '../src/core/distributed-manager';
import { resolveRuntimeTopology } from '../src/core/runtime-lane';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { OpCode } from '../src/core/pop-buffer';
import type { EvolutionConfig } from '../src/core/types';

/**
 * PHASE 65.0: MAJORITY FAMILY GAUNTLET
 * Goals:
 * 1. Verify Domain Detection for arbitrary Majority problems.
 * 2. Verify Strategy Pivot (Majority) with detected Threshold.
 * 3. Measure the honest fitness ceiling for N=5, N=7, N=9 under a stronger discrete regime.
 * 4. Ensure Zero Arithmetic Leakage in final logic.
 */

async function runMajorityGauntlet() {
    console.log("--- 65.0: MAJORITY FAMILY GAUNTLET (HONEST BASELINE) ---");
    
    const cases = [
        { n: 5, t: 3 },
        { n: 7, t: 4 },
        { n: 9, t: 5 }
    ];
    const results: Array<{ problem: string, victory: boolean, fitness: string, arithmeticLeakage: number, invalidReferences: number }> = [];

    for (const { n, t } of cases) {
        console.log(`\n>> TESTING: Majority ${n} Threshold ${t}`);
        
        const agent = new DiscoveryAgent();
        const bridge = new AsyncAgentBridge(agent);
        const problem = createMajorityProblem(n, t);
        const topology = resolveRuntimeTopology(1);
        const maxGenomeSize = n >= 9 ? 512 : 320;
        console.log(`[Topology] ${topology.topologyLabel}`);

        const evolutionConfig: Partial<EvolutionConfig> = {
            seed: 42 + n,
            population: {
                size: n >= 9 ? 450 : n >= 7 ? 300 : 200,
                eliteCount: n >= 9 ? 20 : 16,
                maxGenomeSize,
                crossoverRate: 0.85,
                mutationRate: n >= 9 ? 0.22 : 0.2,
                maxGenomeDepth: n >= 9 ? 16 : 14,
                tournamentSize: n >= 9 ? 8 : 7
            },
            maxGenerations: n >= 9 ? 140 : n >= 7 ? 100 : 60,
            parallelEvaluators: topology.parallelEvaluators,
            selectionMethod: 'lexicase',
            targetFitness: 1.0
        };

        if (topology.useDistributed) {
            const manager = new DistributedManager(topology.distributedIslands, evolutionConfig as EvolutionConfig, bridge);
            try {
                const bestBinary = await manager.start(problem, 42 + n);
                const winner = {
                    fitness: bestBinary?.fitness || 0
                };
                const victory = winner.fitness >= 1.0;
                console.log(`[Result] Victory: ${victory ? 'YES ✅' : 'NO ❌'} | Final Fitness: ${winner.fitness.toFixed(8)}`);

                let arithmeticCount = 0;
                const arithOps = [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.SIN, OpCode.COS];
                for (const opRaw of bestBinary?.opcodes || []) {
                    if (arithOps.includes(opRaw >> 24)) arithmeticCount++;
                }
                let invalidReferences = 0;
                for (const opRaw of bestBinary?.opcodes || []) {
                    const op = opRaw >> 24;
                    const data = opRaw & 0xFFFFFF;
                    if (op === OpCode.XN && data >= n) invalidReferences++;
                }
                console.log(`[Audit] Zero Arithmetic Leakage: ${arithmeticCount === 0 ? 'PASS ✅' : 'FAIL ❌'} (Found: ${arithmeticCount})`);
                console.log(`[Audit] In-Range Variable References: ${invalidReferences === 0 ? 'PASS ✅' : 'FAIL ❌'} (Found: ${invalidReferences})`);
                
                if (arithmeticCount > 0 || invalidReferences > 0) {
                    console.error("CRITICAL: Genetic Leakage Detected in logical solution!");
                    process.exit(1);
                }

                results.push({
                    problem: `Majority ${n}-${t}`,
                    victory,
                    fitness: winner.fitness.toFixed(8),
                    arithmeticLeakage: arithmeticCount,
                    invalidReferences
                });
            } finally {
                await manager.stop();
            }
            continue;
        }

        const engine = new EvolutionEngine(problem, evolutionConfig);
        engine.setDiscoveryAgent(agent);
        // @ts-ignore
        engine.agentBridge = bridge;

        try {
            let victory = false;
            const maxGenerations = n >= 9 ? 140 : n >= 7 ? 100 : 60;
            for (let gen = 0; gen <= maxGenerations; gen++) {
                await (engine as any).step(gen, 0);
                const best = engine.getBestGenome();
                if (best && best.fitness >= 1.0) {
                    console.log(`[Result] Victory at Gen ${gen}: YES ✅ | Fitness: ${best.fitness.toFixed(8)}`);
                    victory = true;
                    break;
                }
            }
            const winner = engine.getBestGenome()!;
            console.log(`[Result] Victory: ${victory ? 'YES ✅' : 'NO ❌'} | Final Fitness: ${winner.fitness.toFixed(8)}`);

            // Audit Arithmetic Leakage
            const winnerBinary = engine.getBestBinary();
            const finalFitness = winnerBinary?.fitness ?? winner.fitness;
            const finalVictory = finalFitness >= 1.0;
            console.log(`[Finalizer] Fitness: ${finalFitness.toFixed(8)} | Victory: ${finalVictory ? 'YES ✅' : 'NO ❌'}`);
            let arithmeticCount = 0;
            const arithOps = [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.SIN, OpCode.COS];
            for (const opRaw of winnerBinary?.opcodes || []) {
                if (arithOps.includes(opRaw >> 24)) arithmeticCount++;
            }
            let invalidReferences = 0;
            for (const opRaw of winnerBinary?.opcodes || []) {
                const op = opRaw >> 24;
                const data = opRaw & 0xFFFFFF;
                if (op === OpCode.XN && data >= n) invalidReferences++;
            }
            console.log(`[Audit] Zero Arithmetic Leakage: ${arithmeticCount === 0 ? 'PASS ✅' : 'FAIL ❌'} (Found: ${arithmeticCount})`);
            console.log(`[Audit] In-Range Variable References: ${invalidReferences === 0 ? 'PASS ✅' : 'FAIL ❌'} (Found: ${invalidReferences})`);
            
            if (arithmeticCount > 0 || invalidReferences > 0) {
                console.error("CRITICAL: Genetic Leakage Detected in logical solution!");
                process.exit(1);
            }

            results.push({
                problem: `Majority ${n}-${t}`,
                victory: finalVictory,
                fitness: finalFitness.toFixed(8),
                arithmeticLeakage: arithmeticCount,
                invalidReferences
            });
        } finally {
            await engine.dispose();
        }
    }

    console.log("\n--- HONEST FAMILY BASELINE SUMMARY ---");
    console.table(results);

    const solvedCount = results.filter(r => r.victory).length;
    console.log(`[Benchmark] Solved ${solvedCount}/${results.length} majority family tasks at 1.0 fitness.`);
    if (solvedCount !== results.length) {
        console.log("[Benchmark] Unsolved family members are reported as the current honest ceiling, not masked by exact-solution injection.");
    }

    console.log("\n--- PHASE 65.0 GAUNTLET COMPLETE: HONEST BASELINE CAPTURED ✅ ---");
}

runMajorityGauntlet().catch(e => {
    console.error("GAUNTLET ERROR:", e);
    process.exit(1);
});
