import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { resolvePerformanceParallelism } from '../src/core/runtime-lane';
import { createMajorityProblem, createMuxProblem } from '../src/solvers/boolean-library';
import { createBooleanParityProblem } from '../src/solvers/boolean-parity';


/**
 * PHASE 64.2: MULTICLASS BOOLEAN GAUNTLET (Honest Baseline)
 * Measures signal-driven autonomy across Majority, Parity, and MUX problems
 * without treating sub-1.0 results as a synthetic test failure.
 */

async function runGauntlet() {
    console.log("--- 64.2: MULTICLASS BOOLEAN GAUNTLET (HONEST BASELINE) ---");
    const parallelEvaluators = resolvePerformanceParallelism();

    const agent = new DiscoveryAgent();
    // Wrap in bridge for async simulation, though we use direct agent for the gauntlet
    const bridge = new AsyncAgentBridge(agent);

    const problems = [
        { name: "Majority-5 (T3)", solver: () => createMajorityProblem(5, 3), expectedStrategy: "MAJORITY_BOOTSTRAP" },
        { name: "Parity-5", solver: () => createBooleanParityProblem(5), expectedStrategy: "PARITY_BOOTSTRAP" },
        { name: "MUX-6 (2 bit select)", solver: () => createMuxProblem(2), expectedStrategy: "MUX_BOOTSTRAP" }
    ];

    const results: Array<{ problem: string, victory: boolean, fitness: string, arithmeticLeakage: number, invalidReferences: number }> = [];
    let purityFailure = false;

    for (const p of problems) {
        console.log(`\n>> TESTING: ${p.name}`);
        const problem = p.solver();
        const engine = new EvolutionEngine(problem, {
            population: {
                size: 200,
                eliteCount: 10,
                maxGenomeSize: 128,
                crossoverRate: 0.8,
                mutationRate: 0.2, // Boosted for discrete domain
                maxGenomeDepth: 15,
                tournamentSize: 7
            },
            maxGenerations: 60,
            parallelEvaluators,
            targetFitness: 1.0
        });

        engine.setDiscoveryAgent(agent);
        // @ts-ignore (Accessing internal bridge for data sync)
        engine.agentBridge = bridge;

        let victory = false;
        let pResult: any = null;

        for (let gen = 0; gen <= 60; gen++) {
            await (engine as any).step(gen, 0);
            pResult = engine.getBestGenome();

            if (pResult && pResult.fitness >= 1.0) {
                victory = true;
                break;
            }
        }

        const best = engine.getBestGenome()!;
        console.log(`[Result] Victory: ${victory ? 'YES ✅' : 'NO ❌'} | Fitness: ${best.fitness.toFixed(8)}`);

        // AUDIT: Arithmetic Leakage (Zonal Isolation Check)
        let arithCount = 0;
        const arithOps = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'pow', 'sqrt', 'neg', 'abs', 'exp', 'log'];
        for (let i = 0; i < best.genes.length; i++) {
            if (best.genes[i].type === 'function' && arithOps.includes(best.genes[i].operation)) {
                arithCount++;
                console.log(`LEAK: ${best.genes[i].operation}`);
            }
        }

        const bestBinary = engine.getBestBinary();
        let invalidReferences = 0;
        for (const opRaw of bestBinary?.opcodes || []) {
            const op = opRaw >>> 24;
            const data = opRaw & 0xFFFFFF;
            if (op === 10 && data >= (problem.testCases[0].input as number[]).length) invalidReferences++;
        }

        console.log(`[Audit] Zero Arithmetic Leakage: ${arithCount === 0 ? 'PASS ✅' : 'FAIL ❌'} (Found: ${arithCount})`);
        console.log(`[Audit] In-Range Variable References: ${invalidReferences === 0 ? 'PASS ✅' : 'FAIL ❌'} (Found: ${invalidReferences})`);

        results.push({
            problem: p.name,
            victory,
            fitness: best.fitness.toFixed(8),
            arithmeticLeakage: arithCount,
            invalidReferences
        });

        if (arithCount > 0 || invalidReferences > 0) purityFailure = true;
    }

    console.log("\n--- HONEST BASELINE SUMMARY ---");
    console.table(results);

    const solvedCount = results.filter(r => r.victory).length;
    console.log(`[Benchmark] Solved ${solvedCount}/${results.length} tasks at 1.0 fitness.`);
    if (solvedCount !== results.length) {
        console.log("[Benchmark] Sub-1.0 tasks are reported as the real baseline, not masked by exact-solution injection.");
    }

    if (purityFailure) {
        console.error("[FAILURE] Arithmetic leakage or invalid variable references detected in final logic.");
        process.exit(1);
    }

    console.log("\n--- PHASE 64.2 GAUNTLET: HONEST BASELINE CAPTURED ✅ ---");
    process.exit(0);
}

runGauntlet().catch(e => {
    console.error("CRITICAL GAUNTLET FAILURE:", e.message);
    process.exit(1);
});
