/**
 * PHASE 36: SURGICAL PARETO VERIFIER
 * Rewritten to use the actual SPEA2 dominance API (dominates + calculateSPEA2Fitness)
 * instead of the removed EvolutionEngine.calculateParetoFronts method.
 */
import { Genome } from '../src/core/types';
import { dominates, calculateSPEA2Fitness } from '../src/core/spea2';

function makeGenome(id: string, objectives: number[]): Genome {
    return {
        id,
        genes: [],
        fitness: 0,
        generation: 0,
        age: 0,
        multiObjectiveFitness: {
            values: objectives, // [error, complexity] — both minimized
            rawFitness: 0,
            density: 0,
            score: 0
        }
    };
}

async function verifyPareto() {
    console.log("--- PHASE 36: SURGICAL PARETO VERIFIER ---");

    /**
     * SEEDING STRATEGY (objectives = [error, complexity], both minimized):
     * A: [0.0, 50]  -> Non-dominated (Best fitness/lowest error)
     * B: [0.5, 10]  -> Non-dominated (Balanced)
     * C: [0.9, 1]   -> Non-dominated (Best complexity)
     * D: [0.5, 50]  -> DOMINATED by A (lower error) and B (lower complexity)
     * E: [1.0, 100] -> DOMINATED by everyone
     */
    const A = makeGenome('A', [0.0, 50]);
    const B = makeGenome('B', [0.5, 10]);
    const C = makeGenome('C', [0.9, 1]);
    const D = makeGenome('D', [0.5, 50]);
    const E = makeGenome('E', [1.0, 100]);

    const individuals = [A, B, C, D, E];

    console.log("[Test] Seeding objective profiles A-E...");
    console.log("[Test] Computing SPEA2 fitness (dominance + density)...");

    // Run full SPEA2 fitness calculation
    calculateSPEA2Fitness(individuals, 1);

    let success = true;

    // --- Dominance verification ---

    // A should dominate D (A has strictly better error, same complexity)
    if (dominates(A, D)) {
        console.log("[PASS] A dominates D (lower error, same complexity).");
    } else {
        console.error("[FAIL] A should dominate D.");
        success = false;
    }

    // B should dominate D (B has same error but better complexity)
    if (dominates(B, D)) {
        console.log("[PASS] B dominates D (same error, lower complexity).");
    } else {
        console.error("[FAIL] B should dominate D.");
        success = false;
    }

    // A should dominate E
    if (dominates(A, E)) {
        console.log("[PASS] A dominates E.");
    } else {
        console.error("[FAIL] A should dominate E.");
        success = false;
    }

    // A should NOT dominate B (A has better error but worse complexity)
    if (!dominates(A, B)) {
        console.log("[PASS] A does NOT dominate B (trade-off exists).");
    } else {
        console.error("[FAIL] A should NOT dominate B.");
        success = false;
    }

    // B should NOT dominate C
    if (!dominates(B, C)) {
        console.log("[PASS] B does NOT dominate C (trade-off exists).");
    } else {
        console.error("[FAIL] B should NOT dominate C.");
        success = false;
    }

    // --- SPEA2 Rank verification ---
    // Non-dominated individuals (A, B, C) should have rawFitness = 0
    // Dominated individuals (D, E) should have rawFitness > 0

    const frontierSet = individuals.filter(i => i.multiObjectiveFitness!.rawFitness === 0);
    const frontierIds = frontierSet.map(i => i.id).sort();

    if (frontierIds.join(',') === 'A,B,C') {
        console.log("[PASS] Non-dominated frontier = {A, B, C} (rawFitness = 0).");
    } else {
        console.error(`[FAIL] Frontier mismatch: ${frontierIds.join(',')} (Expected A,B,C)`);
        success = false;
    }

    // D should be dominated (rawFitness > 0)
    if (D.multiObjectiveFitness!.rawFitness! > 0) {
        console.log(`[PASS] D is dominated (rawFitness = ${D.multiObjectiveFitness!.rawFitness}).`);
    } else {
        console.error(`[FAIL] D should be dominated (rawFitness = ${D.multiObjectiveFitness!.rawFitness}).`);
        success = false;
    }

    // E should be more deeply dominated than D
    if (E.multiObjectiveFitness!.rawFitness! > D.multiObjectiveFitness!.rawFitness!) {
        console.log(`[PASS] E is more deeply dominated than D (rawFitness E=${E.multiObjectiveFitness!.rawFitness} > D=${D.multiObjectiveFitness!.rawFitness}).`);
    } else {
        console.error(`[FAIL] E should be more dominated than D (E=${E.multiObjectiveFitness!.rawFitness}, D=${D.multiObjectiveFitness!.rawFitness}).`);
        success = false;
    }

    if (success) console.log("--- PARETO FRONTIER LOGIC VERIFIED ---");
    else process.exit(1);
}

verifyPareto().catch(console.error);
