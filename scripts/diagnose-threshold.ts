import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { DistributedManager } from '../src/core/distributed-manager';
import { resolveRuntimeTopology } from '../src/core/runtime-lane';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { OpCode } from '../src/core/pop-buffer';
import { auditThresholdCases } from '../src/solvers/threshold-audit';
import { resolveThresholdProofLane } from '../src/solvers/threshold-proof-lane';
import type { EvolutionConfig, TestCase } from '../src/core/types';

async function main() {
    const n = Number(process.argv[2] || 9);
    const threshold = Number(process.argv[3] || Math.floor(n / 2) + 1);
    const generations = Number(process.argv[4] || (n >= 9 ? 140 : n >= 7 ? 100 : 60));
    const mode = (process.argv[5] || 'bridge').toLowerCase();
    const seedArg = process.argv[6];
    const seed = seedArg !== undefined ? Number(seedArg) : undefined;
    const selectionMethodArg = (process.argv[7] || 'lexicase').toLowerCase();
    const selectionMethod = (
        selectionMethodArg === 'epsilon-lexicase'
        || selectionMethodArg === 'lexicase'
        || selectionMethodArg === 'tournament'
    ) ? selectionMethodArg : 'lexicase';
    const dumpMode = (process.argv[8] || '').toLowerCase();
    const dumpBoundaryErrors = dumpMode === 'dump-errors';
    const topology = resolveRuntimeTopology(1);
    const useDistributed = mode === 'distributed' || mode === 'super' || topology.useDistributed;
    const proofLane = resolveThresholdProofLane();

    const problem = createMajorityProblem(n, threshold);
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    const maxGenomeSize = n >= 9 ? 512 : 320;
    const evolutionConfig: Partial<EvolutionConfig> = {
        seed: Number.isFinite(seed) ? seed : undefined,
        population: {
            size: n >= 9 ? 450 : n >= 7 ? 300 : 200,
            eliteCount: n >= 9 ? 20 : 16,
            maxGenomeSize,
            crossoverRate: 0.85,
            mutationRate: n >= 9 ? 0.22 : 0.2,
            maxGenomeDepth: n >= 9 ? 16 : 14,
            tournamentSize: n >= 9 ? 8 : 7
        },
        maxGenerations: generations,
        parallelEvaluators: topology.parallelEvaluators,
        thresholdSelectionPolicy: proofLane.selectionPolicy,
        selectionMethod,
        targetFitness: 1.0
    };

    const emitAudit = (
        bestBinary: { opcodes: Uint32Array; constants: Float64Array; fitness: number },
        length: number
    ) => {
        const audit = auditThresholdCases(problem.testCases as TestCase<number[], number>[], threshold, bestBinary.opcodes, bestBinary.constants);

        const histogram: Record<string, number> = {};
        for (let i = 0; i < bestBinary.opcodes.length; i++) {
            const op = bestBinary.opcodes[i] >>> 24;
            const key = OpCode[op] || `OP_${op}`;
            histogram[key] = (histogram[key] || 0) + 1;
        }

        console.log(JSON.stringify({
            n,
            threshold,
            mode,
            seed: Number.isFinite(seed) ? seed : null,
            selectionMethod,
            proofLane,
            parallelEvaluators: topology.parallelEvaluators,
            distributedIslands: topology.distributedIslands,
            topology: topology.topologyLabel,
            fitness: Number((bestBinary.fitness || 0).toFixed(8)),
            exactAccuracy: Number(audit.exactAccuracy.toFixed(8)),
            length,
            lowerBoundary: Number(audit.lowerBoundary.toFixed(4)),
            upperBoundary: Number(audit.upperBoundary.toFixed(4)),
            boundary: audit.boundary,
            byWeight: audit.byWeight.map(row => ({
                ones: row.ones,
                total: row.total,
                accuracy: Number(row.accuracy.toFixed(4))
            })),
            histogram,
            ...(dumpBoundaryErrors ? { boundaryErrors: audit.boundaryErrors } : {})
        }, null, 2));
    };

    if (useDistributed) {
        const manager = new DistributedManager(topology.distributedIslands, evolutionConfig as any, bridge);
        try {
        const bestBinary = await manager.start(problem, Number.isFinite(seed) ? seed : 42);
        if (!bestBinary || !bestBinary.opcodes || !bestBinary.constants) {
            throw new Error('Distributed run produced no winner.');
        }
        emitAudit(bestBinary, bestBinary.opcodes.length);
        } finally {
            await manager.stop();
        }
        return;
    }

    const engine = new EvolutionEngine(problem, evolutionConfig);

    if (mode !== 'none') {
        engine.setDiscoveryAgent(agent);
        if (mode === 'bridge') {
            (engine as any).agentBridge = bridge;
        }
    }

    try {
        for (let gen = 0; gen <= generations; gen++) {
            await (engine as any).step(gen, 0);
        }

        const bestBinary = engine.getBestBinary();
        const bestGenome = engine.getBestGenome();
        if (!bestBinary || !bestGenome) throw new Error('No winner produced.');
        emitAudit(bestBinary, bestBinary.opcodes.length);
    } finally {
        await engine.dispose();
    }
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
