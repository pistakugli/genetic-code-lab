import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { DistributedManager } from '../src/core/distributed-manager';
import { resolveRuntimeTopology } from '../src/core/runtime-lane';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { auditThresholdCases } from '../src/solvers/threshold-audit';
import { resolveThresholdProofLane } from '../src/solvers/threshold-proof-lane';
import type { EvolutionConfig } from '../src/core/types';

type Mode = 'none' | 'agent' | 'bridge';
type ShapeRow = { ones: number; total: number; accuracy: number };

const originalLog = console.log.bind(console);
const originalError = console.error.bind(console);
const noisyPrefixes = ['[Agent-Bridge]', '[Director]', '[EvolutionEngine]'];

if (process.env.BENCHMARK_VERBOSE !== '1') {
    console.log = (...args: unknown[]) => {
        const first = typeof args[0] === 'string' ? args[0] : '';
        if (noisyPrefixes.some(prefix => first.startsWith(prefix))) return;
        originalLog(...args);
    };
    console.error = (...args: unknown[]) => {
        const first = typeof args[0] === 'string' ? args[0] : '';
        if (noisyPrefixes.some(prefix => first.startsWith(prefix))) return;
        originalError(...args);
    };
}

function parseList(arg: string | undefined, fallback: number[]): number[] {
    if (!arg) return fallback;
    return arg.split(',').map(value => Number(value.trim())).filter(Number.isFinite);
}

function mean(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    if (sorted.length === 0) return 0;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function buildShapeSummary(
    problem: ReturnType<typeof createMajorityProblem>,
    threshold: number,
    opcodes: Uint32Array,
    constants: Float64Array
): {
    byWeight: ShapeRow[];
    exactAccuracy: number;
    boundary: { total: number; fp4: number; fn5: number };
} {
    const audit = auditThresholdCases(problem.testCases, threshold, opcodes, constants);

    return {
        byWeight: audit.byWeight.map(row => ({
            ones: row.ones,
            total: row.total,
            accuracy: Number(row.accuracy.toFixed(4))
        })),
        exactAccuracy: Number(audit.exactAccuracy.toFixed(8)),
        boundary: audit.boundary
    };
}

async function runMode(seed: number, mode: Mode, n: number, threshold: number, generations: number) {
    const problem = createMajorityProblem(n, threshold);
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    const topology = resolveRuntimeTopology(1);
    const proofLane = resolveThresholdProofLane();
    const includeShape = process.env.BENCHMARK_SHAPE === '1';
    const maxGenomeSize = n >= 9 ? 512 : 320;
    const evolutionConfig: Partial<EvolutionConfig> = {
        seed,
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
        selectionMethod: 'lexicase',
        targetFitness: 1.0
    };

    if (topology.useDistributed) {
        const manager = new DistributedManager(topology.distributedIslands, evolutionConfig as EvolutionConfig, bridge);
        try {
            const bestBinary = await manager.start(problem, seed);
            if (!bestBinary || !bestBinary.opcodes || !bestBinary.constants) {
                throw new Error(`No distributed winner for seed=${seed} mode=${mode}`);
            }
            const audit = auditThresholdCases(problem.testCases, threshold, bestBinary.opcodes, bestBinary.constants);
            const shape = includeShape
                ? buildShapeSummary(problem, threshold, bestBinary.opcodes, bestBinary.constants)
                : null;

            return {
                seed,
                mode,
                proofLane,
                parallelEvaluators: topology.parallelEvaluators,
                distributedIslands: topology.distributedIslands,
                topology: topology.topologyLabel,
                fitness: Number(bestBinary.fitness.toFixed(8)),
                exactAccuracy: Number(audit.exactAccuracy.toFixed(8)),
                length: bestBinary.opcodes.length,
                lowerBoundary: Number(audit.lowerBoundary.toFixed(4)),
                upperBoundary: Number(audit.upperBoundary.toFixed(4)),
                boundary: audit.boundary,
                ...(shape ? { shape } : {})
            };
        } finally {
            await manager.stop();
        }
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
        if (!bestBinary || !bestGenome) {
            throw new Error(`No best genome for seed=${seed} mode=${mode}`);
        }
        const composeDebug = includeShape ? (engine as any).getComposeDebugStats?.() : undefined;
        const audit = auditThresholdCases(problem.testCases, threshold, bestBinary.opcodes, bestBinary.constants);
        const shape = includeShape
            ? buildShapeSummary(problem, threshold, bestBinary.opcodes, bestBinary.constants)
            : null;

        return {
            seed,
            mode,
            proofLane,
            parallelEvaluators: topology.parallelEvaluators,
            distributedIslands: topology.distributedIslands,
            topology: topology.topologyLabel,
            fitness: Number(bestBinary.fitness.toFixed(8)),
            exactAccuracy: Number(audit.exactAccuracy.toFixed(8)),
            length: bestBinary.opcodes.length,
            lowerBoundary: Number(audit.lowerBoundary.toFixed(4)),
            upperBoundary: Number(audit.upperBoundary.toFixed(4)),
            boundary: audit.boundary,
            ...(shape ? { shape } : {}),
            ...(composeDebug ? { composeDebug } : {})
        };
    } finally {
        await engine.dispose();
    }
}

async function main() {
    const n = Number(process.argv[2] || 9);
    const threshold = Number(process.argv[3] || 5);
    const generations = Number(process.argv[4] || 60);
    const seeds = parseList(process.argv[5], [1337, 2026, 4242]);
    const modes = ((process.argv[6] || 'none,bridge').split(',').map(value => value.trim().toLowerCase()) as Mode[])
        .filter((mode): mode is Mode => mode === 'none' || mode === 'agent' || mode === 'bridge');

    const perMode: Record<string, Array<Awaited<ReturnType<typeof runMode>>>> = {};

    for (const mode of modes) {
        perMode[mode] = [];
        for (const seed of seeds) {
            perMode[mode].push(await runMode(seed, mode, n, threshold, generations));
        }
    }

    const summary = Object.fromEntries(
        Object.entries(perMode).map(([mode, rows]) => [
            mode,
            {
                meanFitness: Number(mean(rows.map(row => row.fitness)).toFixed(8)),
                medianFitness: Number(median(rows.map(row => row.fitness)).toFixed(8)),
                meanLowerBoundary: Number(mean(rows.map(row => row.lowerBoundary)).toFixed(4)),
                meanUpperBoundary: Number(mean(rows.map(row => row.upperBoundary)).toFixed(4)),
                rows
            }
        ])
    );

    originalLog(JSON.stringify({
        n,
        threshold,
        generations,
        seeds,
        modes,
        summary
    }, null, 2));
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
