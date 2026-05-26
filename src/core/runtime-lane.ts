import * as os from 'os';

export const STABLE_PROOF_PARALLELISM = 8;

export type RuntimeTopology = {
    distributedIslands: number;
    parallelEvaluators: number;
    useDistributed: boolean;
    topologyLabel: string;
};

function detectAvailableParallelism(): number {
    const detected = typeof os.availableParallelism === 'function'
        ? os.availableParallelism()
        : os.cpus().length;
    return Math.max(1, detected || 1);
}

export function resolvePerformanceParallelism(stableDefault: number = STABLE_PROOF_PARALLELISM): number {
    const raw = process.env.GCL_PARALLEL_EVALUATORS;
    if (raw !== undefined) {
        const parsed = Number(raw);
        if (Number.isFinite(parsed) && parsed > 0) {
            return Math.max(1, Math.floor(parsed));
        }
    }

    return Math.max(1, Math.min(Math.max(1, stableDefault), detectAvailableParallelism()));
}

export function resolveDistributedIslands(defaultIslands: number = 1): number {
    const raw = process.env.GCL_DISTRIBUTED_ISLANDS;
    if (raw !== undefined) {
        const parsed = Number(raw);
        if (Number.isFinite(parsed) && parsed > 0) {
            return Math.max(1, Math.min(Math.floor(parsed), detectAvailableParallelism()));
        }
    }

    return Math.max(1, Math.min(Math.max(1, defaultIslands), detectAvailableParallelism()));
}

export function resolveRuntimeTopology(
    defaultIslands: number = 1,
    stableDefaultParallelism: number = STABLE_PROOF_PARALLELISM
): RuntimeTopology {
    const distributedIslands = resolveDistributedIslands(defaultIslands);
    const parallelEvaluators = resolvePerformanceParallelism(stableDefaultParallelism);
    const useDistributed = distributedIslands > 1;
    const islandLabel = distributedIslands === 1 ? 'island' : 'islands';

    return {
        distributedIslands,
        parallelEvaluators,
        useDistributed,
        topologyLabel: `${distributedIslands} ${islandLabel} x ${parallelEvaluators} evaluators`
    };
}
