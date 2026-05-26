import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { resolveRuntimeTopology } from '../src/core/runtime-lane';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { OpCode } from '../src/core/pop-buffer';

type WeightRow = { total: number; correct: number };

type CheckpointAudit = {
    gen: number;
    fitness: number;
    length: number;
    stagnationCount: number;
    weight4Accuracy: number;
    weight5Accuracy: number;
    boundaryErrors: number;
};

function evalBinary(opcodes: Uint32Array, constants: Float64Array, input: number[]): number {
    const stack: number[] = [];
    for (let i = 0; i < opcodes.length; i++) {
        const raw = opcodes[i];
        const op = raw >>> 24;
        const data = raw & 0xFFFFFF;
        switch (op) {
            case OpCode.XN:
                stack.push(input[data] || 0);
                break;
            case OpCode.CONST:
                stack.push(constants[data] || 0);
                break;
            case OpCode.AND: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 && b > 0.5 ? 1 : 0);
                break;
            }
            case OpCode.OR: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 || b > 0.5 ? 1 : 0);
                break;
            }
            case OpCode.NAND: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 && b > 0.5 ? 0 : 1);
                break;
            }
            case OpCode.NOR: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                stack.push(a > 0.5 || b > 0.5 ? 0 : 1);
                break;
            }
            case OpCode.XOR: {
                const b = stack.pop() || 0;
                const a = stack.pop() || 0;
                const aBit = a > 0.5 ? 1 : 0;
                const bBit = b > 0.5 ? 1 : 0;
                stack.push(aBit + bBit === 1 ? 1 : 0);
                break;
            }
            case OpCode.ITE: {
                const whenFalse = stack.pop() || 0;
                const whenTrue = stack.pop() || 0;
                const cond = stack.pop() || 0;
                stack.push(cond > 0.5 ? whenTrue : whenFalse);
                break;
            }
            default:
                stack.push(0);
                break;
        }
    }
    return stack.length > 0 ? stack[stack.length - 1] : 0;
}

function checkpointSetFromArg(value: string | undefined): Set<number> {
    const raw = value?.trim();
    if (!raw) return new Set([20, 40, 60]);
    return new Set(
        raw
            .split(',')
            .map(part => Number(part.trim()))
            .filter(num => Number.isInteger(num) && num >= 0)
    );
}

function auditCheckpoint(
    engine: EvolutionEngine<number[], number>,
    problem: ReturnType<typeof createMajorityProblem>,
    n: number,
    threshold: number,
    gen: number
): CheckpointAudit {
    const state = engine as any;
    const buffer = state.currentBuffer;
    const selectionIndices = state.selectionIndices as Uint32Array | undefined;
    const champIdx = selectionIndices?.[0];
    if (!buffer || champIdx === undefined) {
        throw new Error(`No winner produced at checkpoint gen=${gen}.`);
    }

    const byWeight: WeightRow[] = Array.from({ length: n + 1 }, () => ({ total: 0, correct: 0 }));
    let boundaryErrors = 0;
    const opcodes = buffer.getOpcodes(champIdx);
    const constants = buffer.getConstants(champIdx);
    for (const tc of problem.testCases) {
        const ones = tc.input.reduce((sum, value) => sum + (value ? 1 : 0), 0);
        const pred = evalBinary(opcodes, constants, tc.input) > 0 ? 1 : 0;
        byWeight[ones].total++;
        if (pred === tc.expectedOutput) {
            byWeight[ones].correct++;
        } else if (ones === threshold - 1 || ones === threshold) {
            boundaryErrors++;
        }
    }

    return {
        gen,
        fitness: Number(buffer.fitness[champIdx]?.toFixed(8) || 0),
        length: Number(buffer.lengths[champIdx] || 0),
        stagnationCount: Number(state.stagnationCount || 0),
        weight4Accuracy: Number(((byWeight[threshold - 1]?.correct || 0) / (byWeight[threshold - 1]?.total || 1)).toFixed(4)),
        weight5Accuracy: Number(((byWeight[threshold]?.correct || 0) / (byWeight[threshold]?.total || 1)).toFixed(4)),
        boundaryErrors
    };
}

function isRejected(audit: CheckpointAudit): boolean {
    return audit.gen >= 40
        && audit.stagnationCount >= 8
        && audit.fitness <= 0.968
        && audit.weight4Accuracy <= 0.922
        && audit.weight5Accuracy <= 0.94;
}

async function main() {
    const n = 9;
    const threshold = 5;
    const generationsArg = process.argv[4];
    const generations = Number.isFinite(Number(generationsArg)) ? Number(generationsArg) : 60;
    const mode = 'bridge';
    const seedArg = process.argv[2];
    const seed = seedArg !== undefined ? Number(seedArg) : undefined;
    const selectionMethodArg = (process.argv[3] || 'lexicase').toLowerCase();
    const selectionMethod = (
        selectionMethodArg === 'epsilon-lexicase'
        || selectionMethodArg === 'lexicase'
        || selectionMethodArg === 'tournament'
    ) ? selectionMethodArg : 'lexicase';
    const checkpoints = checkpointSetFromArg(process.argv[5]);
    const topology = resolveRuntimeTopology(1, 6);
    const maxGenomeSize = n >= 9 ? 512 : 320;

    const problem = createMajorityProblem(n, threshold);
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    const engine = new EvolutionEngine(problem, {
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
        selectionMethod,
        targetFitness: 1.0
    });

    engine.setDiscoveryAgent(agent);
    (engine as any).agentBridge = bridge;

    try {
        const audits: CheckpointAudit[] = [];
        for (let gen = 0; gen <= generations; gen++) {
            await (engine as any).step(gen, 0);
            if (!checkpoints.has(gen)) continue;

            const audit = auditCheckpoint(engine, problem, n, threshold, gen);
            audits.push(audit);
            if (isRejected(audit)) {
                console.log(JSON.stringify({
                    verdict: 'LANE_REJECTED',
                    reason: 'known-bad-majority9-basin',
                    n,
                    threshold,
                    generations,
                    mode,
                    seed: Number.isFinite(seed) ? seed : null,
                    selectionMethod,
                    parallelEvaluators: topology.parallelEvaluators,
                    distributedIslands: topology.distributedIslands,
                    topology: topology.topologyLabel,
                    checkpoints: audits
                }, null, 2));
                process.exitCode = 2;
                return;
            }
        }

        const finalAudit = auditCheckpoint(engine, problem, n, threshold, generations);

        console.log(JSON.stringify({
            verdict: 'SURVIVED_PREFLIGHT',
            n,
            threshold,
            generations,
            mode,
            seed: Number.isFinite(seed) ? seed : null,
            selectionMethod,
            parallelEvaluators: topology.parallelEvaluators,
            distributedIslands: topology.distributedIslands,
            topology: topology.topologyLabel,
            checkpoints: audits,
            final: finalAudit
        }, null, 2));
    } finally {
        await engine.dispose();
    }
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
