import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { resolveRuntimeTopology } from '../src/core/runtime-lane';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';
import type { EvolutionConfig } from '../src/core/types';

type BoundaryError = {
    ones: number;
    mask: string;
    expected: number;
    predicted: number;
};

type Audit = {
    fitness: number;
    length: number;
    boundaryErrors: BoundaryError[];
    byWeight: Array<{ ones: number; total: number; accuracy: number }>;
};

function evalBinary(opcodes: ArrayLike<number>, constants: Float64Array, input: number[]): number {
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

function appendFoldedTerm(out: number[], vars: number[], op: OpCode.AND | OpCode.OR): void {
    if (vars.length === 0) return;
    for (const variable of vars) out.push(PopulationBuffer.pack(OpCode.XN, variable));
    for (let i = 1; i < vars.length; i++) out.push(PopulationBuffer.pack(op, 0));
}

function inactiveVarsFromMask(mask: string): number[] {
    const vars: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i] === '0') vars.push(i);
    return vars;
}

function activeVarsFromMask(mask: string): number[] {
    const vars: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i] === '1') vars.push(i);
    return vars;
}

function appendNegativeGuard(candidate: number[], mask: string): void {
    appendFoldedTerm(candidate, inactiveVarsFromMask(mask), OpCode.OR);
    candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
}

function appendPositiveTerm(candidate: number[], mask: string): void {
    appendFoldedTerm(candidate, activeVarsFromMask(mask), OpCode.AND);
    candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
}

function combinations<T>(items: T[]): T[][] {
    const result: T[][] = [[]];
    for (const item of items) {
        const current = result.slice();
        for (const subset of current) result.push([...subset, item]);
    }
    return result;
}

function audit(
    problem: ReturnType<typeof createMajorityProblem>,
    threshold: number,
    opcodes: ArrayLike<number>,
    constants: Float64Array
): Audit {
    const byWeight = Array.from({ length: 10 }, (_, ones) => ({ ones, total: 0, correct: 0 }));
    const boundaryErrors: BoundaryError[] = [];
    let correct = 0;

    for (const tc of problem.testCases) {
        const ones = tc.input.reduce((sum, value) => sum + (value ? 1 : 0), 0);
        const pred = evalBinary(opcodes, constants, tc.input) > 0 ? 1 : 0;
        byWeight[ones].total++;
        if (pred === tc.expectedOutput) {
            byWeight[ones].correct++;
            correct++;
        } else if (ones === threshold - 1 || ones === threshold) {
            boundaryErrors.push({
                ones,
                mask: tc.input.map(value => value ? '1' : '0').join(''),
                expected: tc.expectedOutput,
                predicted: pred
            });
        }
    }

    return {
        fitness: Number((correct / problem.testCases.length).toFixed(8)),
        length: opcodes.length,
        boundaryErrors,
        byWeight: byWeight.map(row => ({
            ones: row.ones,
            total: row.total,
            accuracy: Number((row.correct / (row.total || 1)).toFixed(4))
        }))
    };
}

async function main() {
    const n = 9;
    const threshold = 5;
    const generations = 140;
    const seed = 1337;
    const topology = resolveRuntimeTopology();
    const maxGenomeSize = 512;

    const problem = createMajorityProblem(n, threshold);
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    const evolutionConfig: Partial<EvolutionConfig> = {
        seed,
        population: {
            size: 450,
            eliteCount: 20,
            maxGenomeSize,
            crossoverRate: 0.85,
            mutationRate: 0.22,
            maxGenomeDepth: 16,
            tournamentSize: 8
        },
        maxGenerations: generations,
        parallelEvaluators: topology.parallelEvaluators,
        selectionMethod: 'lexicase',
        targetFitness: 1.0
    };

    const engine = new EvolutionEngine(problem, evolutionConfig);
    engine.setDiscoveryAgent(agent);
    (engine as any).agentBridge = bridge;

    try {
        for (let gen = 0; gen <= generations; gen++) {
            await (engine as any).step(gen, 0);
        }

        const bestBinary = engine.getBestBinaryRaw();
        if (!bestBinary) throw new Error('No winner produced.');

        const baseline = audit(problem, threshold, bestBinary.opcodes, bestBinary.constants);
        const negatives = baseline.boundaryErrors.filter(err => err.expected === 0 && err.predicted === 1);
        const positives = baseline.boundaryErrors.filter(err => err.expected === 1 && err.predicted === 0);

        const candidates = [];
        for (const negSubset of combinations(negatives)) {
            for (const posSubset of combinations(positives)) {
                if (negSubset.length === 0 && posSubset.length === 0) continue;
                const candidate = Array.from(bestBinary.opcodes);
                for (const err of negSubset) appendNegativeGuard(candidate, err.mask);
                for (const err of posSubset) appendPositiveTerm(candidate, err.mask);
                const result = audit(problem, threshold, candidate, bestBinary.constants);
                candidates.push({
                    negMasks: negSubset.map(err => err.mask),
                    posMasks: posSubset.map(err => err.mask),
                    result
                });
            }
        }

        candidates.sort((a, b) => {
            if (b.result.fitness !== a.result.fitness) return b.result.fitness - a.result.fitness;
            return a.result.length - b.result.length;
        });

        console.log(JSON.stringify({
            parallelEvaluators: topology.parallelEvaluators,
            distributedIslands: topology.distributedIslands,
            topology: topology.topologyLabel,
            baseline,
            best: candidates.slice(0, 10)
        }, null, 2));
    } finally {
        await engine.dispose();
    }
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
