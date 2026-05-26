import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { AsyncAgentBridge } from '../src/core/async-agent-bridge';
import { resolveRuntimeTopology } from '../src/core/runtime-lane';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { OpCode, PopulationBuffer } from '../src/core/pop-buffer';

type BoundaryError = {
    ones: number;
    mask: string;
    expected: number;
    predicted: number;
};

type AuditResult = {
    fitness: number;
    byWeight: Array<{ ones: number; total: number; accuracy: number }>;
    boundaryErrors: BoundaryError[];
    histogram: Record<string, number>;
    length: number;
};

function evalBinary(opcodes: Uint32Array | number[], constants: Float64Array, input: number[]): number {
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

function appendNegativeBoundaryGuard(candidate: number[], inactiveMask: string): void {
    const vars: number[] = [];
    for (let i = 0; i < inactiveMask.length; i++) {
        if (inactiveMask[i] === '0') vars.push(i);
    }
    appendFoldedTerm(candidate, vars, OpCode.OR);
    candidate.push(PopulationBuffer.pack(OpCode.AND, 0));
}

function appendPositiveBoundaryTerm(candidate: number[], activeMask: string): void {
    const vars: number[] = [];
    for (let i = 0; i < activeMask.length; i++) {
        if (activeMask[i] === '1') vars.push(i);
    }
    appendFoldedTerm(candidate, vars, OpCode.AND);
    candidate.push(PopulationBuffer.pack(OpCode.OR, 0));
}

function auditCandidate(
    problem: ReturnType<typeof createMajorityProblem>,
    opcodes: Uint32Array | number[],
    constants: Float64Array,
    threshold: number
): AuditResult {
    const byWeight = Array.from({ length: problem.testCases[0].input.length + 1 }, (_, ones) => ({
        ones,
        total: 0,
        correct: 0
    }));
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

    const histogram: Record<string, number> = {};
    for (let i = 0; i < opcodes.length; i++) {
        const key = OpCode[opcodes[i] >>> 24] || `OP_${opcodes[i] >>> 24}`;
        histogram[key] = (histogram[key] || 0) + 1;
    }

    return {
        fitness: Number((correct / problem.testCases.length).toFixed(8)),
        byWeight: byWeight.map(row => ({
            ones: row.ones,
            total: row.total,
            accuracy: Number((row.correct / (row.total || 1)).toFixed(4))
        })),
        boundaryErrors,
        histogram,
        length: opcodes.length
    };
}

async function main() {
    const n = 9;
    const threshold = 5;
    const generations = 140;
    const seed = 1337;
    const topology = resolveRuntimeTopology();

    const problem = createMajorityProblem(n, threshold);
    const agent = new DiscoveryAgent();
    const bridge = new AsyncAgentBridge(agent);
    const engine = new EvolutionEngine(problem, {
        seed,
        population: {
            size: 450,
            eliteCount: 20,
            maxGenomeSize: 420,
            crossoverRate: 0.85,
            mutationRate: 0.22,
            maxGenomeDepth: 16,
            tournamentSize: 8
        },
        maxGenerations: generations,
        parallelEvaluators: topology.parallelEvaluators,
        selectionMethod: 'lexicase',
        targetFitness: 1.0
    });

    engine.setDiscoveryAgent(agent);
    (engine as any).agentBridge = bridge;

    try {
        for (let gen = 0; gen <= generations; gen++) {
            await (engine as any).step(gen, 0);
        }

        const bestBinary = engine.getBestBinaryRaw();
        if (!bestBinary) throw new Error('No winner produced.');

        const baseline = auditCandidate(problem, bestBinary.opcodes, bestBinary.constants, threshold);
        const courtCandidate = Array.from(bestBinary.opcodes);
        for (const err of baseline.boundaryErrors) {
            if (err.expected === 0 && err.predicted === 1) appendNegativeBoundaryGuard(courtCandidate, err.mask);
        }
        for (const err of baseline.boundaryErrors) {
            if (err.expected === 1 && err.predicted === 0) appendPositiveBoundaryTerm(courtCandidate, err.mask);
        }
        const court = auditCandidate(problem, courtCandidate, bestBinary.constants, threshold);

        console.log(JSON.stringify({
            parallelEvaluators: topology.parallelEvaluators,
            distributedIslands: topology.distributedIslands,
            topology: topology.topologyLabel,
            baseline,
            court
        }, null, 2));
    } finally {
        await engine.dispose();
    }
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
