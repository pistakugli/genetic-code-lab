import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { DomainProfile } from '../src/core/domain-analyzer';
import { OpCode } from '../src/core/pop-buffer';

function evaluateBinary(opcodes: Uint32Array, constants: Float64Array, input: number[]): number {
    const stack: number[] = [];

    for (let i = 0; i < opcodes.length; i++) {
        const raw = opcodes[i];
        const op = (raw >>> 24) as OpCode;
        const data = raw & 0xFFFFFF;

        switch (op) {
            case OpCode.XN: stack.push(input[data]); break;
            case OpCode.CONST: stack.push(constants[data]); break;
            case OpCode.AND: { const b = stack.pop()!; const a = stack.pop()!; stack.push((a > 0.5 && b > 0.5) ? 1 : 0); break; }
            case OpCode.OR: { const b = stack.pop()!; const a = stack.pop()!; stack.push((a > 0.5 || b > 0.5) ? 1 : 0); break; }
            case OpCode.NAND: { const b = stack.pop()!; const a = stack.pop()!; stack.push((a > 0.5 && b > 0.5) ? 0 : 1); break; }
            case OpCode.NOR: { const b = stack.pop()!; const a = stack.pop()!; stack.push((a > 0.5 || b > 0.5) ? 0 : 1); break; }
            default: throw new Error(`Unsupported opcode ${op}`);
        }
    }

    return stack.pop() ?? 0;
}

function createMuxProfile(): DomainProfile {
    return {
        domainClass: 'discrete',
        confidence: 1.0,
        signals: {
            cardinality: 2,
            range: [0, 1],
            isBinary: true,
            inputDensity: 1.0,
            entropy: 1.0,
            symmetryScore: 0.1,
            monotonicityScore: 0.1,
            alternationScore: 0.1,
            selectorMap: [1, 1, 0, 0, 0, 0]
        },
        recommendations: {
            evaluator: 'bitwise',
            refiner: 'boolean',
            operatorSet: 'logic',
            inputWidth: 6
        }
    };
}

function triggerMuxMotifs() {
    const agent = new DiscoveryAgent();
    agent.setDomainProfile(createMuxProfile());
    agent.setEvaluationData(new Float64Array(1), new Float64Array(1), 6, 'classification');
    agent.decideGlobalAction([{ gen: 20, bestFitness: 0.1, stagnationCount: 10, bestSize: 1 }], 'mux-6');
    return agent.pullAgentCandidates(20).filter(candidate => candidate.log.sourceId === 'MUX_MOTIF');
}

function evaluateMuxAccuracy(opcodes: Uint32Array, constants: Float64Array): number {
    let correct = 0;

    for (let i = 0; i < 64; i++) {
        const input: number[] = [];
        for (let bit = 0; bit < 6; bit++) input.push((i >> bit) & 1);

        const s0 = input[0];
        const s1 = input[1];
        const d0 = input[2];
        const d1 = input[3];
        const d2 = input[4];
        const d3 = input[5];

        let expected = 0;
        if (s1 === 0 && s0 === 0) expected = d0;
        else if (s1 === 0 && s0 === 1) expected = d1;
        else if (s1 === 1 && s0 === 0) expected = d2;
        else expected = d3;

        const actual = evaluateBinary(opcodes, constants, input);
        if (actual === expected) correct++;
    }

    return correct / 64;
}

async function runMuxProof() {
    console.log('--- MUX MOTIF HONESTY PROOF ---');

    const motifs = triggerMuxMotifs();
    const motifCountPass = motifs.length > 0;
    const structurePass = motifs.every(candidate => {
        const opcodes = candidate.genome.opcodes as Uint32Array;
        const counts = Array.from(opcodes).reduce<Record<number, number>>((acc, raw) => {
            const op = raw >>> 24;
            acc[op] = (acc[op] || 0) + 1;
            return acc;
        }, {});

        return opcodes.length === 9
            && (counts[OpCode.XN] || 0) === 5
            && (counts[OpCode.AND] || 0) === 2
            && (counts[OpCode.NAND] || 0) === 1
            && (counts[OpCode.OR] || 0) === 1;
    });
    const variableRangePass = motifs.every(candidate =>
        Array.from(candidate.genome.opcodes as Uint32Array)
            .filter(raw => ((raw >>> 24) as OpCode) === OpCode.XN)
            .every(raw => (raw & 0xFFFFFF) < 6)
    );
    const accuracies = motifs.map(candidate =>
        evaluateMuxAccuracy(candidate.genome.opcodes as Uint32Array, candidate.genome.constants as Float64Array)
    );
    const honestyPass = accuracies.every(accuracy => accuracy > 0.5 && accuracy < 1.0);

    console.log(`MOTIF COUNT ${motifCountPass ? 'PASS' : 'FAIL'} (${motifs.length})`);
    console.log(`STRUCTURE ${structurePass ? 'PASS' : 'FAIL'}`);
    console.log(`VARIABLE RANGE ${variableRangePass ? 'PASS' : 'FAIL'}`);
    console.log(`HONEST ACCURACY ${honestyPass ? 'PASS' : 'FAIL'} (${accuracies.map(a => a.toFixed(4)).join(', ')})`);

    if (!(motifCountPass && structurePass && variableRangePass && honestyPass)) process.exit(1);
}

runMuxProof().catch(e => {
    console.error(e);
    process.exit(1);
});
