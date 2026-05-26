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

function createMajorityProfile(n: number, t: number): DomainProfile {
    return {
        domainClass: 'discrete',
        confidence: 1.0,
        signals: {
            cardinality: 2,
            range: [0, 1],
            isBinary: true,
            inputDensity: 1.0,
            entropy: 1.0,
            symmetryScore: 1.0,
            monotonicityScore: 1.0,
            alternationScore: 0.1,
            logicThreshold: t
        },
        recommendations: {
            evaluator: 'bitwise',
            refiner: 'boolean',
            operatorSet: 'logic',
            inputWidth: n
        }
    };
}

function triggerMajorityMotifs(n: number, t: number) {
    const agent = new DiscoveryAgent();
    agent.setDomainProfile(createMajorityProfile(n, t));
    agent.setEvaluationData(new Float64Array(1), new Float64Array(1), n, 'classification');
    agent.decideGlobalAction([{ gen: 20, bestFitness: 0.1, stagnationCount: 10, bestSize: 1 }], `majority-${n}-${t}`);
    return agent.pullAgentCandidates(20).filter(candidate => candidate.log.sourceId === 'MAJORITY_MOTIF');
}

function evaluateMajorityAccuracy(opcodes: Uint32Array, constants: Float64Array, n: number, t: number): number {
    const totalCases = Math.pow(2, n);
    let correct = 0;

    for (let i = 0; i < totalCases; i++) {
        const input: number[] = [];
        let ones = 0;

        for (let bit = 0; bit < n; bit++) {
            const value = (i >> bit) & 1;
            input.push(value);
            if (value === 1) ones++;
        }

        const expected = ones >= t ? 1 : 0;
        const actual = evaluateBinary(opcodes, constants, input);
        if (actual === expected) correct++;
    }

    return correct / totalCases;
}

async function runProof() {
    console.log('--- MAJORITY MOTIF HONESTY PROOF ---');

    const motifs = triggerMajorityMotifs(5, 3);
    const motifCountPass = motifs.length > 0;
    const structurePass = motifs.every(candidate => {
        const opcodes = candidate.genome.opcodes as Uint32Array;
        return opcodes.length === 3
            && ((opcodes[0] >>> 24) as OpCode) === OpCode.XN
            && ((opcodes[1] >>> 24) as OpCode) === OpCode.XN
            && ((opcodes[2] >>> 24) as OpCode) === OpCode.AND;
    });
    const variableRangePass = motifs.every(candidate => {
        const opcodes = candidate.genome.opcodes as Uint32Array;
        return Array.from(opcodes)
            .filter(raw => ((raw >>> 24) as OpCode) === OpCode.XN)
            .every(raw => (raw & 0xFFFFFF) < 5);
    });
    const accuracies = motifs.map(candidate =>
        evaluateMajorityAccuracy(candidate.genome.opcodes as Uint32Array, candidate.genome.constants as Float64Array, 5, 3)
    );
    const honestyPass = accuracies.every(accuracy => accuracy > 0.5 && accuracy < 1.0);

    console.log(`MOTIF COUNT ${motifCountPass ? 'PASS' : 'FAIL'} (${motifs.length})`);
    console.log(`STRUCTURE ${structurePass ? 'PASS' : 'FAIL'}`);
    console.log(`VARIABLE RANGE ${variableRangePass ? 'PASS' : 'FAIL'}`);
    console.log(`HONEST ACCURACY ${honestyPass ? 'PASS' : 'FAIL'} (${accuracies.map(a => a.toFixed(4)).join(', ')})`);

    if (!(motifCountPass && structurePass && variableRangePass && honestyPass)) process.exit(1);
}

runProof().catch(e => {
    console.error(e);
    process.exit(1);
});
