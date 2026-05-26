import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { OpCode } from '../src/core/pop-buffer';
import { DomainProfile } from '../src/core/domain-analyzer';

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

function collectMajorityMotifs(n: number, t: number) {
    const agent = new DiscoveryAgent();
    agent.setDomainProfile(createMajorityProfile(n, t));
    agent.setEvaluationData(new Float64Array(1), new Float64Array(1), n, 'classification');
    agent.decideGlobalAction([{ gen: 20, bestFitness: 0.1, stagnationCount: 10, bestSize: 1 }], `majority-${n}-${t}`);
    return agent.pullAgentCandidates(20).filter(candidate => candidate.log.sourceId === 'MAJORITY_MOTIF');
}

async function runMismatchAudit() {
    console.log('--- PHASE 65.3 MAJORITY MOTIF MISMATCH AUDIT ---');

    const cases = [
        { n: 3, t: 2 },
        { n: 5, t: 3 }
    ];

    for (const { n, t } of cases) {
        console.log(`\nAnalyzing mismatches for Majority ${n}-${t} motifs...`);

        const motifs = collectMajorityMotifs(n, t);
        const candidate = motifs[0];

        if (!candidate) {
            console.log('FAILED to generate motif.');
            continue;
        }

        const genome = candidate.genome as { opcodes: Uint32Array, constants: Float64Array };
        console.log(`Motif size: ${genome.opcodes.length} ops. Candidate count: ${motifs.length}.`);

        const totalCases = Math.pow(2, n);
        let mismatches = 0;

        for (let i = 0; i < totalCases; i++) {
            const input: number[] = [];
            let ones = 0;

            for (let bit = 0; bit < n; bit++) {
                const value = (i >> bit) & 1;
                input.push(value);
                if (value === 1) ones++;
            }

            const expected = ones >= t ? 1 : 0;
            const actual = evaluateBinary(genome.opcodes, genome.constants, input);

            if (actual !== expected) {
                mismatches++;
                console.log(`  [MISMATCH] Row ${i.toString(2).padStart(n, '0')}: expected ${expected}, got ${actual} (ones: ${ones})`);
            }
        }

        console.log(`Summary: ${mismatches} mismatches out of ${totalCases} rows.`);
    }
}

runMismatchAudit().catch(e => {
    console.error(e);
    process.exit(1);
});
