import { EvolutionEngine } from '../src/core/evolution';
import { DiscoveryAgent } from '../src/agents/discovery-agent';
import { createMajorityProblem } from '../src/solvers/boolean-library';
import { OpCode } from '../src/core/pop-buffer';
import { DomainProfile } from '../src/core/domain-analyzer';
import { resolvePerformanceParallelism } from '../src/core/runtime-lane';

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

function evaluateMajorityFitness(opcodes: Uint32Array, constants: Float64Array, n: number, t: number): number {
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

async function runContributionAudit() {
    console.log('--- PHASE 65.2 MAJORITY MOTIF CONTRIBUTION AUDIT ---');
    const parallelEvaluators = resolvePerformanceParallelism();

    const cases = [
        { n: 3, t: 2 },
        { n: 5, t: 3 },
        { n: 7, t: 4 },
        { n: 9, t: 5 }
    ];

    const results: any[] = [];

    for (const { n, t } of cases) {
        console.log(`\nEvaluating Majority ${n}-${t} performance...`);

        const motifs = collectMajorityMotifs(n, t);
        const agentOnlyFitness = motifs.length > 0
            ? Math.max(...motifs.map(candidate =>
                evaluateMajorityFitness(candidate.genome.opcodes as Uint32Array, candidate.genome.constants as Float64Array, n, t)
            ))
            : 0;

        const problem = createMajorityProblem(n, t);
        const engineEvo = new EvolutionEngine(problem, {
            population: { size: 100, eliteCount: 10, maxGenomeSize: 200, crossoverRate: 0.8, mutationRate: 0.2, maxGenomeDepth: 12, tournamentSize: 4 },
            parallelEvaluators,
            maxGenerations: 40
        });

        let evolutionOnlyFitness = 0;
        for (let gen = 0; gen <= 40; gen++) {
            await (engineEvo as any).step(gen, 0);
            const best = engineEvo.getBestGenome();
            if (best && best.fitness > evolutionOnlyFitness) evolutionOnlyFitness = best.fitness;
            if (evolutionOnlyFitness >= 1.0) break;
        }

        const engineAssisted = new EvolutionEngine(problem, {
            population: {
                size: n <= 5 ? 1000 : 250,
                eliteCount: 10,
                maxGenomeSize: 2000,
                crossoverRate: 0.1,
                mutationRate: 0.8,
                maxGenomeDepth: 40,
                tournamentSize: 4
            },
            parallelEvaluators,
            maxGenerations: n <= 5 ? 500 : 200
        });
        (engineAssisted as any).opCodes = [OpCode.AND, OpCode.OR, OpCode.XN];

        const assistedAgent = new DiscoveryAgent();
        engineAssisted.setDiscoveryAgent(assistedAgent);

        let assistedFitness = 0;
        let genOfFirstLift = -1;
        let genOfVictory = -1;

        for (let gen = 0; gen <= (n <= 5 ? 500 : 200); gen++) {
            const bestBefore = engineAssisted.getBestGenome()?.fitness || 0;
            await (engineAssisted as any).step(gen, 0);
            const bestAfter = engineAssisted.getBestGenome()?.fitness || 0;

            if (genOfFirstLift === -1 && bestAfter > bestBefore + 0.05) genOfFirstLift = gen;
            if (bestAfter >= 0.999999 && genOfVictory === -1) genOfVictory = gen;

            assistedFitness = bestAfter;
            if (assistedFitness >= 0.999999) break;
        }

        results.push({
            problem: `Majority ${n}-${t}`,
            motifs: motifs.length,
            agentOnly: agentOnlyFitness.toFixed(4),
            evolutionOnly: evolutionOnlyFitness.toFixed(4),
            agentAssisted: assistedFitness.toFixed(4),
            genOfFirstLift: genOfFirstLift === -1 ? 'None' : genOfFirstLift,
            genOfVictory: genOfVictory === -1 ? 'Fail' : genOfVictory,
            improvement: (assistedFitness - evolutionOnlyFitness).toFixed(4)
        });
    }

    console.log('\n--- PHASE 65.2 CONTRIBUTION REPORT ---');
    console.table(results);
}

runContributionAudit().catch(e => {
    console.error(e);
    process.exit(1);
});
