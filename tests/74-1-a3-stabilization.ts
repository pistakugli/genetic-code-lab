/**
 * 🕵️ Phase 74.1: A3 Stabilization Audit
 * Focused on stabilizing the 2D Rational Quotient Search (A3 problem).
 */

import { SuperEngine } from '../src/core/super-engine';
import { JITCompiler } from '../src/core/jit-compiler';
import { Genome, Gene, ProblemDefinition } from '../src/core/types';
import { resolveDistributedIslands, resolvePerformanceParallelism } from '../src/core/runtime-lane';
import chalk from 'chalk';

function evaluateAST(genome: Genome, nodeId: string, inputs: number[], nodeMap?: Map<string, any>): number {
    if (!nodeMap) {
        nodeMap = new Map();
        for (const g of genome.genes) nodeMap.set(g.id, g);
    }
    const node = nodeMap.get(nodeId);
    if (!node) return 0;

    if (node.type === 'terminal') {
        if (node.operation === 'const' || node.operation === 'constant') return node.operands[0] as number;
        const idx = parseInt(node.operation.replace('x', ''));
        return inputs[idx] || 0;
    }

    const left = evaluateAST(genome, node.operands[0] as string, inputs, nodeMap);
    const right = node.operands[1] ? evaluateAST(genome, node.operands[1] as string, inputs, nodeMap) : 0;

    switch (node.operation) {
        case 'add': return left + right;
        case 'sub': return left - right;
        case 'mul': return left * right;
        case 'div': return right === 0 ? NaN : left / right;
        default: return 0;
    }
}

function generateTestCases(domain: [number, number], vars: number, count: number, func: (x: number[]) => number, noiseLevel: number = 0) {
    const cases = [];
    for (let i = 0; i < count; i++) {
        const input = [];
        for (let v = 0; v < vars; v++) {
            input.push(domain[0] + Math.random() * (domain[1] - domain[0]));
        }
        let output = func(input);
        if (noiseLevel > 0) {
            output += (Math.random() * 2 - 1) * noiseLevel * output; // Relative noise
        }
        cases.push({ input, expectedOutput: output });
    }
    return cases;
}

const prob = {
    name: "A3: 2D Heavy Cross-Terms (xy + x - y) / (1 + x^2 + y^2)",
    vars: ['x0', 'x1'],
    func: (x: number[]) => (x[0]*x[1] + x[0] - x[1]) / (1 + x[0]*x[0] + x[1]*x[1]),
    domain: [-2, 2] as [number, number],
    noise: 0,
    testCount: 30 
};

async function runA3Audit() {
    console.log(chalk.cyan.bold('\n=== PHASE 74.1: A3 STABILIZATION AUDIT ===\n'));

    const SEEDS = [1001, 2002, 3003, 4004, 5005]; 
    const results: any[] = [];
    const distributedIslands = resolveDistributedIslands(4);
    const islandParallelEvaluators = resolvePerformanceParallelism(1);
    console.log(chalk.white(`Topology: ${distributedIslands} islands x ${islandParallelEvaluators} evaluators`));

    const testCases = generateTestCases(prob.domain, prob.vars.length, prob.testCount, prob.func, prob.noise);
    // Phase 74: Ensure origin is explicitly in the training set so the binary engine feels the quotient penalty
    testCases.push({ input: [0, 0], expectedOutput: 0 });
    testCases.push({ input: [0.001, 0], expectedOutput: prob.func([0.001, 0]) });
    testCases.push({ input: [0, 0.001], expectedOutput: prob.func([0, 0.001]) });
    const terminalSet: any[] = prob.vars.map(v => ({ name: v, type: 'variable', value: 0 }));
    
    // Provide a strict 1.0 constant to help form 1 + x^2 + y^2 without noise, plus random ones
    terminalSet.push({ name: 'const', type: 'constant', value: 1.0 });

    const problemDef: ProblemDefinition<any, any> = {
        name: prob.name,
        functionSet: [
            { name: 'add', arity: 2, returnType: 'number', implementation: (a: any, b: any) => a + b },
            { name: 'sub', arity: 2, returnType: 'number', implementation: (a: any, b: any) => a - b },
            { name: 'mul', arity: 2, returnType: 'number', implementation: (a: any, b: any) => a * b }
        ],
        terminalSet,
        testCases,
        fitnessFunction: () => 0 
    } as any;

    for (const seed of SEEDS) {
        console.log(chalk.yellow(`\n🚀 Testing A3 with Seed: ${seed}`));
        
        const engine = new SuperEngine(problemDef, {
            seed,
            maxGenerations: 200, 
            distributedIslands,
            parallelEvaluators: islandParallelEvaluators,
            selectionMethod: 'epsilon-lexicase',
            quotientMode: true,
            strictQuotientValidation: true,
            optimizer: {
                enabled: true, 
                iterations: 50,
                learningRate: 0.05,
                interval: 10,
                targetCount: 5
            },
            population: { 
                size: 5000, 
                maxGenomeSize: 100, 
                maxGenomeDepth: 12, 
                eliteCount: 20, 
                crossoverRate: 0.8, 
                mutationRate: 0.5, 
                tournamentSize: 7,
                parsimonyCoefficient: 1e-4 // gentle bloat control
            }
        });

        const result = await engine.evolve();
        const bestGenome = result.bestGenome;
        const root = bestGenome.genes[0];

        let numStr = '', denStr = '', structure = '';
        if (root && root.operands[1]) {
            numStr = JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[0]), ...bestGenome.genes].filter((x): x is Gene => !!x) });
            denStr = JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[1]), ...bestGenome.genes].filter((x): x is Gene => !!x) });
            structure = `[ ${numStr} ] / [ ${denStr} ]`;
        } else {
            structure = JITCompiler.decompile(bestGenome);
        }

        let hasRemovableFactor = false;
        let denomCollapses = false;
        
        const probePoints = [];
        for (let i = 0; i < 5; i++) {
            const pt = [];
            for(let v=0; v<prob.vars.length; v++) pt.push(prob.domain[0] + Math.random() * (prob.domain[1] - prob.domain[0]));
            probePoints.push(pt);
        }
        if (prob.domain[0] <= 0 && prob.domain[1] >= 0) {
           probePoints.push(new Array(prob.vars.length).fill(0));
        }
        
        if (root && root.operands[1]) {
            for (const probe of probePoints) {
                const numProbe = evaluateAST(bestGenome, root.operands[0] as string, probe);
                const denProbe = evaluateAST(bestGenome, root.operands[1] as string, probe);
                if (Math.abs(numProbe) < 1e-5 && Math.abs(denProbe) < 1e-5) hasRemovableFactor = true;
                if (Math.abs(denProbe) < 1e-7) denomCollapses = true;
            }
        }

        const fitness = bestGenome.fitness;
        const mae = fitness > 0 ? (1 / fitness - 1) : Infinity;
        
        let verdict = 'INVALID CANDIDATE';
        if (!denomCollapses && !hasRemovableFactor && mae < 0.05) {
            if (mae < 0.0001) verdict = 'RAW VALID SYMBOLIC DISCOVERY';
            else verdict = 'RAW VALID RATIONAL APPROXIMATION';
        } else if (!denomCollapses && !hasRemovableFactor && mae < Infinity) {
            verdict = 'POOR APPROXIMATION';
        }

        results.push({ prob: prob.name.split(':')[0], seed, mae, verdict, structure, validDenom: !denomCollapses, artifact: hasRemovableFactor });
        await engine.shutdown();
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('A3 STABILIZATION TABLE:'));
    console.log(' | Problem | Seed | Best Structure | Raw MAE | Denom Valid | Artifact | Verdict |');
    for (const r of results) {
        console.log(` | ${r.prob} | ${r.seed} | \`${r.structure}\` | ${r.mae.toFixed(6)} | ${r.validDenom ? 'YES' : 'NO'} | ${r.artifact ? 'YES' : 'NO'} | ${r.verdict} |`);
    }
}

runA3Audit().catch(console.error);
