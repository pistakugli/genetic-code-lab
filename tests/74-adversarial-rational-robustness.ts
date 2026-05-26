/**
 * 🕵️ Phase 74: Adversarial Rational Robustness Audit
 * Tests if the integrated quotient mode can survive adversarial conditions:
 * redundant constants, algebraically equivalent but ugly targets,
 * wider input ranges, and noise.
 */

import { SuperEngine } from '../src/core/super-engine';
import { JITCompiler } from '../src/core/jit-compiler';
import { Genome, Gene, ProblemDefinition } from '../src/core/types';
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

const adversarialFamily = [
    {
        name: "A1: Wide Domain & Noise (x / 1+x^2)",
        vars: ['x0'],
        func: (x: number[]) => x[0] / (1 + x[0] * x[0]),
        domain: [-10, 10] as [number, number], // Wider domain
        noise: 0.01, // 1% noise
        testCount: 40
    },
    {
        name: "A2: Ugly Constants (3.14x + 2.71) / (x^2 + 1.41)",
        vars: ['x0'],
        func: (x: number[]) => (3.14 * x[0] + 2.71) / (x[0] * x[0] + 1.41),
        domain: [-5, 5] as [number, number],
        noise: 0,
        testCount: 40
    },
    {
        name: "A3: 2D Heavy Cross-Terms (xy + x - y) / (1 + x^2 + y^2)",
        vars: ['x0', 'x1'],
        func: (x: number[]) => (x[0]*x[1] + x[0] - x[1]) / (1 + x[0]*x[0] + x[1]*x[1]),
        domain: [-2, 2] as [number, number],
        noise: 0,
        testCount: 40
    }
];

async function runAdversarialAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 74: ADVERSARIAL RATIONAL ROBUSTNESS AUDIT ===\n'));

    const SEEDS = [1001, 2002, 3003]; 
    const results: any[] = [];

    for (const prob of adversarialFamily) {
        console.log(chalk.yellow(`\n🚀 Testing Adversarial Problem: ${prob.name}`));

        const testCases = generateTestCases(prob.domain, prob.vars.length, prob.testCount, prob.func, prob.noise);
        const terminalSet: any[] = prob.vars.map(v => ({ name: v, type: 'variable', value: 0 }));
        
        // Provide standard constants, the engine must synthesize the ugly ones or approximate
        terminalSet.push({ name: 'const', type: 'constant', value: () => Number((Math.random() * 5).toFixed(3)) });

        const problemDef: ProblemDefinition<any, any> = {
            id: prob.name,
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
            const engine = new SuperEngine(problemDef, {
                seed,
                maxGenerations: 150, // More generations for harder problems
                // Deterministic control lane: keep this audit local until it is intentionally promoted.
                parallelEvaluators: 1,
                selectionMethod: 'epsilon-lexicase',
                quotientMode: true,
                strictQuotientValidation: true,
                optimizer: {
                    enabled: true, // Enable optimizer to help with ugly constants
                    iterations: 50,
                    learningRate: 0.05,
                    interval: 10,
                    targetCount: 5
                },
                population: { size: 3000, maxGenomeSize: 40, maxGenomeDepth: 10, eliteCount: 20, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
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
                // Relaxed threshold for discovery due to noise and ugly constants
                if (mae < 0.02) verdict = 'ROBUST SYMBOLIC DISCOVERY';
                else verdict = 'ROBUST RATIONAL APPROXIMATION';
            } else if (!denomCollapses && !hasRemovableFactor && mae < Infinity) {
                verdict = 'POOR APPROXIMATION';
            }

            results.push({ prob: prob.name.split(':')[0], seed, mae, verdict, structure, validDenom: !denomCollapses, artifact: hasRemovableFactor });
            await engine.shutdown();
        }
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('ADVERSARIAL ROBUSTNESS TABLE:'));
    console.log(' | Problem | Seed | Best Structure | Raw MAE | Denom Valid | Artifact | Verdict |');
    for (const r of results) {
        console.log(` | ${r.prob} | ${r.seed} | \`${r.structure.substring(0, 45)}\` | ${r.mae.toFixed(6)} | ${r.validDenom ? 'YES' : 'NO'} | ${r.artifact ? 'YES' : 'NO'} | ${r.verdict} |`);
    }
}

runAdversarialAudit().catch(console.error);
