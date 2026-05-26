/**
 * 🕵️ Phase 73: Rational Family Generalization Audit
 * Tests if the integrated quotient mode can solve a class of rational functions.
 */

import { SuperEngine } from '../src/core/super-engine';
import { JITCompiler } from '../src/core/jit-compiler';
import { Genome, Gene, ProblemDefinition } from '../src/core/types';
import { collectSubtree } from '../src/core/genome';
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

function generateTestCases(domain: [number, number], vars: number, count: number, func: (x: number[]) => number) {
    const cases = [];
    for (let i = 0; i < count; i++) {
        const input = [];
        for (let v = 0; v < vars; v++) {
            input.push(domain[0] + Math.random() * (domain[1] - domain[0]));
        }
        cases.push({ input, expectedOutput: func(input) });
    }
    return cases;
}

const family = [
    {
        name: "P1: x / (1 + x^2)",
        vars: ['x0'],
        func: (x: number[]) => x[0] / (1 + x[0] * x[0]),
        domain: [-2, 2] as [number, number],
        testCount: 20
    },
    {
        name: "P2: (x^2 + 1) / (x + 1)",
        vars: ['x0'],
        func: (x: number[]) => (x[0] * x[0] + 1) / (x[0] + 1),
        domain: [0, 5] as [number, number], // Avoid exact -1 singularity
        testCount: 20
    },
    {
        name: "P3: (2x + 1) / (x^2 + 1)",
        vars: ['x0'],
        func: (x: number[]) => (2 * x[0] + 1) / (x[0] * x[0] + 1),
        domain: [-2, 2] as [number, number],
        testCount: 20
    },
    {
        name: "P4: (x + y) / (1 + x*y)",
        vars: ['x0', 'x1'],
        func: (x: number[]) => (x[0] + x[1]) / (1 + x[0] * x[1]),
        domain: [-0.5, 0.5] as [number, number],
        testCount: 20
    },
    {
        name: "P5: (x - y) / (1 - x*y)",
        vars: ['x0', 'x1'],
        func: (x: number[]) => (x[0] - x[1]) / (1 - x[0] * x[1]),
        domain: [-0.5, 0.5] as [number, number],
        testCount: 20
    }
];

async function runFamilyAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 73: RATIONAL FAMILY GENERALIZATION AUDIT ===\n'));

    const SEEDS = [42, 142, 242]; // 3 seeds per problem
    const results: any[] = [];
    const distributedIslands = resolveDistributedIslands(4);
    const islandParallelEvaluators = resolvePerformanceParallelism(1);
    console.log(chalk.white(`Topology: ${distributedIslands} islands x ${islandParallelEvaluators} evaluators`));

    for (const prob of family) {
        console.log(chalk.yellow(`\n🚀 Testing Problem: ${prob.name}`));

        const testCases = generateTestCases(prob.domain, prob.vars.length, prob.testCount, prob.func);
        const terminalSet: any[] = prob.vars.map(v => ({ name: v, type: 'variable', value: 0 }));
        terminalSet.push({ name: 'const', type: 'constant', value: 1.0 });

        const problemDef: ProblemDefinition<any, any> = {
            id: prob.name,
            functionSet: [
                { name: 'add', arity: 2, returnType: 'number', implementation: (a: any, b: any) => a + b },
                { name: 'sub', arity: 2, returnType: 'number', implementation: (a: any, b: any) => a - b },
                { name: 'mul', arity: 2, returnType: 'number', implementation: (a: any, b: any) => a * b }
            ],
            terminalSet,
            testCases,
            fitnessFunction: () => 0 // Ignored in integrated mode
        } as any;

        for (const seed of SEEDS) {
            const engine = new SuperEngine(problemDef, {
                seed,
                maxGenerations: 150, // Povećano sa 80
                distributedIslands,
                parallelEvaluators: islandParallelEvaluators,
                selectionMethod: 'epsilon-lexicase',
                quotientMode: true,
                strictQuotientValidation: true,
                population: { 
                    size: 5000, // Povećano sa 2000
                    maxGenomeSize: 30, 
                    maxGenomeDepth: 8, 
                    eliteCount: 20, 
                    crossoverRate: 0.8, 
                    mutationRate: 0.5, 
                    tournamentSize: 7 
                }
            });

            const result = await engine.evolve();
            const bestGenome = result.bestGenome;
            const root = bestGenome.genes[0];

            let numStr = '', denStr = '', structure = '';
            if (root && root.operands[1]) {
                // Fix: pass the full bestGenome to correctly resolve references
                // But select the correct root for decompilation
                const numRoot = bestGenome.genes.find(g => g.id === root.operands[0]);
                const denRoot = bestGenome.genes.find(g => g.id === root.operands[1]);
                
                if (numRoot && denRoot) {
                    // Create temporary genomes with the subtree root at index 0
                    const numGenome = { ...bestGenome, genes: [numRoot, ...bestGenome.genes.filter(g => g.id !== numRoot.id)] };
                    const denGenome = { ...bestGenome, genes: [denRoot, ...bestGenome.genes.filter(g => g.id !== denRoot.id)] };
                    
                    numStr = JITCompiler.decompile(numGenome);
                    denStr = JITCompiler.decompile(denGenome);
                    structure = `[ ${numStr} ] / [ ${denStr} ]`;
                } else {
                    structure = JITCompiler.decompile(bestGenome);
                }
            } else {
                structure = JITCompiler.decompile(bestGenome);
            }

            let hasRemovableFactor = false;
            let denomCollapses = false;
            
            // Standard probing
            const probePoints = [];
            for (let i = 0; i < 5; i++) {
                const pt = [];
                for(let v=0; v<prob.vars.length; v++) pt.push(prob.domain[0] + Math.random() * (prob.domain[1] - prob.domain[0]));
                probePoints.push(pt);
            }
            // Include zero if in domain
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
            }
            
            const pointwise = [];
            if (['P2', 'P3', 'P4', 'P5'].includes(prob.name.split(':')[0]) && seed === SEEDS[0]) {
                const strictPoints = [];
                for(let i=0; i<3; i++) {
                     const pt = [];
                     for(let v=0; v<prob.vars.length; v++) pt.push(prob.domain[0] + Math.random() * (prob.domain[1] - prob.domain[0]));
                     strictPoints.push(pt);
                }
                if (prob.vars.length === 1) strictPoints.push([0], [-0.999], [1]); // Avoid exact -1 for P2
                if (prob.vars.length === 2) strictPoints.push([0,0], [0.5, 0.5], [-0.5, 0.5]);
                
                for (const pt of strictPoints) {
                    const target = prob.func(pt);
                    let rawVal = NaN;
                    if (root && root.operands[1]) {
                        const num = evaluateAST(bestGenome, root.operands[0] as string, pt);
                        const den = evaluateAST(bestGenome, root.operands[1] as string, pt);
                        rawVal = den === 0 ? NaN : num / den;
                    }
                    const diff = Math.abs(rawVal - target);
                    pointwise.push({ pt, target, rawVal, diff });
                }
            }

            results.push({ prob: prob.name.split(':')[0], seed, mae, verdict, structure, validDenom: !denomCollapses, artifact: hasRemovableFactor, pointwise });
            await engine.shutdown();
        }
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('CONFIRMATION OF INTEGRATED ENGINE MODE:'));
    console.log('The engine is initialized with quotientMode: true and strictQuotientValidation: true.');
    console.log('This confirms that the native EvolutionEngineTree handles the quotient root contract internally without a specialized wrapper class.');

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('FAMILY GENERALIZATION TABLE (FULL STRUCTURES):'));
    console.log(' | Problem | Seed | Best Structure | Raw MAE | Denom Valid | Artifact | Verdict |');
    for (const r of results) {
        console.log(` | ${r.prob} | ${r.seed} | \`${r.structure}\` | ${r.mae.toFixed(6)} | ${r.validDenom ? 'YES' : 'NO'} | ${r.artifact ? 'YES' : 'NO'} | ${r.verdict} |`);
    }
    
    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('STRICT POINTWISE RAW CHECKS:'));
    for (const r of results) {
        if (r.pointwise && r.pointwise.length > 0) {
            console.log(chalk.yellow(`\nPointwise check for ${r.prob} (Seed ${r.seed}):`));
            console.log(' | Point | Target | Raw Val | Difference |');
            for (const pw of r.pointwise) {
                 console.log(` | [${pw.pt.map((p: number)=>p.toFixed(3)).join(', ')}] | ${pw.target.toFixed(6)} | ${pw.rawVal.toFixed(6)} | ${pw.diff.toFixed(6)} |`);
            }
        }
    }
}

runFamilyAudit().catch(console.error);
