/**
 * 🕵️ Phase 72.2: Quotient Search Repair Audit
 * Tests if the search mechanism can autonomously produce raw valid quotients.
 */

import { SuperEngine } from '../src/core/super-engine';
import { EvolutionEngineTree } from '../src/core/evolution-tree';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { Genome, ProblemDefinition, Gene } from '../src/core/types';
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

class CoPrimeQuotientEngine {
    private engine: EvolutionEngineTree;
    public searchPath: any[] = [];

    constructor(baseProblem: ProblemDefinition<any, any>, config: any) {
        const quotientProblem: ProblemDefinition<number[], number> = {
            ...baseProblem,
            functionSet: baseProblem.functionSet.filter(f => ['add', 'sub', 'mul'].includes(f.name)),
            fitnessFunction: (genome: Genome, testCases: any[]) => {
                if (!genome || genome.genes.length === 0) return 0.000001;
                
                const root = genome.genes[0];
                if (root.type === 'terminal' || !root.operands[1]) return 0.000001;

                const nodeMap = new Map();
                for (const g of genome.genes) nodeMap.set(g.id, g);

                let penalty = 0;
                const probePoints = [
                    [0, 0.5], [0.5, 0], [0, 0], [0.1, 0.1], [0.9, 0.9], [0.5, 0.5]
                ];
                
                for (const probe of probePoints) {
                    const numProbe = evaluateAST(genome, root.operands[0] as string, probe, nodeMap);
                    const denProbe = evaluateAST(genome, root.operands[1] as string, probe, nodeMap);
                    
                    const absNum = Math.abs(numProbe);
                    const absDen = Math.abs(denProbe);

                    if (absNum < 0.1 && absDen < 0.1) {
                        penalty += (0.1 - absNum) + (0.1 - absDen); 
                    }
                    
                    if (absDen < 0.1) {
                        penalty += (0.1 - absDen) * 10; 
                    }
                }

                let totalError = 0;
                for (const tc of testCases) {
                    const num = evaluateAST(genome, root.operands[0] as string, tc.input, nodeMap);
                    const rawDenom = evaluateAST(genome, root.operands[1] as string, tc.input, nodeMap);
                    
                    let safeDenom = rawDenom;
                    if (Math.abs(rawDenom) < 1e-3) {
                        safeDenom = rawDenom >= 0 ? 1e-3 : -1e-3;
                        penalty += 1.0; 
                    }
                    
                    const pred = num / safeDenom;
                    if (isNaN(pred) || !isFinite(pred)) {
                        totalError += 10;
                    } else {
                        totalError += Math.abs(pred - tc.expectedOutput);
                    }
                }
                
                const mae = (totalError / testCases.length) + penalty;
                return 1 / (1 + mae);
            }
        };

        const configWithHooks = {
            ...config,
            onGeneration: (pop: any, stats: any) => {
                const best = pop.genomes.reduce((a: any, b: any) => b.fitness > a.fitness ? b : a);
                if (stats.generation === 0 || stats.generation === 1 || best.fitness > (this.searchPath[this.searchPath.length - 1]?.fitness || -Infinity)) {
                    const root = best.genes[0];
                    let structure = '';
                    let isQuotient = false;
                    if (root && root.operands[1]) {
                        const numStr = JITCompiler.decompile({ ...best, genes: [best.genes.find((g: any) => g.id === root.operands[0]), ...best.genes].filter((x: any) => !!x) });
                        const denStr = JITCompiler.decompile({ ...best, genes: [best.genes.find((g: any) => g.id === root.operands[1]), ...best.genes].filter((x: any) => !!x) });
                        structure = `[ ${numStr} ] / [ ${denStr} ]`;
                        isQuotient = true;
                    } else {
                        structure = JITCompiler.decompile(best);
                    }
                    
                    this.searchPath.push({
                        gen: stats.generation,
                        fitness: best.fitness,
                        structure,
                        isQuotient
                    });
                }
            }
        };

        this.engine = new EvolutionEngineTree(quotientProblem as any, configWithHooks);
    }

    public async evolve() {
        return this.engine.evolve();
    }
}

async function runSearchRepairAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 72.2: QUOTIENT SEARCH REPAIR AUDIT (10 RUNS) ===\n'));
    
    const problem = SotaBenchmarks.getFeynman153();
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul'].includes(f.name)),
        terminalSet: [
            ...problem.terminalSet,
            { name: 'const', type: 'constant', value: 1.0 } as any
        ]
    };

    const results = [];
    let successCount = 0;
    let perfectFormCount = 0;
    let successfulSearchPath = null;
    let successfulGenome = null;

    for (let i = 1; i <= 10; i++) {
        const seed = 42 + i * 100;
        
        const pivotEngine = new CoPrimeQuotientEngine(constrainedProblem, {
            seed,
            maxGenerations: 100, 
            population: { size: 3000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 15, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
        });
        
        const result = await pivotEngine.evolve();
        const bestGenome = result.bestGenome;
        const root = bestGenome.genes[0];

        const numStr = root && root.operands[1] ? JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[0]), ...bestGenome.genes].filter((x): x is Gene => !!x) }) : '';
        const denStr = root && root.operands[1] ? JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[1]), ...bestGenome.genes].filter((x): x is Gene => !!x) }) : '';
        const structure = `[ ${numStr} ] / [ ${denStr} ]`;

        let hasRemovableFactor = false;
        let denomCollapses = false;
        
        const probePoints = [
            [0, 0.5], [0.5, 0], [0, 0], [0.000001, 0.5], [0.5, 0.000001]
        ];
        
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
        
        let verdict = 'INVALID RATIONAL CANDIDATE';
        if (!denomCollapses && !hasRemovableFactor && mae < 0.05) {
            if (mae < 0.0001) {
                verdict = 'RAW VALID SYMBOLIC DISCOVERY';
                successCount++;
                if (!successfulSearchPath) {
                    successfulSearchPath = pivotEngine.searchPath;
                    successfulGenome = bestGenome;
                }
                const isTargetForm = structure.includes('(x0 + x1)') || structure.includes('(x1 + x0)') && structure.includes('((x0 * x1) + 1.000)') || structure.includes('((x1 * x0) + 1.000)');
                if (isTargetForm) perfectFormCount++;
            }
            else verdict = 'RAW VALID RATIONAL APPROXIMATION';
        }

        results.push({ seed, mae, verdict, structure, validDenom: !denomCollapses, artifact: hasRemovableFactor });
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('REPRODUCIBILITY TABLE:'));
    console.log(' | Seed | Best Structure | Raw MAE | Denom Valid | Artifact Exists | Verdict |');
    for (const r of results) {
        console.log(` | ${r.seed} | \`${r.structure.substring(0, 50)}\` | ${r.mae.toFixed(8)} | ${r.validDenom ? 'YES' : 'NO'} | ${r.artifact ? 'YES' : 'NO'} | ${r.verdict} |`);
    }

    const maes = results.map(r => r.mae).filter(m => isFinite(m)).sort((a, b) => a - b);
    const meanMae = maes.length > 0 ? maes.reduce((a, b) => a + b, 0) / maes.length : Infinity;
    const medianMae = maes.length > 0 ? maes[Math.floor(maes.length / 2)] : Infinity;

    console.log(`\nSuccess Count: ${successCount} / 10`);
    console.log(`Mean Raw MAE: ${meanMae.toFixed(8)}`);
    console.log(`Median Raw MAE: ${medianMae.toFixed(8)}`);
    console.log(`Perfect Form Equivalents Found: ${perfectFormCount} / 10`);

    if (successfulSearchPath) {
        console.log(chalk.cyan('\n' + '='.repeat(100)));
        console.log(chalk.white.bold('SEARCH PATH EVIDENCE (From a successful run):'));
        console.log(' | Gen | Raw MAE | Quotient? | Structure |');
        for (const sp of successfulSearchPath) {
            const spMae = sp.fitness > 0 ? (1 / sp.fitness - 1) : Infinity;
            console.log(` | ${sp.gen} | ${spMae.toFixed(6)} | ${sp.isQuotient ? 'YES' : 'NO'} | \`${sp.structure.substring(0, 60)}\` |`);
        }
    }

    if (successfulGenome) {
        console.log(chalk.cyan('\n' + '='.repeat(100)));
        console.log(chalk.white.bold('INDEPENDENT VALIDITY DOKAZ (Strict Raw Audit on Winner):'));
        
        const root = successfulGenome.genes[0];
        const testPoints = [
            [0.5, 0], [0.5, 0.000001], [0.5, -0.000001], [0.5, 0.001], [0.5, -0.001], [0.2, 0.3], [0.8, 0.9]
        ];

        console.log(' | u (x0) | v (x1) | Target | Pivot Raw | Difference | Status |');
        let validRun = true;
        for (const pt of testPoints) {
            const u = pt[0];
            const v = pt[1];
            const target = (u + v) / (1 + u * v);

            const num = evaluateAST(successfulGenome, root.operands[0] as string, pt);
            const den = evaluateAST(successfulGenome, root.operands[1] as string, pt);
            
            const rawVal = den === 0 ? NaN : num / den;
            const diff = Math.abs(rawVal - target);
            
            const status = (isNaN(rawVal) || !isFinite(rawVal) || diff > 0.1) ? 'INVALID' : 'VALID';
            if (status === 'INVALID') validRun = false;
            
            console.log(` | ${u.toFixed(6)} | ${v.toFixed(6)} | ${target.toFixed(6)} | ${rawVal.toFixed(6)} | ${diff.toFixed(6)} | ${status} |`);
        }
        console.log(`\nStrict Raw Validity Passes: ${validRun ? 'YES' : 'NO'}`);
    }
}

runSearchRepairAudit().catch(console.error);
