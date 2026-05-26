/**
 * 🕵️ Phase 72.4: Quotient Root Contract Audit
 */

import { SuperEngine } from '../src/core/super-engine';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { Genome, Gene } from '../src/core/types';
import { collectSubtree } from '../src/core/genome';
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

function checkRootContract(genome: Genome): boolean {
    if (!genome || !genome.genes || genome.genes.length === 0) return false;
    const root = genome.genes[0];
    
    // Root must be a div with two operands
    if (root.operation !== 'div' || root.operands.length !== 2) return false;
    
    // Numerator and Denominator subtrees must exist
    const numId = root.operands[0] as string;
    const denId = root.operands[1] as string;
    
    const numSubtree = collectSubtree(genome, numId);
    const denSubtree = collectSubtree(genome, denId);
    
    if (numSubtree.length === 0 || denSubtree.length === 0) return false;
    
    // Check for "leakage" (optional but good): root should not be present in its own children subtrees
    const numIds = numSubtree.map(g => g.id);
    const denIds = denSubtree.map(g => g.id);
    
    if (numIds.includes(root.id) || denIds.includes(root.id)) return false;
    
    return true;
}

async function runRootContractAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 72.4: QUOTIENT ROOT CONTRACT AUDIT ===\n'));
    
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

    for (let i = 1; i <= 5; i++) {
        const seed = 42 + i * 100;
        console.log(chalk.yellow(`🚀 Run ${i} (Seed ${seed})...`));
        
        const engine = new SuperEngine(constrainedProblem, {
            seed,
            maxGenerations: 100,
            // Deterministic control lane: root-contract audit is kept local for reproducible structural checks.
            parallelEvaluators: 1, 
            selectionMethod: 'epsilon-lexicase',
            quotientMode: true,
            strictQuotientValidation: true,
            population: { size: 3000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 15, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
        });
        
        const result = await engine.evolve();
        const bestGenome = result.bestGenome;
        const root = bestGenome.genes[0];

        const contractPreserved = checkRootContract(bestGenome);

        let numStr = '', denStr = '', structure = '';
        if (contractPreserved) {
            numStr = JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[0]), ...bestGenome.genes].filter((x): x is Gene => !!x) });
            denStr = JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[1]), ...bestGenome.genes].filter((x): x is Gene => !!x) });
            structure = `[ ${numStr} ] / [ ${denStr} ]`;
        } else {
            structure = JITCompiler.decompile(bestGenome);
        }

        let hasRemovableFactor = false;
        let denomCollapses = false;
        const probePoints = [[0, 0.5], [0.5, 0], [0, 0], [0.000001, 0.5], [0.5, 0.000001]];
        
        if (contractPreserved) {
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
        if (contractPreserved && !denomCollapses && !hasRemovableFactor && mae < 0.05) {
            if (mae < 0.0001) {
                verdict = 'RAW VALID SYMBOLIC DISCOVERY';
                successCount++;
            } else verdict = 'RAW VALID RATIONAL APPROXIMATION';
        }

        results.push({ seed, mae, verdict, structure, validDenom: !denomCollapses, artifact: hasRemovableFactor, contractPreserved });
        await engine.shutdown();
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('REPRODUCIBILITY TABLE:'));
    console.log(' | Seed | Best Structure | Raw MAE | Denom Valid | Artifact Exists | Contract OK | Verdict |');
    for (const r of results) {
        console.log(` | ${r.seed} | \`${r.structure.substring(0, 45)}\` | ${r.mae.toFixed(8)} | ${r.validDenom ? 'YES' : 'NO'} | ${r.artifact ? 'YES' : 'NO'} | ${r.contractPreserved ? 'YES' : 'NO'} | ${r.verdict} |`);
    }

    console.log(`\nSuccess Count: ${successCount} / 5`);
}

runRootContractAudit().catch(console.error);
