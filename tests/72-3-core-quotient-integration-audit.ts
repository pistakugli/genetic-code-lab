/**
 * 🕵️ Phase 72.3: Core Quotient Integration Audit
 */

import { SuperEngine } from '../src/core/super-engine';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { BlockEvaluator } from '../src/core/block-evaluator';
import { Genome, Gene } from '../src/core/types';
import chalk from 'chalk';

// We import the wrapper from 72.2 to test against it
import * as fs from 'fs';
import * as path from 'path';

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

async function runIntegrationAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 72.3: CORE QUOTIENT INTEGRATION AUDIT ===\n'));
    
    const problem = SotaBenchmarks.getFeynman153();
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul', 'div'].includes(f.name)),
        terminalSet: [
            ...problem.terminalSet,
            { name: 'const', type: 'constant', value: 1.0 } as any
        ]
    };

    // We must use 'div' in the grammar for the engine to even try building quotients, 
    // but in quotientMode, the root is implicitly a division. We'll modify the function set
    // for the Integrated engine to NOT have 'div' to match the wrapper's environment.
    const noDivProblem = {
        ...constrainedProblem,
        functionSet: constrainedProblem.functionSet.filter(f => f.name !== 'div')
    };

    const results: any[] = [];

    // --- RUN 1: BASELINE (Standard Engine, Full Rational Grammar) ---
    console.log(chalk.yellow(`\n🚀 Launching BASELINE Run...`));
    BlockEvaluator.divisionMode = 'strict';
    const baselineEngine = new SuperEngine(constrainedProblem, {
        maxGenerations: 100,
        // Deterministic control lane: preserve baseline audit semantics while quotient lanes are compared.
        parallelEvaluators: 1, 
        selectionMethod: 'epsilon-lexicase',
        population: { size: 3000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 15, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
    });
    const resBase = await baselineEngine.evolve();
    results.push(analyze('BASELINE', resBase.bestGenome, false));
    await baselineEngine.shutdown();

    // --- RUN 2: INTEGRATED CORE QUOTIENT ---
    console.log(chalk.yellow(`\n🚀 Launching INTEGRATED CORE QUOTIENT Run...`));
    const integratedEngine = new SuperEngine(noDivProblem, {
        maxGenerations: 100,
        // Deterministic control lane: keep the integrated quotient audit comparable to the baseline run above.
        parallelEvaluators: 1, 
        selectionMethod: 'epsilon-lexicase',
        quotientMode: true,
        strictQuotientValidation: true,
        population: { size: 3000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 15, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
    });
    const resInt = await integratedEngine.evolve();
    results.push(analyze('INTEGRATED', resInt.bestGenome, true));
    await integratedEngine.shutdown();

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('BASELINE VS INTEGRATED TABLE:'));
    console.log(' | Run Mode | Best Structure | Raw MAE | Denom Valid | Artifact Exists | Verdict |');
    for (const r of results) {
        console.log(` | ${r.mode} | \`${r.structure.substring(0, 50)}\` | ${r.mae.toFixed(8)} | ${r.validDenom ? 'YES' : 'NO'} | ${r.artifact ? 'YES' : 'NO'} | ${r.verdict} |`);
    }

    // --- REPRODUCIBILITY CHECK ---
    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('REPRODUCIBILITY TABLE (Integrated Core Quotient):'));
    console.log(' | Seed | Best Structure | Raw MAE | Valid | Artifact | Verdict |');
    
    let successCount = 0;
    for (let i = 1; i <= 5; i++) {
        const seed = 42 + i * 100;
        const repEngine = new SuperEngine(noDivProblem, {
            seed,
            maxGenerations: 100,
            // Deterministic control lane: reproducibility audit stays local until intentionally promoted.
            parallelEvaluators: 1, 
            selectionMethod: 'epsilon-lexicase',
            quotientMode: true,
            strictQuotientValidation: true,
            population: { size: 3000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 15, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
        });
        const res = await repEngine.evolve();
        const stat = analyze(`Seed ${seed}`, res.bestGenome, true);
        
        console.log(` | ${seed} | \`${stat.structure.substring(0, 45)}\` | ${stat.mae.toFixed(8)} | ${stat.validDenom ? 'YES' : 'NO'} | ${stat.artifact ? 'YES' : 'NO'} | ${stat.verdict} |`);
        if (stat.verdict.includes('SYMBOLIC DISCOVERY')) successCount++;
        await repEngine.shutdown();
    }
}

function analyze(mode: string, bestGenome: Genome, isQuotientMode: boolean) {
    if (!bestGenome || !bestGenome.genes || bestGenome.genes.length === 0) {
        return { mode, structure: 'EMPTY', mae: Infinity, validDenom: false, artifact: false, verdict: 'FAIL' };
    }

    const root = bestGenome.genes[0];
    let numStr = '';
    let denStr = '';
    let structure = '';

    if (isQuotientMode && root && root.operands[1]) {
        numStr = JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[0]), ...bestGenome.genes].filter((x): x is Gene => !!x) });
        denStr = JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[1]), ...bestGenome.genes].filter((x): x is Gene => !!x) });
        structure = `[ ${numStr} ] / [ ${denStr} ]`;
    } else {
        structure = JITCompiler.decompile(bestGenome);
    }

    let hasRemovableFactor = false;
    let denomCollapses = false;
    
    const probePoints = [
        [0, 0.5], [0.5, 0], [0, 0], [0.000001, 0.5], [0.5, 0.000001]
    ];
    
    if (isQuotientMode && root && root.operands[1]) {
        for (const probe of probePoints) {
            const numProbe = evaluateAST(bestGenome, root.operands[0] as string, probe);
            const denProbe = evaluateAST(bestGenome, root.operands[1] as string, probe);
            
            if (Math.abs(numProbe) < 1e-5 && Math.abs(denProbe) < 1e-5) hasRemovableFactor = true;
            if (Math.abs(denProbe) < 1e-7) denomCollapses = true;
        }
    } else if (!isQuotientMode && structure.includes('/')) {
        // Very rough check for baseline if it managed to use DIV
         denomCollapses = true; // Assume strictly failing for now unless proven otherwise by a deep parser
    }

    const fitness = bestGenome.fitness;
    const mae = fitness > 0 ? (1 / fitness - 1) : Infinity;
    
    let verdict = 'INVALID RATIONAL CANDIDATE';
    if (!isQuotientMode && !structure.includes('/')) {
        verdict = 'NUMERICAL MIMICRY (Polynomial)';
    } else if (!denomCollapses && !hasRemovableFactor && mae < 0.05) {
        if (mae < 0.0001) verdict = 'RAW VALID SYMBOLIC DISCOVERY';
        else verdict = 'RAW VALID RATIONAL APPROXIMATION';
    } else if (!denomCollapses && !hasRemovableFactor && mae < Infinity) {
         verdict = 'POOR RAW VALID APPROXIMATION';
    }

    return { mode, structure, mae, validDenom: !denomCollapses, artifact: hasRemovableFactor, verdict };
}

runIntegrationAudit().catch(console.error);
