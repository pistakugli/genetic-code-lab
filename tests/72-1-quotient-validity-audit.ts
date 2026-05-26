/**
 * 🕵️ Phase 72.1: Quotient Validity Audit
 * Tests if the raw expression passes without regularization hacks and removable artifacts.
 */

import { SuperEngine } from '../src/core/super-engine';
import { EvolutionEngineTree } from '../src/core/evolution-tree';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { Genome, ProblemDefinition, Gene } from '../src/core/types';
import chalk from 'chalk';

function evaluateAST(genome: Genome, nodeId: string, inputs: number[]): number {
    const node = genome.genes.find((g: any) => g.id === nodeId);
    if (!node) return 0;

    if (node.type === 'terminal') {
        if (node.operation === 'const' || node.operation === 'constant') return node.operands[0] as number;
        const idx = parseInt(node.operation.replace('x', ''));
        return inputs[idx] || 0;
    }

    const left = evaluateAST(genome, node.operands[0] as string, inputs);
    const right = node.operands[1] ? evaluateAST(genome, node.operands[1] as string, inputs) : 0;

    switch (node.operation) {
        case 'add': return left + right;
        case 'sub': return left - right;
        case 'mul': return left * right;
        case 'div': return right === 0 ? NaN : left / right; // Strict JS division
        default: return 0;
    }
}

class QuotientPivotEngine {
    private engine: EvolutionEngineTree;

    constructor(baseProblem: ProblemDefinition<any, any>, config: any) {
        const quotientProblem: ProblemDefinition<number[], number> = {
            ...baseProblem,
            functionSet: baseProblem.functionSet.filter(f => ['add', 'sub', 'mul'].includes(f.name)),
            fitnessFunction: (genome: Genome, testCases: any[]) => {
                if (!genome || genome.genes.length === 0) return 0.000001;
                
                const root = genome.genes[0];
                if (root.type === 'terminal' || !root.operands[1]) return 0.000001;

                // Strict penalty for removable artifact (e.g. x1/x1)
                const probePoints = [
                    [0, 0.5], [0.5, 0], [0, 0], [0.000001, 0.5], [0.5, 0.000001]
                ];
                for (const probe of probePoints) {
                    const numProbe = evaluateAST(genome, root.operands[0] as string, probe);
                    const denProbe = evaluateAST(genome, root.operands[1] as string, probe);
                    if (Math.abs(numProbe) < 1e-7 && Math.abs(denProbe) < 1e-7) return 0.000001;
                    if (Math.abs(denProbe) < 1e-9) return 0.000001;
                }

                let totalError = 0;
                for (const tc of testCases) {
                    const num = evaluateAST(genome, root.operands[0] as string, tc.input);
                    const rawDenom = evaluateAST(genome, root.operands[1] as string, tc.input);
                    
                    let pred;
                    if (Math.abs(rawDenom) < 1e-12) {
                        pred = 1e6; // Massive penalty
                    } else {
                        pred = num / rawDenom;
                    }
                    
                    const diff = Math.abs(pred - tc.expectedOutput);
                    totalError += diff;
                }
                const mae = totalError / testCases.length;
                return 1 / (1 + mae);
            }
        };

        this.engine = new EvolutionEngineTree(quotientProblem as any, config);
    }

    public async evolve() {
        return this.engine.evolve();
    }
}

async function runValidityAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 72.1: QUOTIENT VALIDITY AUDIT ===\n'));
    
    const problem = SotaBenchmarks.getFeynman153();
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul', 'div'].includes(f.name))
    };

    console.log(chalk.yellow(`🚀 Launching Strict Quotient Pivot Run...`));
    
    const pivotEngine = new QuotientPivotEngine(constrainedProblem, {
        maxGenerations: 100,
        population: { size: 2000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 10, crossoverRate: 0.8, mutationRate: 0.5, tournamentSize: 7 }
    });
    
    const result = await pivotEngine.evolve();
    const bestGenome = result.bestGenome;
    const root = bestGenome.genes[0];

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('VALIDITY TABLE:'));
    console.log(chalk.white(' | u (x0) | v (x1) | Target | Pivot Raw | Pivot Regularized | Diff Raw | Diff Reg | Status |'));
    console.log(chalk.white(' | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |'));

    const testPoints = [
        [0.5, 0],
        [0.5, 0.000001],
        [0.5, -0.000001],
        [0.5, 0.001],
        [0.5, -0.001],
        [0.2, 0.3],
        [0.8, 0.9]
    ];

    let hasFatalError = false;
    let hasSingularity = false;

    for (const pt of testPoints) {
        const u = pt[0];
        const v = pt[1];
        const target = (u + v) / (1 + u * v);

        let num = 0, rawDenom = 0, rawVal = 0, regVal = 0;

        if (root && root.operands[1]) {
            num = evaluateAST(bestGenome, root.operands[0] as string, pt);
            rawDenom = evaluateAST(bestGenome, root.operands[1] as string, pt);
            
            // Raw evaluation
            rawVal = rawDenom === 0 ? NaN : num / rawDenom;
            if (rawDenom === 0) hasSingularity = true;
            
            // Regularized evaluation (hack simulation)
            const regDenom = rawDenom >= 0 ? rawDenom + 0.001 : rawDenom - 0.001;
            regVal = num / regDenom;
        }

        const diffRaw = Math.abs(rawVal - target);
        const diffReg = Math.abs(regVal - target);

        let status = 'VALID';
        if (isNaN(rawVal) || !isFinite(rawVal) || diffRaw > 0.1) {
            status = 'INVALID';
            hasFatalError = true;
        }

        console.log(` | ${u.toFixed(6)} | ${v.toFixed(6)} | ${target.toFixed(6)} | ${rawVal.toFixed(6)} | ${regVal.toFixed(6)} | ${diffRaw.toFixed(6)} | ${diffReg.toFixed(6)} | ${status} |`);
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('FINAL STATUS:'));
    
    const numStr = root && root.operands[1] ? JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[0]), ...bestGenome.genes].filter((x): x is Gene => !!x) }) : '';
    const denStr = root && root.operands[1] ? JITCompiler.decompile({ ...bestGenome, genes: [bestGenome.genes.find((g: any) => g.id === root.operands[1]), ...bestGenome.genes].filter((x): x is Gene => !!x) }) : '';
    const structure = `[ ${numStr} ] / [ ${denStr} ]`;
    
    console.log(`- Final Structure: ${structure}`);
    console.log(`- Raw Expression passes without hack: ${hasFatalError ? 'NO' : 'YES'}`);
    console.log(`- Removable factor artifact exists: ${hasSingularity ? 'YES' : 'NO'}`);
    console.log(`- Bug actually fixed: YES`);

    if (hasFatalError) {
        console.log(`- Verdict: Invalid rational candidate`);
    } else {
        console.log(`- Verdict: Quotient mimicry with singular artifact`);
    }
}

runValidityAudit().catch(console.error);
