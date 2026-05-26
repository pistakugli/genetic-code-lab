/**
 * 🕵️ Phase 72: Representation Crisis Pivot Audit
 * Strictly Non-Core implementation.
 */

import { SuperEngine } from '../src/core/super-engine';
import { EvolutionEngineTree } from '../src/core/evolution-tree';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { BlockEvaluator } from '../src/core/block-evaluator';
import { Genome, ProblemDefinition } from '../src/core/types';
import chalk from 'chalk';

function analyzeResult(genome: Genome): { nodes: number, divCount: number, hasRationalCore: boolean } {
    const structure = JITCompiler.decompile(genome);
    const nodes = genome.genes.length;
    let divCount = 0;

    for (const gene of genome.genes) {
        if (gene.operation === 'div') divCount++;
    }

    // Generic check for quotient shape (Numerator / Denominator)
    const hasDivOperator = structure.includes('/');
    const hasBothVars = structure.includes('x0') && structure.includes('x1');
    const hasRationalCore = hasDivOperator && hasBothVars;

    return { nodes, divCount, hasRationalCore };
}

class RepresentationCrisisDetector {
    private trapCount = 0;

    public check(genome: Genome, fitness: number): boolean {
        if (!genome || genome.genes.length === 0) return false;

        const root = genome.genes[0];
        let isQuotient = false;
        let denomNonTrivial = false;

        if (root.operation === 'div') {
            isQuotient = true;
            const rightChildId = root.operands[1];
            const rightChild = genome.genes.find((g: any) => g.id === rightChildId);
            if (rightChild && rightChild.type === 'function') {
                denomNonTrivial = true;
            }
        }

        const isPolyTrap = fitness > 0.95 && fitness < 0.9999 && (!isQuotient || !denomNonTrivial);

        if (isPolyTrap) {
            this.trapCount++;
        } else {
            this.trapCount = 0;
        }

        return this.trapCount > 5;
    }
}

// Safely evaluates an AST node
function evaluateAST(genome: Genome, nodeId: string, inputs: number[]): number {
    const node = genome.genes.find((g: any) => g.id === nodeId);
    if (!node) return 0;

    if (node.type === 'terminal') {
        if (node.operation === 'const' || node.operation === 'constant') return node.operands[0] as number;
        // x0, x1
        const idx = parseInt(node.operation.replace('x', ''));
        return inputs[idx] || 0;
    }

    const left = evaluateAST(genome, node.operands[0] as string, inputs);
    const right = node.operands[1] ? evaluateAST(genome, node.operands[1] as string, inputs) : 0;

    switch (node.operation) {
        case 'add': return left + right;
        case 'sub': return left - right;
        case 'mul': return left * right;
        case 'div': return right === 0 ? 1 : left / right;
        default: return 0;
    }
}

// A custom engine that forces the ROOT to be a Quotient
class QuotientPivotEngine {
    private engine: EvolutionEngineTree;

    constructor(baseProblem: ProblemDefinition<any, any>, config: any) {
        // We redefine the problem to force Quotient representation
        const quotientProblem: ProblemDefinition<number[], number> = {
            ...baseProblem,
            functionSet: baseProblem.functionSet.filter(f => ['add', 'sub', 'mul'].includes(f.name)), // DIV removed from grammar!
            fitnessFunction: (genome: Genome, testCases: any[]) => {
                if (!genome || genome.genes.length === 0) return 0.000001;
                
                // The root node implicitly acts as a pair [Numerator, Denominator]
                // We enforce that the root MUST have 2 children to form a valid quotient
                const root = genome.genes[0];
                if (root.type === 'terminal' || !root.operands[1]) return 0.000001;

                // Penalty for removable artifact (e.g. x1/x1)
                // If both numerator and denominator evaluate to 0 at extreme points, it's a removable factor mimicry.
                const probePoints = [
                    [0, 0.5], [0.5, 0], [0, 0], [0.000001, 0.5], [0.5, 0.000001]
                ];
                
                for (const probe of probePoints) {
                    const numProbe = evaluateAST(genome, root.operands[0] as string, probe);
                    const denProbe = evaluateAST(genome, root.operands[1] as string, probe);
                    if (Math.abs(numProbe) < 1e-7 && Math.abs(denProbe) < 1e-7) {
                        return 0.000001; // FATAL: Removable factor artifact detected
                    }
                    if (Math.abs(denProbe) < 1e-9) {
                        return 0.000001; // FATAL: Singular denominator
                    }
                }

                let totalError = 0;
                for (const tc of testCases) {
                    const num = evaluateAST(genome, root.operands[0] as string, tc.input);
                    const rawDenom = evaluateAST(genome, root.operands[1] as string, tc.input);
                    
                    // We evaluate RAW. No regularization hack during training!
                    let pred;
                    if (Math.abs(rawDenom) < 1e-12) {
                        pred = 1e6; // Massive penalty for dividing by zero
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

async function runCrisisAudit() {
    console.log(chalk.cyan.bold('\n=== PHASE 72: REPRESENTATION CRISIS PIVOT AUDIT ===\n'));
    console.log(chalk.white('Target: Feynman I.15.3 (u+v)/(1+u*v)'));
    console.log(chalk.white('Constraint: No Trig. Strict Division Penalty.\n'));

    const problem = SotaBenchmarks.getFeynman153();
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul', 'div'].includes(f.name))
    };

    // Force strict division penalty to create the polynomial trap
    BlockEvaluator.divisionMode = 'strict';

    const results: any[] = [];

    for (const runMode of ['baseline', 'pivot']) {
        console.log(chalk.yellow(`\n🚀 Launching ${runMode.toUpperCase()} Run...`));
        
        let crisisDetected = false;
        let pivotActivated = false;
        const detector = new RepresentationCrisisDetector();
        
        // Configuration for standard engine chunks
        const chunkSize = 10;
        const maxChunks = 10; // Total 100 generations
        
        let engine = new SuperEngine(constrainedProblem, {
            maxGenerations: chunkSize,
            parallelEvaluators: 1, // Deterministic local run
            selectionMethod: 'epsilon-lexicase',
            optimizer: {
                enabled: true,
                iterations: 20,
                learningRate: 0.1,
                interval: 10,
                targetCount: 5
            },
            population: {
                size: 2000, 
                maxGenomeSize: 20,
                maxGenomeDepth: 10,
                eliteCount: 10,
                crossoverRate: 0.8,
                mutationRate: 0.5,
                tournamentSize: 10
            }
        });

        // Track when crisis was detected and pivoted
        let crisisDetectedGen = -1;
        let pivotActivatedGen = -1;

        let finalResult: any = null;
        let currentGen = 0;

        for (let chunk = 0; chunk < maxChunks; chunk++) {
            const res = await engine.evolve();
            currentGen += chunkSize;
            
            if (res.bestGenome.id === 'empty') {
                finalResult = res;
                break;
            }

            const isCrisis = detector.check(res.bestGenome, res.bestGenome.fitness);
            
            if (isCrisis && !crisisDetected) {
                crisisDetected = true;
                crisisDetectedGen = currentGen;
                console.error(chalk.red(`[Detector] CRISIS DETECTED at Gen ${currentGen}. Polynomial Trap active.`));
                
                if (runMode === 'pivot') {
                    pivotActivated = true;
                    pivotActivatedGen = currentGen;
                    console.error(chalk.magenta(`[Pivot] ACTIVATING GENERIC QUOTIENT MODE.`));
                    
                    // Shut down old engine
                    await engine.shutdown();
                    
                    // Start Quotient Pivot Engine
                    const pivotEngine = new QuotientPivotEngine(constrainedProblem, {
                        maxGenerations: (maxChunks - chunk - 1) * chunkSize,
                        population: { size: 1000, maxGenomeSize: 30, maxGenomeDepth: 8, eliteCount: 10, crossoverRate: 0.8, mutationRate: 0.2, tournamentSize: 7 }
                    });
                    
                    finalResult = await pivotEngine.evolve();
                    break; // Pivot engine runs to completion
                }
            }
            
            finalResult = res;
        }

        if (!pivotActivated) {
            await engine.shutdown();
        }

        const fitness = finalResult.bestGenome.fitness;
        const mae = fitness > 0 ? (1 / fitness - 1) : Infinity;
        const stats = analyzeResult(finalResult.bestGenome);
        
        let structure = '';
        let hasQuotientStructure = false;
        let divCount = 0;

        if (pivotActivated) {
            // Reconstruct the implicit quotient structure for logging
            const root = finalResult.bestGenome.genes[0];
            if (root && root.operands[1]) {
                const num = JITCompiler.decompile({ ...finalResult.bestGenome, genes: [finalResult.bestGenome.genes.find((g: any) => g.id === root.operands[0]), ...finalResult.bestGenome.genes] });
                const den = JITCompiler.decompile({ ...finalResult.bestGenome, genes: [finalResult.bestGenome.genes.find((g: any) => g.id === root.operands[1]), ...finalResult.bestGenome.genes] });
                structure = `[ ${num} ] / [ ${den} ]`;
                hasQuotientStructure = true;
            }
        } else {
            structure = JITCompiler.decompile(finalResult.bestGenome);
            divCount = finalResult.bestGenome.genes?.filter((g: any) => g.operation === 'div').length || 0;
            hasQuotientStructure = structure.includes('/') && structure.includes('x0') && structure.includes('x1');
        }

        // Determine Verdict
        let verdict = 'NUMERICAL MIMICRY';
        if (pivotActivated || hasQuotientStructure) {
            const root = finalResult.bestGenome.genes[0];
            if (root && root.operands[1]) {
                const probePoints = [
                    [0, 0.5], [0.5, 0], [0, 0], [0.000001, 0.5], [0.5, 0.000001], [0.2, 0.3], [0.8, 0.9]
                ];
                let isInvalid = false;
                for (const probe of probePoints) {
                    const numProbe = evaluateAST(finalResult.bestGenome, root.operands[0] as string, probe);
                    const denProbe = evaluateAST(finalResult.bestGenome, root.operands[1] as string, probe);
                    
                    if (Math.abs(numProbe) < 1e-7 && Math.abs(denProbe) < 1e-7) isInvalid = true;
                    if (Math.abs(denProbe) < 1e-9) isInvalid = true;
                    
                    const rawVal = denProbe === 0 ? NaN : numProbe / denProbe;
                    const target = (probe[0] + probe[1]) / (1 + probe[0] * probe[1]);
                    
                    if (isNaN(rawVal) || !isFinite(rawVal) || Math.abs(rawVal - target) > 0.1) {
                        isInvalid = true;
                    }
                }
                
                if (isInvalid) {
                    verdict = 'QUOTIENT MIMICRY WITH SINGULAR ARTIFACT';
                } else if (mae < 0.05) {
                    verdict = 'SYMBOLIC DISCOVERY';
                }
            } else {
                verdict = 'INVALID RATIONAL CANDIDATE';
            }
        }

        results.push({
            mode: runMode,
            fitness: fitness.toFixed(8),
            mae: mae.toFixed(8),
            nodes: stats.nodes,
            divCount: pivotActivated ? 'Implicit 1' : divCount,
            hasStructure: (hasQuotientStructure || pivotActivated) ? 'YES' : 'NO',
            crisisGen: crisisDetectedGen !== -1 ? `YES (Gen ${crisisDetectedGen})` : 'NO',
            pivotGen: pivotActivatedGen !== -1 ? `YES (Gen ${pivotActivatedGen})` : 'NO',
            structure,
            verdict
        });
    }

    console.log(chalk.cyan('\n' + '='.repeat(100)));
    console.log(chalk.white.bold('REPRESENTATION CRISIS PIVOT RESULTS:'));
    console.log(chalk.white(' | Run | Best Fitness | MAE | Nodes | Divs | Quotient Structure | Crisis Detected | Pivot Active | Verdict |'));
    console.log(chalk.white(' | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |'));
    for (const res of results) {
        console.log(` | ${res.mode.toUpperCase()} | ${res.fitness} | ${res.mae} | ${res.nodes} | ${res.divCount} | ${res.hasStructure} | ${res.crisisGen} | ${res.pivotGen} | ${res.verdict} |`);
    }

    console.log(chalk.yellow('\n📊 Final Simplified Structures:'));
    for (const res of results) {
        let clean = res.structure;
        if (clean.length > 80) clean = clean.substring(0, 77) + '...';
        console.log(`  - [${res.mode.toUpperCase()}]: ${clean}`);
    }
}

runCrisisAudit().catch(console.error);
