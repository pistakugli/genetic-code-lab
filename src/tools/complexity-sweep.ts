/**
 * 🕵️ Genetic Code Lab - Structural Complexity Sweep
 * Testing Feynman I.15.3 across tight structural constraints [10, 15, 20 nodes].
 */

import { SuperEngine } from '../core/super-engine';
import { SotaBenchmarks } from './quality-benchmarks';
import { JITCompiler } from '../core/jit-compiler';
import { MLP } from '../core/neural-network';
import chalk from 'chalk';

async function runComplexitySweep() {
    console.log(chalk.cyan.bold('\n=== GENETIC CODE LAB: STRUCTURAL COMPLEXITY SWEEP ===\n'));
    console.log(chalk.white('Problem: Feynman I.15.3 -> Target: (u+v)/(1+uv)'));
    
    const problem = SotaBenchmarks.getFeynman153();
    // Constrain the problem to rational operators only to eliminate trig distractions
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul', 'div'].includes(f.name))
    };

    const sizes = [10, 15, 20];
    const resultsTable: any[] = [];

    // Prepare NN Training Data once for all audits
    const X = problem.testCases.map(tc => tc.input as number[]);
    const Y = problem.testCases.map(tc => tc.expectedOutput as number);

    for (const maxSize of sizes) {
        console.log(chalk.yellow(`\n🚀 Launching Sweep Run: maxGenomeSize = ${maxSize}...`));
        
        const engine = new SuperEngine(constrainedProblem, {
            maxGenerations: 50, 
            parallelEvaluators: 1, // Single-core Local Fallback for guaranteed completion
            selectionMethod: 'epsilon-lexicase',
            optimizer: {
                enabled: true,
                iterations: 20,
                learningRate: 0.1,
                interval: 10,
                targetCount: 5
            },
            population: {
                size: 2000, // Sufficient for tiny genomes
                maxGenomeSize: maxSize,
                maxGenomeDepth: 8,
                eliteCount: 10,
                crossoverRate: 0.8,
                mutationRate: 0.5,
                tournamentSize: 10
            }
        });

        const result = await engine.evolve();

        // Check if we got a valid result
        if (result.bestGenome.id === 'empty') {
            console.log(chalk.red(`[Fatal] PIPELINE FAILURE on Size ${maxSize}`));
            await engine.shutdown();
            continue;
        }

        const fitness = result.bestGenome.fitness;
        const mae = fitness > 0 ? (1 / fitness - 1) : Infinity;
        const structure = JITCompiler.decompile(result.bestGenome);

        // Neural Integrity Audit
        const gpFeature = JITCompiler.compile(result.bestGenome).fn;
        const gpConstants = result.bestGenome.genes.filter(g => g.operation === 'const').map(g => g.operands[0] as number);
        const combinedX = X.map(x => [...x, gpFeature(x, gpConstants)]);
        
        const nn = new MLP([3, 4, 1], ['tanh', 'sigmoid']);
        nn.train(combinedX, Y, 500, 0.05, 10, false);
        const nnMAE = 1 - nn.accuracy(combinedX, Y.map(y => y > 0.5 ? 1 : 0)); 

        // Heuristic & Neural Verdict
        const isHonestMatch = structure.includes('/') && structure.includes('x0') && structure.includes('x1');
        const verdict = (isHonestMatch && mae < 0.005 && nnMAE < 0.1) 
            ? 'SYMBOLIC DISCOVERY' 
            : 'NUMERICAL MIMICRY';

        resultsTable.push({
            maxSize,
            fitness: fitness.toFixed(8),
            mae: mae.toFixed(8),
            structure,
            verdict
        });

        await engine.shutdown();
    }

    console.log(chalk.cyan('\n' + '='.repeat(80)));
    console.log(chalk.white.bold('COMPLEXITY SWEEP RESULTS:'));
    console.log(chalk.white(' | Max Nodes | Best Fitness | MAE | Verdict | Final Structure |'));
    console.log(chalk.white(' | :--- | :--- | :--- | :--- | :--- |'));
    for (const res of resultsTable) {
        let cleanStructure = res.structure;
        if (cleanStructure.length > 60) cleanStructure = cleanStructure.substring(0, 57) + '...';
        console.log(` | ${res.maxSize} | ${res.fitness} | ${res.mae} | ${res.verdict} | \`${cleanStructure}\` |`);
    }
}

runComplexitySweep().catch(console.error);
