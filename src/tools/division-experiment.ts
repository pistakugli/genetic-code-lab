/**
 * 🕵️ Genetic Code Lab - Division Penalty Experiment
 * Testing the hypothesis: Does strict division penalty force Taylor expansion mimicry?
 */

import { SuperEngine } from '../core/super-engine';
import { SotaBenchmarks } from './quality-benchmarks';
import { JITCompiler } from '../core/jit-compiler';
import { BlockEvaluator } from '../core/block-evaluator';
import chalk from 'chalk';

async function runDivisionExperiment() {
    console.log(chalk.cyan.bold('\n=== GENETIC CODE LAB: DIVISION PENALTY EXPERIMENT ===\n'));
    console.log(chalk.white('Problem: Feynman I.15.3 -> Target: (u+v)/(1+uv)'));
    console.log(chalk.white('Constraint: Max 15 nodes, NO Trigonometry.\n'));

    const problem = SotaBenchmarks.getFeynman153();
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul', 'div'].includes(f.name))
    };

    const modes: ('strict' | 'mild' | 'safe')[] = ['strict', 'mild', 'safe'];
    const resultsTable: any[] = [];

    for (const mode of modes) {
        console.log(chalk.yellow(`\n🚀 Launching Experiment Run: Division Mode = [${mode.toUpperCase()}]...`));
        
        // Set the global static division mode before evaluating
        BlockEvaluator.divisionMode = mode;

        const engine = new SuperEngine(constrainedProblem, {
            maxGenerations: 60, // Keep it fast, enough for 15 nodes
            parallelEvaluators: 1, // Single-core for deterministic, crash-free execution
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
                maxGenomeSize: 15,
                maxGenomeDepth: 8,
                eliteCount: 10,
                crossoverRate: 0.8,
                mutationRate: 0.5,
                tournamentSize: 10
            }
        });

        const result = await engine.evolve();

        if (result.bestGenome.id === 'empty') {
            console.log(chalk.red(`[Fatal] PIPELINE FAILURE on Mode ${mode}`));
            await engine.shutdown();
            continue;
        }

        const fitness = result.bestGenome.fitness;
        const mae = fitness > 0 ? (1 / fitness - 1) : Infinity;
        const structure = JITCompiler.decompile(result.bestGenome);

        resultsTable.push({
            mode,
            fitness: fitness.toFixed(8),
            mae: mae.toFixed(8),
            structure
        });

        await engine.shutdown();
    }

    console.log(chalk.cyan('\n' + '='.repeat(90)));
    console.log(chalk.white.bold('DIVISION EXPERIMENT RESULTS:'));
    console.log(chalk.white(' | Division Mode | Best Fitness | MAE | Final Structure |'));
    console.log(chalk.white(' | :--- | :--- | :--- | :--- |'));
    for (const res of resultsTable) {
        let cleanStructure = res.structure;
        if (cleanStructure.length > 70) cleanStructure = cleanStructure.substring(0, 67) + '...';
        console.log(` | ${res.mode.toUpperCase()} | ${res.fitness} | ${res.mae} | \`${cleanStructure}\` |`);
    }
}

runDivisionExperiment().catch(console.error);
