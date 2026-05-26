/**
 * 🏆 Genetic Code Lab - BEYOND SOTA AUDIT
 * Running Pagie-1 (1997) Classic Hard Benchmark
 */

import { SuperEngine } from '../src/core/super-engine';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { formatTime } from '../src/utils/helpers';
import chalk from 'chalk';

async function runAudit() {
    console.log(chalk.cyan.bold('\n=== GENETIC CODE LAB: BEYOND SOTA AUDIT ===\n'));
    console.log(chalk.white('Problem: Feynman I.15.3 (Relativistic Velocity Addition)'));
    console.log(chalk.white('Target Formula: (u+v)/(1+uv)'));
    console.log(chalk.gray('Search: Multi-Island (4) + Discovery Agent Steering\n'));

    const problem = SotaBenchmarks.getFeynman153();
    
    const engine = new SuperEngine(problem, {
        maxGenerations: 150,
        targetFitness: 0.999999,
        parallelEvaluators: 4,
        population: {
            size: 5000, 
            maxGenomeSize: 150,
            maxGenomeDepth: 10,
            eliteCount: 15,
            crossoverRate: 0.8,
            mutationRate: 0.2,
            tournamentSize: 10,
            parsimonyCoefficient: 0.0 // Allow complex structures
        }
    });

    console.log(chalk.yellow('🚀 Launching Distributed Search Core...'));
    const startTime = Date.now();
    const result = await engine.evolve();
    const totalTime = Date.now() - startTime;

    console.log(chalk.cyan('\n' + '='.repeat(60)));
    console.log(chalk.white.bold('AUDIT RESULTS:'));
    console.log(chalk.white(`  Status: ${result.success ? chalk.green('SOLVED') : chalk.yellow('APPROXIMATED')}`));
    console.log(chalk.white(`  Best Fitness: ${result.bestGenome.fitness.toFixed(8)}`));
    console.log(chalk.white(`  Error (MAE): ${(1/result.bestGenome.fitness - 1).toFixed(8)}`));
    console.log(chalk.white(`  Total Evals: ${result.totalEvaluations.toLocaleString()}`));
    console.log(chalk.white(`  Wall Clock: ${formatTime(totalTime)}`));

    console.log(chalk.yellow('\n📊 Evolved Solution Structure:'));
    console.log(chalk.white(`  ${JITCompiler.decompile(result.bestGenome)}`));

    // Performance verification
    console.log(chalk.yellow('\n⚡ Throughput Verification (JIT):'));
    const { fn, constants } = JITCompiler.compile(result.bestGenome);
    const benchmarkCycles = 5000000;
    const jitStart = Date.now();
    for (let i = 0; i < benchmarkCycles; i++) {
        fn([1.2, -0.8], constants);
    }
    const jitTime = Date.now() - jitStart;
    console.log(chalk.green(`  Native Execution Speed: ${(benchmarkCycles / (jitTime / 1000)).toLocaleString()} ops/sec`));

    await engine.shutdown();
}

runAudit().catch(async (err) => {
    console.error(chalk.red('\nAUDIT FAILED:'), err);
    process.exit(1);
});
