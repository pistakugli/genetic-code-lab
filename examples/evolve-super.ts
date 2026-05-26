/**
 * 🚀 Genetic Code Lab - SUPER EVOLUTION DEMO
 * Showcasing the full power: Distributed Islands + Agent Steering + JIT Speedup.
 */

import { SuperEngine } from '../src/core/super-engine';
import { SotaBenchmarks } from '../src/tools/quality-benchmarks';
import { JITCompiler } from '../src/core/jit-compiler';
import { formatTime } from '../src/utils/helpers';
import chalk from 'chalk';

async function runSuperDemo() {
    console.log(chalk.cyan.bold('\n🧬 GENETIC CODE LAB: ORCHESTRATED SUPER-EVOLUTION\n'));
    console.log(chalk.white('Target: Beyond SOTA Rational Approximation (F3)'));
    console.log(chalk.gray('Mode: Multi-Island Distributed + Discovery Agent Steering\n'));

    const problem = SotaBenchmarks.getF3();
    
    // 1. Initialise Super Engine (Massive Scale)
    const engine = new SuperEngine(problem, {
        maxGenerations: 50,
        parallelEvaluators: 4, // 4 CPU Cores / Islands
        population: {
            size: 2500, // Total 10,000 individuals across islands
            maxGenomeSize: 150,
            maxGenomeDepth: 10,
            eliteCount: 10,
            crossoverRate: 0.9,
            mutationRate: 0.1,
            tournamentSize: 7
        }
    });

    console.log(chalk.yellow('🚀 Launching Distributed Search...'));
    const result = await engine.evolve();

    console.log(chalk.cyan('\n' + '='.repeat(60)));
    console.log(chalk.white.bold('EVOLUTION COMPLETE'));
    console.log(chalk.white(`  Success: ${result.success ? chalk.green('YES') : chalk.red('NO')}`));
    console.log(chalk.white(`  Best Fitness: ${result.bestGenome.fitness.toFixed(8)}`));
    console.log(chalk.white(`  Total Evaluations: ${result.totalEvaluations.toLocaleString()}`));
    console.log(chalk.white(`  Total Time: ${formatTime(result.totalTimeMs)}`));

    // 2. JIT COMPILATION POWER
    console.log(chalk.yellow('\n⚡ Native JIT Compilation Phase...'));
    const compileStart = Date.now();
    const { fn, constants } = JITCompiler.compile(result.bestGenome);
    const compileTime = Date.now() - compileStart;
    
    console.log(chalk.white(`  Genome compiled to native JS in ${compileTime}ms`));
    
    // Benchmark JIT vs Interpretation (Simulation)
    const benchmarkCycles = 1000000;
    const testInput = [1.5];
    
    console.log(chalk.white(`  Benchmarking ${benchmarkCycles.toLocaleString()} predictions...`));
    
    const jitStart = Date.now();
    for (let i = 0; i < benchmarkCycles; i++) {
        fn(testInput, constants);
    }
    const jitTime = Date.now() - jitStart;
    
    console.log(chalk.green(`  JIT Throughput: ${(benchmarkCycles / (jitTime / 1000)).toLocaleString()} ops/sec`));
    console.log(chalk.white(`  JIT Execution Time: ${jitTime}ms`));

    // 3. FINAL VERDICT
    console.log(chalk.yellow('\n📊 Evolved Solution (Math):'));
    console.log(chalk.white(`  ${JITCompiler.decompile(result.bestGenome)}`));

    await engine.shutdown();
}

runSuperDemo().catch(async (err) => {
    console.error(chalk.red('\nSUPER DEMO FAILED:'), err);
    process.exit(1);
});
