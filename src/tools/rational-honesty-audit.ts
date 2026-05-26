/**
 * 🕵️ Genetic Code Lab - Feynman I.15.3 Rational Honesty Audit
 * Comparing Full Operator Set vs. Constrained Rational Set.
 */

import { SuperEngine } from '../core/super-engine';
import { SotaBenchmarks } from './quality-benchmarks';
import { JITCompiler } from '../core/jit-compiler';
import { Genome } from '../core/types';

interface AuditStats {
    fitness: number;
    mae: number;
    structure: string;
    nodes: number;
    hasTrig: boolean;
    divCount: number;
    hasRationalCore: boolean;
}

function runForensicScan(genome: Genome): AuditStats {
    const structure = JITCompiler.decompile(genome);
    const nodes = genome.genes.length;
    let trig = 0;
    let divs = 0;

    for (const gene of genome.genes) {
        const op = gene.operation.toLowerCase();
        if (op === 'sin' || op === 'cos') trig++;
        if (op === 'div') divs++;
    }

    // Precise check for (u+v)/(1+uv)
    const hasNumerator = structure.includes('x0 + x1') || structure.includes('x1 + x0');
    const hasDenominator = structure.includes('1 + (x0 * x1)') || structure.includes('1 + (x1 * x0)') || structure.includes('1 + (x0*x1)');
    const hasMul = structure.includes('*');
    const hasRationalCore = divs > 0 && hasNumerator && (hasDenominator || hasMul);

    return {
        fitness: genome.fitness,
        mae: 1 / genome.fitness - 1,
        structure, // Now contains the full string
        nodes,
        hasTrig: trig > 0,
        divCount: divs,
        hasRationalCore
    };
}

async function performAudit() {
    const problem = SotaBenchmarks.getFeynman153();

    // --- RUN A: FULL SET ---
    console.error('[Audit] Run A (Full Set)...');
    const engineA = new SuperEngine(problem, {
        maxGenerations: 100,
        parallelEvaluators: 1,
        population: { size: 2500, maxGenomeSize: 150, maxGenomeDepth: 10, eliteCount: 10, crossoverRate: 0.8, mutationRate: 0.2, tournamentSize: 7 }
    });
    const resultA = await engineA.evolve();
    const statsA = runForensicScan(resultA.bestGenome);
    await engineA.shutdown();

    // --- RUN B: CONSTRAINED ---
    console.error('[Audit] Run B (Constrained Rational)...');
    const constrainedProblem = {
        ...problem,
        functionSet: problem.functionSet.filter(f => ['add', 'sub', 'mul', 'div'].includes(f.name))
    };
    const engineB = new SuperEngine(constrainedProblem, {
        maxGenerations: 100,
        parallelEvaluators: 1,
        population: { size: 2500, maxGenomeSize: 150, maxGenomeDepth: 10, eliteCount: 10, crossoverRate: 0.8, mutationRate: 0.2, tournamentSize: 7 }
    });
    const resultB = await engineB.evolve();
    const statsB = runForensicScan(resultB.bestGenome);
    await engineB.shutdown();

    // --- REPORTING ---
    console.log('\n### FEYNMAN I.15.3 RATIONAL HONESTY AUDIT REPORT');
    console.log('| Metric | Run A (Full Set) | Run B (Constrained) |');
    console.log('| :--- | :--- | :--- |');
    console.log(`| Best Fitness | ${statsA.fitness.toFixed(8)} | ${statsB.fitness.toFixed(8)} |`);
    console.log(`| MAE | ${statsA.mae.toFixed(8)} | ${statsB.mae.toFixed(8)} |`);
    console.log(`| Node Count | ${statsA.nodes} | ${statsB.nodes} |`);
    console.log(`| Contains Trig | ${statsA.hasTrig ? 'YES' : 'NO'} | ${statsB.hasTrig ? 'YES' : 'NO'} |`);
    console.log(`| Division Count | ${statsA.divCount} | ${statsB.divCount} |`);
    console.log(`| Rational Core | ${statsA.hasRationalCore ? 'YES' : 'NO'} | ${statsB.hasRationalCore ? 'YES' : 'NO'} |`);
    
    console.log('\n### FULL STRUCTURES');
    console.log(`- RUN A: ${statsA.structure}`);
    console.log(`- RUN B: ${statsB.structure}`);

    console.log('\n### FORENSIC VERDICT');
    
    const isDiscoveryA = statsA.hasRationalCore && statsA.mae < 0.0001;
    const isDiscoveryB = statsB.hasRationalCore && statsB.mae < 0.0001;

    console.log(`- Which run is closer to true formula: ${statsB.mae < statsA.mae ? 'Run B' : 'Run A'}`);
    console.log(`- Run A Result: ${isDiscoveryA ? 'SYMBOLIC DISCOVERY' : 'VERY STRONG RATIONAL APPROXIMATION'}`);
    console.log(`- Run B Result: ${isDiscoveryB ? 'SYMBOLIC DISCOVERY' : 'VERY STRONG RATIONAL APPROXIMATION'}`);
}

performAudit().catch(console.error);
