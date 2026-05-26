/**
 * 🕵️ Genetic Code Lab - Rigorous Discovery Audit
 * Testing for Structural Integrity vs Numerical Mimicry.
 */

import { SuperEngine } from '../core/super-engine';
import { SotaBenchmarks } from './quality-benchmarks';
import { JITCompiler } from '../core/jit-compiler';
import { ProblemDefinition, Genome, FunctionDefinition } from '../core/types';

function analyzeStructure(genome: Genome): {
    nodes: number,
    trig: number,
    divs: number,
    consts: number,
    hasRationalCore: boolean
} {
    const nodes = genome.genes.length;
    let trig = 0;
    let divs = 0;
    let consts = 0;

    for (const gene of genome.genes) {
        if (gene.operation === 'sin' || gene.operation === 'cos') trig++;
        if (gene.operation === 'div') divs++;
        if (gene.operation === 'const' || gene.operation === 'constant') consts++;
    }

    const structure = JITCompiler.decompile(genome);
    // Looking for (u+v) or (x0+x1) combined with a division
    const hasRationalCore = /\(.*\+.*\)\s*\//.test(structure);

    return { nodes, trig, divs, consts, hasRationalCore };
}

async function runRigorousAudit() {
    const feynman = SotaBenchmarks.getFeynman153();

    // --- RUN A: CHAOTIC (Trig allowed) ---
    console.error('[Audit] Starting Run A (Chaotic)...');
    const chaoticEngine = new SuperEngine(feynman, {
        maxGenerations: 50,
        population: { size: 1000, maxGenomeSize: 100, maxGenomeDepth: 8, eliteCount: 10, crossoverRate: 0.8, mutationRate: 0.2, tournamentSize: 7 }
    });
    const resultA = await chaoticEngine.evolve();
    const statsA = analyzeStructure(resultA.bestGenome);
    await chaoticEngine.shutdown();

    // --- RUN B: CONSTRAINED (No Trig) ---
    console.error('[Audit] Starting Run B (Constrained)...');
    const constrainedFeynman: ProblemDefinition<number[], number> = {
        ...feynman,
        functionSet: feynman.functionSet.filter((f: FunctionDefinition) => !['sin', 'cos', 'exp', 'log', 'sqrt'].includes(f.name))
    };
    const constrainedEngine = new SuperEngine(constrainedFeynman, {
        maxGenerations: 50,
        population: { size: 1000, maxGenomeSize: 100, maxGenomeDepth: 8, eliteCount: 10, crossoverRate: 0.8, mutationRate: 0.2, tournamentSize: 7 }
    });
    const resultB = await constrainedEngine.evolve();
    const statsB = analyzeStructure(resultB.bestGenome);
    await constrainedEngine.shutdown();

    // --- TABLE OUTPUT ---
    console.log('\n| Run Type | Best Fitness | MAE | Nodes | Trig | Divs | Rational Core | Simplified Structure |');
    console.log('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |');
    
    const row = (type: string, res: any, stats: any) => {
        const mae = (1 / res.bestGenome.fitness - 1).toFixed(6);
        const structure = JITCompiler.decompile(res.bestGenome);
        const core = stats.hasRationalCore ? 'YES' : 'NO';
        const hasTrig = stats.trig > 0 ? 'YES' : 'NO';
        console.log(`| ${type} | ${res.bestGenome.fitness.toFixed(6)} | ${mae} | ${stats.nodes} | ${hasTrig} | ${stats.divs} | ${core} | ${structure.substring(0, 50)}${structure.length > 50 ? '...' : ''} |`);
    };

    row('Chaotic', resultA, statsA);
    row('Constrained', resultB, statsB);
}

runRigorousAudit().catch(console.error);
