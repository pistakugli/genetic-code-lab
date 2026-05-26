#!/usr/bin/env ts-node

import path from 'path';
import { spawnSync } from 'child_process';

type Row = {
    proofLane: {
        laneLabel: string;
        composeChooser: string;
        composeExperimentClass: string;
        composeCandidateSet: string[];
        composeConditionalCandidateSet?: string[];
        composeIncumbent?: string | null;
        composeVetoPolicy?: string | null;
    };
    exactAccuracy: number;
    boundary: { total: number; fp4: number; fn5: number };
    composeDebug?: {
        thresholdStaticPairedWitnessComposeCount?: number;
        thresholdAdaptiveDuelCount?: number;
        thresholdAdaptivePairedWitnessCount?: number;
        thresholdAdaptiveCrossCount?: number;
        thresholdAdaptiveMirroredCrossCount?: number;
        thresholdAdaptiveTriadWitnessCount?: number;
        thresholdAdaptiveGuardedUpperRepairCount?: number;
        thresholdAdaptiveWitnessedOrCount?: number;
        thresholdAdaptivePairedWitnessIncumbentCount?: number;
        thresholdAdaptivePairedWitnessVetoCount?: number;
        thresholdAdaptiveCrossVetoCount?: number;
        thresholdAdaptiveMirroredCrossVetoCount?: number;
        thresholdAdaptiveTriadWitnessVetoCount?: number;
        thresholdAdaptiveGuardedCrossVetoCount?: number;
        thresholdAdaptiveGuardedUpperRepairVetoCount?: number;
        thresholdAdaptiveWitnessedOrVetoCount?: number;
        thresholdAdaptiveVetoExactRegressionCount?: number;
        thresholdAdaptiveVetoUpperExactRegressionCount?: number;
        thresholdAdaptiveVetoLowerExactRegressionCount?: number;
        thresholdAdaptiveVetoUpperBoundaryRegressionCount?: number;
        thresholdAdaptiveVetoLowerBoundaryRegressionCount?: number;
        thresholdAdaptiveVetoBoundaryRegressionCount?: number;
        thresholdAdaptiveVetoPressureRegressionCount?: number;
        thresholdAdaptiveVetoWeightedRegressionCount?: number;
        thresholdAdaptiveVetoNoMaterialGainCount?: number;
        thresholdAdaptiveShadowWinnerCount?: number;
        thresholdAdaptiveShadowCrossCount?: number;
        thresholdAdaptiveShadowMirroredCrossCount?: number;
        thresholdAdaptiveShadowTriadWitnessCount?: number;
        thresholdAdaptiveShadowGuardedUpperRepairCount?: number;
        thresholdAdaptiveShadowWitnessedOrCount?: number;
        thresholdAdaptivePromotedWinnerCount?: number;
        thresholdAdaptivePromotedCrossCount?: number;
        thresholdAdaptivePromotedMirroredCrossCount?: number;
        thresholdAdaptivePromotedTriadWitnessCount?: number;
        thresholdAdaptivePromotedGuardedUpperRepairCount?: number;
        thresholdAdaptivePromotedWitnessedOrCount?: number;
        thresholdAdaptivePromotionChampionBlockCount?: number;
    };
};

const repoRoot = path.resolve(__dirname, '..');
const benchmarkScript = path.join(repoRoot, 'scripts', 'benchmark-majority-seeds.ts');

function parseJsonPayload(raw: string): any {
    const trimmed = raw.trim();
    try {
        return JSON.parse(trimmed);
    } catch {
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1));
        throw new Error('benchmark output did not contain parseable JSON');
    }
}

function runBenchmark(envPatch: Record<string, string | undefined>, generations: number): Row {
    const env: NodeJS.ProcessEnv = {
        ...process.env,
        BENCHMARK_SHAPE: '1',
        GCL_THRESHOLD_COMPOSE_EXPERIMENT: '1'
    };
    for (const [key, value] of Object.entries(envPatch)) {
        if (value === undefined) delete env[key];
        else env[key] = value;
    }

    const result = spawnSync(
        path.join(repoRoot, 'node_modules', '.bin', 'ts-node'),
        [benchmarkScript, '9', '5', String(generations), '1337', 'bridge'],
        {
            cwd: repoRoot,
            env,
            encoding: 'utf8',
            maxBuffer: 64 * 1024 * 1024
        }
    );

    if (result.status !== 0) {
        if (result.stdout) process.stdout.write(result.stdout);
        if (result.stderr) process.stderr.write(result.stderr);
        throw new Error(`benchmark-majority-seeds failed with code ${result.status ?? 'unknown'}`);
    }

    return parseJsonPayload(result.stdout).summary.bridge.rows[0] as Row;
}

function main(): void {
    const generations = Number(process.argv[2] || 1);
    const staticRun = runBenchmark({
        GCL_THRESHOLD_UPPER_COMPOSE: 'paired-witness',
        GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT: undefined
    }, generations);
    const duelRun = runBenchmark({
        GCL_THRESHOLD_UPPER_COMPOSE: undefined,
        GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT: '1',
        GCL_THRESHOLD_ADAPTIVE_DUEL_COMMIT: undefined,
        GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION: '0'
    }, generations);

    const staticOk =
        staticRun.proofLane.laneLabel === 'compose-paired-witness'
        && staticRun.proofLane.composeChooser === 'static'
        && staticRun.proofLane.composeExperimentClass === 'single-candidate'
        && (staticRun.composeDebug?.thresholdStaticPairedWitnessComposeCount || 0) > 0
        && (staticRun.composeDebug?.thresholdAdaptiveDuelCount || 0) === 0;

    const duelOk =
        duelRun.proofLane.laneLabel === 'compose-adaptive-duel'
        && duelRun.proofLane.composeChooser === 'duel'
        && duelRun.proofLane.composeExperimentClass === 'adaptive-duel'
        && !duelRun.proofLane.composeCandidateSet.includes('paired-witness')
        && duelRun.proofLane.composeCandidateSet.includes('mirrored-cross')
        && duelRun.proofLane.composeCandidateSet.includes('triad-witness')
        && (duelRun.proofLane.composeConditionalCandidateSet || []).includes('guarded-upper-repair')
        && duelRun.proofLane.composeIncumbent === 'paired-witness'
        && duelRun.proofLane.composeVetoPolicy === 'paired-witness-incumbent'
        && (duelRun.composeDebug?.thresholdAdaptiveDuelCount || 0) > 0
        && (duelRun.composeDebug?.thresholdAdaptivePairedWitnessIncumbentCount || 0) > 0;

    const report = {
        pass: staticOk && duelOk,
        generations,
        static: {
            laneLabel: staticRun.proofLane.laneLabel,
            chooser: staticRun.proofLane.composeChooser,
            class: staticRun.proofLane.composeExperimentClass,
            staticPairedWitness: staticRun.composeDebug?.thresholdStaticPairedWitnessComposeCount || 0,
            adaptiveDuel: staticRun.composeDebug?.thresholdAdaptiveDuelCount || 0,
            exactAccuracy: staticRun.exactAccuracy,
            boundary: staticRun.boundary
        },
        adaptive: {
            laneLabel: duelRun.proofLane.laneLabel,
            chooser: duelRun.proofLane.composeChooser,
            class: duelRun.proofLane.composeExperimentClass,
            candidates: duelRun.proofLane.composeCandidateSet,
            conditionalCandidates: duelRun.proofLane.composeConditionalCandidateSet || [],
            incumbent: duelRun.proofLane.composeIncumbent || null,
            vetoPolicy: duelRun.proofLane.composeVetoPolicy || null,
            adaptiveDuel: duelRun.composeDebug?.thresholdAdaptiveDuelCount || 0,
            adaptivePairedWitness: duelRun.composeDebug?.thresholdAdaptivePairedWitnessCount || 0,
            adaptivePairedWitnessIncumbent: duelRun.composeDebug?.thresholdAdaptivePairedWitnessIncumbentCount || 0,
            adaptivePairedWitnessVeto: duelRun.composeDebug?.thresholdAdaptivePairedWitnessVetoCount || 0,
            adaptiveCross: duelRun.composeDebug?.thresholdAdaptiveCrossCount || 0,
            adaptiveMirroredCross: duelRun.composeDebug?.thresholdAdaptiveMirroredCrossCount || 0,
            adaptiveTriadWitness: duelRun.composeDebug?.thresholdAdaptiveTriadWitnessCount || 0,
            adaptiveGuardedUpperRepair: duelRun.composeDebug?.thresholdAdaptiveGuardedUpperRepairCount || 0,
            adaptiveWitnessedOr: duelRun.composeDebug?.thresholdAdaptiveWitnessedOrCount || 0,
            shadowWinners: {
                total: duelRun.composeDebug?.thresholdAdaptiveShadowWinnerCount || 0,
                cross: duelRun.composeDebug?.thresholdAdaptiveShadowCrossCount || 0,
                mirroredCross: duelRun.composeDebug?.thresholdAdaptiveShadowMirroredCrossCount || 0,
                triadWitness: duelRun.composeDebug?.thresholdAdaptiveShadowTriadWitnessCount || 0,
                guardedUpperRepair: duelRun.composeDebug?.thresholdAdaptiveShadowGuardedUpperRepairCount || 0,
                witnessedOr: duelRun.composeDebug?.thresholdAdaptiveShadowWitnessedOrCount || 0
            },
            promotedWinners: {
                total: duelRun.composeDebug?.thresholdAdaptivePromotedWinnerCount || 0,
                cross: duelRun.composeDebug?.thresholdAdaptivePromotedCrossCount || 0,
                mirroredCross: duelRun.composeDebug?.thresholdAdaptivePromotedMirroredCrossCount || 0,
                triadWitness: duelRun.composeDebug?.thresholdAdaptivePromotedTriadWitnessCount || 0,
                guardedUpperRepair: duelRun.composeDebug?.thresholdAdaptivePromotedGuardedUpperRepairCount || 0,
                witnessedOr: duelRun.composeDebug?.thresholdAdaptivePromotedWitnessedOrCount || 0,
                championBlocks: duelRun.composeDebug?.thresholdAdaptivePromotionChampionBlockCount || 0
            },
            vetoByCandidate: {
                cross: duelRun.composeDebug?.thresholdAdaptiveCrossVetoCount || 0,
                mirroredCross: duelRun.composeDebug?.thresholdAdaptiveMirroredCrossVetoCount || 0,
                triadWitness: duelRun.composeDebug?.thresholdAdaptiveTriadWitnessVetoCount || 0,
                guardedCross: duelRun.composeDebug?.thresholdAdaptiveGuardedCrossVetoCount || 0,
                guardedUpperRepair: duelRun.composeDebug?.thresholdAdaptiveGuardedUpperRepairVetoCount || 0,
                witnessedOr: duelRun.composeDebug?.thresholdAdaptiveWitnessedOrVetoCount || 0
            },
            vetoReasons: {
                exactRegression: duelRun.composeDebug?.thresholdAdaptiveVetoExactRegressionCount || 0,
                upperExactRegression: duelRun.composeDebug?.thresholdAdaptiveVetoUpperExactRegressionCount || 0,
                lowerExactRegression: duelRun.composeDebug?.thresholdAdaptiveVetoLowerExactRegressionCount || 0,
                upperBoundaryRegression: duelRun.composeDebug?.thresholdAdaptiveVetoUpperBoundaryRegressionCount || 0,
                lowerBoundaryRegression: duelRun.composeDebug?.thresholdAdaptiveVetoLowerBoundaryRegressionCount || 0,
                boundaryRegression: duelRun.composeDebug?.thresholdAdaptiveVetoBoundaryRegressionCount || 0,
                pressureRegression: duelRun.composeDebug?.thresholdAdaptiveVetoPressureRegressionCount || 0,
                weightedRegression: duelRun.composeDebug?.thresholdAdaptiveVetoWeightedRegressionCount || 0,
                noMaterialGain: duelRun.composeDebug?.thresholdAdaptiveVetoNoMaterialGainCount || 0
            },
            exactAccuracy: duelRun.exactAccuracy,
            boundary: duelRun.boundary
        }
    };

    console.log(JSON.stringify(report, null, 2));
    if (!report.pass) process.exit(1);
}

try {
    main();
} catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
}
