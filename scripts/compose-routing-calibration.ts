#!/usr/bin/env ts-node

import path from 'path';
import { spawnSync } from 'child_process';

type Mode = 'none' | 'agent' | 'bridge';
type UpperComposeStrategy = 'adaptive' | 'coalition' | 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or' | 'paired-witness';
type CalibrationStrategy = UpperComposeStrategy | 'adaptive-duel';
type SelectionPolicy = 'weighted' | 'hybrid' | 'exact';
type ComposeChooser = 'none' | 'static' | 'duel';
type ComposeExperimentClass = 'none' | 'single-candidate' | 'adaptive-routing' | 'adaptive-duel';

type BenchmarkRow = {
    seed: number;
    mode: Mode;
    proofLane: {
        selectionPolicy: SelectionPolicy;
        selectionExperiment: boolean;
        composeExperiment: boolean;
        composeRequested: UpperComposeStrategy | null;
        composeResolved: UpperComposeStrategy | null;
        upperCompose: UpperComposeStrategy | null;
        adaptiveDuelExperiment: boolean;
        composeChooser: ComposeChooser;
        composeCandidateSet: Array<'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or' | 'paired-witness' | 'coalition'>;
        composeConditionalCandidateSet?: Array<'guarded-upper-repair'>;
        composeIncumbent?: 'paired-witness' | null;
        composeVetoPolicy?: 'paired-witness-incumbent' | null;
        composeExperimentClass: ComposeExperimentClass;
        laneLabel: string;
        proofClass: 'honest' | 'selection-experiment' | 'compose-experiment' | 'compound-experiment';
    };
    parallelEvaluators: number;
    distributedIslands: number;
    topology: string;
    fitness: number;
    exactAccuracy: number;
    length: number;
    lowerBoundary: number;
    upperBoundary: number;
    boundary: {
        total: number;
        fp4: number;
        fn5: number;
    };
    composeDebug?: {
        thresholdComposeFocusCount: number;
        thresholdRoleComposeCount: number;
        thresholdUpperComposeOpportunityCount?: number;
        thresholdAdaptiveDuelCount?: number;
        thresholdAdaptiveCrossCount: number;
        thresholdAdaptiveMirroredCrossCount?: number;
        thresholdAdaptiveTriadWitnessCount?: number;
        thresholdAdaptivePairedWitnessCount?: number;
        thresholdAdaptiveGuardedCrossCount?: number;
        thresholdAdaptiveGuardedUpperRepairCount?: number;
        thresholdAdaptiveWitnessedOrCount: number;
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
        thresholdStaticPairedWitnessComposeCount?: number;
    };
};

type BenchmarkPayload = {
    n: number;
    threshold: number;
    generations: number;
    seeds: number[];
    modes: Mode[];
    summary: Record<string, {
        meanFitness: number;
        medianFitness: number;
        meanLowerBoundary: number;
        meanUpperBoundary: number;
        rows: BenchmarkRow[];
    }>;
};

type CalibrationRow = BenchmarkRow & {
    generations: number;
    strategyRequested: CalibrationStrategy;
    strategyResolved: UpperComposeStrategy | null;
    selectionRequested: SelectionPolicy;
    selectionResolved: SelectionPolicy;
    weakSide: 'lower' | 'upper' | null;
    boundaryGap: number;
    upperMinusLower: number;
    composeShare: {
        cross: number;
        mirroredCross: number;
        triadWitness: number;
        pairedWitness: number;
        guardedCross: number;
        guardedUpperRepair: number;
        witnessedOr: number;
    };
    composeDecisionRate: number;
    staticPairedWitnessComposeRate: number;
    pairedWitnessIncumbentRate: number;
    pairedWitnessVetoRate: number;
};

type AggregateSummary = {
    runs: number;
    meanFitness: number;
    medianFitness: number;
    meanExactAccuracy: number;
    medianExactAccuracy: number;
    meanBoundaryTotal: number;
    medianBoundaryTotal: number;
    meanFp4: number;
    medianFp4: number;
    meanFn5: number;
    medianFn5: number;
    meanLowerBoundary: number;
    meanUpperBoundary: number;
    meanBoundaryGap: number;
    meanUpperMinusLower: number;
    meanLength: number;
    meanAdaptiveCrossShare: number;
    meanAdaptiveMirroredCrossShare: number;
    meanAdaptiveTriadWitnessShare: number;
    meanAdaptivePairedWitnessShare: number;
    meanAdaptiveGuardedCrossShare: number;
    meanAdaptiveGuardedUpperRepairShare: number;
    meanAdaptiveWitnessedOrShare: number;
    meanComposeDecisionRate: number;
    meanStaticPairedWitnessComposeRate: number;
    meanAdaptivePairedWitnessIncumbentRate: number;
    meanAdaptivePairedWitnessVetoRate: number;
    composeDebugTotals: {
        thresholdComposeFocusCount: number;
        thresholdRoleComposeCount: number;
        thresholdUpperComposeOpportunityCount: number;
        thresholdAdaptiveDuelCount: number;
        thresholdAdaptiveCrossCount: number;
        thresholdAdaptiveMirroredCrossCount: number;
        thresholdAdaptiveTriadWitnessCount: number;
        thresholdAdaptivePairedWitnessCount: number;
        thresholdAdaptiveGuardedCrossCount: number;
        thresholdAdaptiveGuardedUpperRepairCount: number;
        thresholdAdaptiveWitnessedOrCount: number;
        thresholdAdaptivePairedWitnessIncumbentCount: number;
        thresholdAdaptivePairedWitnessVetoCount: number;
        thresholdAdaptiveCrossVetoCount: number;
        thresholdAdaptiveMirroredCrossVetoCount: number;
        thresholdAdaptiveTriadWitnessVetoCount: number;
        thresholdAdaptiveGuardedCrossVetoCount: number;
        thresholdAdaptiveGuardedUpperRepairVetoCount: number;
        thresholdAdaptiveWitnessedOrVetoCount: number;
        thresholdAdaptiveVetoExactRegressionCount: number;
        thresholdAdaptiveVetoUpperExactRegressionCount: number;
        thresholdAdaptiveVetoLowerExactRegressionCount: number;
        thresholdAdaptiveVetoUpperBoundaryRegressionCount: number;
        thresholdAdaptiveVetoLowerBoundaryRegressionCount: number;
        thresholdAdaptiveVetoBoundaryRegressionCount: number;
        thresholdAdaptiveVetoPressureRegressionCount: number;
        thresholdAdaptiveVetoWeightedRegressionCount: number;
        thresholdAdaptiveVetoNoMaterialGainCount: number;
        thresholdAdaptiveShadowWinnerCount: number;
        thresholdAdaptiveShadowCrossCount: number;
        thresholdAdaptiveShadowMirroredCrossCount: number;
        thresholdAdaptiveShadowTriadWitnessCount: number;
        thresholdAdaptiveShadowGuardedUpperRepairCount: number;
        thresholdAdaptiveShadowWitnessedOrCount: number;
        thresholdAdaptivePromotedWinnerCount: number;
        thresholdAdaptivePromotedCrossCount: number;
        thresholdAdaptivePromotedMirroredCrossCount: number;
        thresholdAdaptivePromotedTriadWitnessCount: number;
        thresholdAdaptivePromotedGuardedUpperRepairCount: number;
        thresholdAdaptivePromotedWitnessedOrCount: number;
        thresholdAdaptivePromotionChampionBlockCount: number;
        thresholdStaticPairedWitnessComposeCount: number;
    };
    exactWins: number;
    boundaryWins: number;
    fitnessWins: number;
};

type WinnerRecord = {
    generations: number;
    mode: Mode;
    seed: number;
    exactWinners: CalibrationStrategy[];
    boundaryWinners: CalibrationStrategy[];
    fitnessWinners: CalibrationStrategy[];
    fp4Winners: CalibrationStrategy[];
    fn5Winners: CalibrationStrategy[];
    balanceWinners: CalibrationStrategy[];
    rows: Array<{
        strategyRequested: CalibrationStrategy;
        strategyResolved: UpperComposeStrategy | null;
        exactAccuracy: number;
        boundaryTotal: number;
        fp4: number;
        fn5: number;
        fitness: number;
        lowerBoundary: number;
        upperBoundary: number;
        boundaryGap: number;
        length: number;
    }>;
};

const repoRoot = path.resolve(__dirname, '..');
const benchmarkScript = path.join(repoRoot, 'scripts', 'benchmark-majority-seeds.ts');

function parseNumber(value: string | undefined, fallback: number): number {
    if (value === undefined) return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function parseNumberList(value: string | undefined, fallback: number[]): number[] {
    if (!value) return fallback;
    const parsed = value
        .split(',')
        .map(item => Number(item.trim()))
        .filter(item => Number.isFinite(item));
    return parsed.length > 0 ? parsed : fallback;
}

function parseModeList(value: string | undefined, fallback: Mode[]): Mode[] {
    if (!value) return fallback;
    const parsed = value
        .split(',')
        .map(item => item.trim().toLowerCase())
        .filter((item): item is Mode => item === 'none' || item === 'agent' || item === 'bridge');
    return parsed.length > 0 ? parsed : fallback;
}

function parseStrategyList(value: string | undefined, fallback: CalibrationStrategy[]): CalibrationStrategy[] {
    if (!value) return fallback;
    const parsed = value
        .split(',')
        .map(item => item.trim().toLowerCase())
        .map(item => item === 'adaptive_duel' || item === 'adaptiveduel' ? 'adaptive-duel' : item)
        .map(item => item === 'mirrored_cross' || item === 'mirroredcross' ? 'mirrored-cross' : item)
        .map(item => item === 'triad_witness' || item === 'triadwitness' ? 'triad-witness' : item)
        .map(item => item === 'witnessed_or' || item === 'witnessedor' ? 'witnessed-or' : item)
        .map(item => item === 'paired_witness' || item === 'pairedwitness' ? 'paired-witness' : item)
        .filter((item): item is CalibrationStrategy =>
            item === 'adaptive'
            || item === 'adaptive-duel'
            || item === 'coalition'
            || item === 'cross'
            || item === 'mirrored-cross'
            || item === 'triad-witness'
            || item === 'witnessed-or'
            || item === 'paired-witness'
        );
    return parsed.length > 0 ? parsed : fallback;
}

function parseSelectionPolicy(value: string | undefined): SelectionPolicy {
    const normalized = (value || '').trim().toLowerCase();
    if (normalized === 'hybrid' || normalized === 'exact') return normalized;
    return 'weighted';
}

function mean(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    if (sorted.length === 0) return 0;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function parsePayload(raw: string): BenchmarkPayload {
    const trimmed = raw.trim();
    if (!trimmed) throw new Error('benchmark-majority-seeds produced empty output');

    try {
        return JSON.parse(trimmed) as BenchmarkPayload;
    } catch {
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return JSON.parse(trimmed.slice(start, end + 1)) as BenchmarkPayload;
        }
        throw new Error('benchmark-majority-seeds output did not contain parseable JSON');
    }
}

function runBenchmark(
    n: number,
    threshold: number,
    generations: number,
    seeds: number[],
    modes: Mode[],
    selectionPolicy: SelectionPolicy,
    strategyRequested: CalibrationStrategy
): BenchmarkPayload {
    console.error(
        `[compose-calibration] start gens=${generations} strategy=${strategyRequested} `
        + `selection=${selectionPolicy} seeds=${seeds.join(',')} modes=${modes.join(',')}`
    );
    const env = { ...process.env };
    env.BENCHMARK_SHAPE = '1';
    env.GCL_THRESHOLD_COMPOSE_EXPERIMENT = '1';
    env.GCL_THRESHOLD_SELECTION_POLICY = selectionPolicy;

    if (strategyRequested === 'adaptive' || strategyRequested === 'adaptive-duel') {
        delete env.GCL_THRESHOLD_UPPER_COMPOSE;
    } else {
        env.GCL_THRESHOLD_UPPER_COMPOSE = strategyRequested;
    }
    if (strategyRequested === 'adaptive-duel') {
        env.GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT = '1';
    } else if (strategyRequested === 'adaptive') {
        env.GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT = '0';
    } else {
        delete env.GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT;
    }

    const result = spawnSync(
        path.join(repoRoot, 'node_modules', '.bin', 'ts-node'),
        [benchmarkScript, String(n), String(threshold), String(generations), seeds.join(','), modes.join(',')],
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
        throw new Error(
            `benchmark-majority-seeds failed for strategy=${strategyRequested} generations=${generations} `
            + `with code ${result.status ?? 'unknown'}`
        );
    }

    const payload = parsePayload(result.stdout);
    const rows = Object.values(payload.summary).flatMap(modeSummary => modeSummary.rows);
    console.error(
        `[compose-calibration] done gens=${generations} strategy=${strategyRequested} `
        + `meanFitness=${mean(rows.map(row => row.fitness)).toFixed(8)} `
        + `meanExact=${mean(rows.map(row => row.exactAccuracy)).toFixed(8)} `
        + `meanBoundary=${mean(rows.map(row => row.boundary.total)).toFixed(4)}`
    );
    return payload;
}

function flattenRows(
    payload: BenchmarkPayload,
    generations: number,
    selectionRequested: SelectionPolicy,
    strategyRequested: CalibrationStrategy
): CalibrationRow[] {
    return Object.values(payload.summary).flatMap(modeSummary =>
        modeSummary.rows.map(row => {
            const composeTotal =
                (row.composeDebug?.thresholdAdaptiveCrossCount || 0)
                + (row.composeDebug?.thresholdAdaptiveMirroredCrossCount || 0)
                + (row.composeDebug?.thresholdAdaptiveTriadWitnessCount || 0)
                + (row.composeDebug?.thresholdAdaptivePairedWitnessCount || 0)
                + (row.composeDebug?.thresholdAdaptiveGuardedCrossCount || 0)
                + (row.composeDebug?.thresholdAdaptiveGuardedUpperRepairCount || 0)
                + (row.composeDebug?.thresholdAdaptiveWitnessedOrCount || 0);
            const opportunityTotal =
                row.composeDebug?.thresholdUpperComposeOpportunityCount
                ?? row.composeDebug?.thresholdComposeFocusCount
                ?? 0;
            const boundaryGap = Math.abs(row.lowerBoundary - row.upperBoundary);
            const upperMinusLower = Number((row.upperBoundary - row.lowerBoundary).toFixed(4));
            return {
                ...row,
                generations,
                strategyRequested,
                strategyResolved: row.proofLane.composeResolved,
                selectionRequested,
                selectionResolved: row.proofLane.selectionPolicy,
                weakSide:
                    row.lowerBoundary < row.upperBoundary ? 'lower'
                    : row.upperBoundary < row.lowerBoundary ? 'upper'
                    : null,
                boundaryGap: Number(boundaryGap.toFixed(4)),
                upperMinusLower,
                composeShare: {
                    cross: Number((((row.composeDebug?.thresholdAdaptiveCrossCount || 0) / Math.max(1, composeTotal))).toFixed(4)),
                    mirroredCross: Number((((row.composeDebug?.thresholdAdaptiveMirroredCrossCount || 0) / Math.max(1, composeTotal))).toFixed(4)),
                    triadWitness: Number((((row.composeDebug?.thresholdAdaptiveTriadWitnessCount || 0) / Math.max(1, composeTotal))).toFixed(4)),
                    pairedWitness: Number((((row.composeDebug?.thresholdAdaptivePairedWitnessCount || 0) / Math.max(1, composeTotal))).toFixed(4)),
                    guardedCross: Number((((row.composeDebug?.thresholdAdaptiveGuardedCrossCount || 0) / Math.max(1, composeTotal))).toFixed(4)),
                    guardedUpperRepair: Number((((row.composeDebug?.thresholdAdaptiveGuardedUpperRepairCount || 0) / Math.max(1, composeTotal))).toFixed(4)),
                    witnessedOr: Number((((row.composeDebug?.thresholdAdaptiveWitnessedOrCount || 0) / Math.max(1, composeTotal))).toFixed(4))
                },
                composeDecisionRate: Number((((row.composeDebug?.thresholdAdaptiveDuelCount || 0) / Math.max(1, opportunityTotal))).toFixed(4)),
                staticPairedWitnessComposeRate: Number((((row.composeDebug?.thresholdStaticPairedWitnessComposeCount || 0) / Math.max(1, opportunityTotal))).toFixed(4)),
                pairedWitnessIncumbentRate: Number((((row.composeDebug?.thresholdAdaptivePairedWitnessIncumbentCount || 0) / Math.max(1, opportunityTotal))).toFixed(4)),
                pairedWitnessVetoRate: Number((((row.composeDebug?.thresholdAdaptivePairedWitnessVetoCount || 0) / Math.max(1, opportunityTotal))).toFixed(4))
            };
        })
    );
}

function sumComposeDebug(rows: CalibrationRow[]): AggregateSummary['composeDebugTotals'] {
    return rows.reduce((totals, row) => {
        totals.thresholdComposeFocusCount += row.composeDebug?.thresholdComposeFocusCount || 0;
        totals.thresholdRoleComposeCount += row.composeDebug?.thresholdRoleComposeCount || 0;
        totals.thresholdUpperComposeOpportunityCount += row.composeDebug?.thresholdUpperComposeOpportunityCount || 0;
        totals.thresholdAdaptiveDuelCount += row.composeDebug?.thresholdAdaptiveDuelCount || 0;
        totals.thresholdAdaptiveCrossCount += row.composeDebug?.thresholdAdaptiveCrossCount || 0;
        totals.thresholdAdaptiveMirroredCrossCount += row.composeDebug?.thresholdAdaptiveMirroredCrossCount || 0;
        totals.thresholdAdaptiveTriadWitnessCount += row.composeDebug?.thresholdAdaptiveTriadWitnessCount || 0;
        totals.thresholdAdaptivePairedWitnessCount += row.composeDebug?.thresholdAdaptivePairedWitnessCount || 0;
        totals.thresholdAdaptiveGuardedCrossCount += row.composeDebug?.thresholdAdaptiveGuardedCrossCount || 0;
        totals.thresholdAdaptiveGuardedUpperRepairCount += row.composeDebug?.thresholdAdaptiveGuardedUpperRepairCount || 0;
        totals.thresholdAdaptiveWitnessedOrCount += row.composeDebug?.thresholdAdaptiveWitnessedOrCount || 0;
        totals.thresholdAdaptivePairedWitnessIncumbentCount += row.composeDebug?.thresholdAdaptivePairedWitnessIncumbentCount || 0;
        totals.thresholdAdaptivePairedWitnessVetoCount += row.composeDebug?.thresholdAdaptivePairedWitnessVetoCount || 0;
        totals.thresholdAdaptiveCrossVetoCount += row.composeDebug?.thresholdAdaptiveCrossVetoCount || 0;
        totals.thresholdAdaptiveMirroredCrossVetoCount += row.composeDebug?.thresholdAdaptiveMirroredCrossVetoCount || 0;
        totals.thresholdAdaptiveTriadWitnessVetoCount += row.composeDebug?.thresholdAdaptiveTriadWitnessVetoCount || 0;
        totals.thresholdAdaptiveGuardedCrossVetoCount += row.composeDebug?.thresholdAdaptiveGuardedCrossVetoCount || 0;
        totals.thresholdAdaptiveGuardedUpperRepairVetoCount += row.composeDebug?.thresholdAdaptiveGuardedUpperRepairVetoCount || 0;
        totals.thresholdAdaptiveWitnessedOrVetoCount += row.composeDebug?.thresholdAdaptiveWitnessedOrVetoCount || 0;
        totals.thresholdAdaptiveVetoExactRegressionCount += row.composeDebug?.thresholdAdaptiveVetoExactRegressionCount || 0;
        totals.thresholdAdaptiveVetoUpperExactRegressionCount += row.composeDebug?.thresholdAdaptiveVetoUpperExactRegressionCount || 0;
        totals.thresholdAdaptiveVetoLowerExactRegressionCount += row.composeDebug?.thresholdAdaptiveVetoLowerExactRegressionCount || 0;
        totals.thresholdAdaptiveVetoUpperBoundaryRegressionCount += row.composeDebug?.thresholdAdaptiveVetoUpperBoundaryRegressionCount || 0;
        totals.thresholdAdaptiveVetoLowerBoundaryRegressionCount += row.composeDebug?.thresholdAdaptiveVetoLowerBoundaryRegressionCount || 0;
        totals.thresholdAdaptiveVetoBoundaryRegressionCount += row.composeDebug?.thresholdAdaptiveVetoBoundaryRegressionCount || 0;
        totals.thresholdAdaptiveVetoPressureRegressionCount += row.composeDebug?.thresholdAdaptiveVetoPressureRegressionCount || 0;
        totals.thresholdAdaptiveVetoWeightedRegressionCount += row.composeDebug?.thresholdAdaptiveVetoWeightedRegressionCount || 0;
        totals.thresholdAdaptiveVetoNoMaterialGainCount += row.composeDebug?.thresholdAdaptiveVetoNoMaterialGainCount || 0;
        totals.thresholdAdaptiveShadowWinnerCount += row.composeDebug?.thresholdAdaptiveShadowWinnerCount || 0;
        totals.thresholdAdaptiveShadowCrossCount += row.composeDebug?.thresholdAdaptiveShadowCrossCount || 0;
        totals.thresholdAdaptiveShadowMirroredCrossCount += row.composeDebug?.thresholdAdaptiveShadowMirroredCrossCount || 0;
        totals.thresholdAdaptiveShadowTriadWitnessCount += row.composeDebug?.thresholdAdaptiveShadowTriadWitnessCount || 0;
        totals.thresholdAdaptiveShadowGuardedUpperRepairCount += row.composeDebug?.thresholdAdaptiveShadowGuardedUpperRepairCount || 0;
        totals.thresholdAdaptiveShadowWitnessedOrCount += row.composeDebug?.thresholdAdaptiveShadowWitnessedOrCount || 0;
        totals.thresholdAdaptivePromotedWinnerCount += row.composeDebug?.thresholdAdaptivePromotedWinnerCount || 0;
        totals.thresholdAdaptivePromotedCrossCount += row.composeDebug?.thresholdAdaptivePromotedCrossCount || 0;
        totals.thresholdAdaptivePromotedMirroredCrossCount += row.composeDebug?.thresholdAdaptivePromotedMirroredCrossCount || 0;
        totals.thresholdAdaptivePromotedTriadWitnessCount += row.composeDebug?.thresholdAdaptivePromotedTriadWitnessCount || 0;
        totals.thresholdAdaptivePromotedGuardedUpperRepairCount += row.composeDebug?.thresholdAdaptivePromotedGuardedUpperRepairCount || 0;
        totals.thresholdAdaptivePromotedWitnessedOrCount += row.composeDebug?.thresholdAdaptivePromotedWitnessedOrCount || 0;
        totals.thresholdAdaptivePromotionChampionBlockCount += row.composeDebug?.thresholdAdaptivePromotionChampionBlockCount || 0;
        totals.thresholdStaticPairedWitnessComposeCount += row.composeDebug?.thresholdStaticPairedWitnessComposeCount || 0;
        return totals;
    }, {
        thresholdComposeFocusCount: 0,
        thresholdRoleComposeCount: 0,
        thresholdUpperComposeOpportunityCount: 0,
        thresholdAdaptiveDuelCount: 0,
        thresholdAdaptiveCrossCount: 0,
        thresholdAdaptiveMirroredCrossCount: 0,
        thresholdAdaptiveTriadWitnessCount: 0,
        thresholdAdaptivePairedWitnessCount: 0,
        thresholdAdaptiveGuardedCrossCount: 0,
        thresholdAdaptiveGuardedUpperRepairCount: 0,
        thresholdAdaptiveWitnessedOrCount: 0,
        thresholdAdaptivePairedWitnessIncumbentCount: 0,
        thresholdAdaptivePairedWitnessVetoCount: 0,
        thresholdAdaptiveCrossVetoCount: 0,
        thresholdAdaptiveMirroredCrossVetoCount: 0,
        thresholdAdaptiveTriadWitnessVetoCount: 0,
        thresholdAdaptiveGuardedCrossVetoCount: 0,
        thresholdAdaptiveGuardedUpperRepairVetoCount: 0,
        thresholdAdaptiveWitnessedOrVetoCount: 0,
        thresholdAdaptiveVetoExactRegressionCount: 0,
        thresholdAdaptiveVetoUpperExactRegressionCount: 0,
        thresholdAdaptiveVetoLowerExactRegressionCount: 0,
        thresholdAdaptiveVetoUpperBoundaryRegressionCount: 0,
        thresholdAdaptiveVetoLowerBoundaryRegressionCount: 0,
        thresholdAdaptiveVetoBoundaryRegressionCount: 0,
        thresholdAdaptiveVetoPressureRegressionCount: 0,
        thresholdAdaptiveVetoWeightedRegressionCount: 0,
        thresholdAdaptiveVetoNoMaterialGainCount: 0,
        thresholdAdaptiveShadowWinnerCount: 0,
        thresholdAdaptiveShadowCrossCount: 0,
        thresholdAdaptiveShadowMirroredCrossCount: 0,
        thresholdAdaptiveShadowTriadWitnessCount: 0,
        thresholdAdaptiveShadowGuardedUpperRepairCount: 0,
        thresholdAdaptiveShadowWitnessedOrCount: 0,
        thresholdAdaptivePromotedWinnerCount: 0,
        thresholdAdaptivePromotedCrossCount: 0,
        thresholdAdaptivePromotedMirroredCrossCount: 0,
        thresholdAdaptivePromotedTriadWitnessCount: 0,
        thresholdAdaptivePromotedGuardedUpperRepairCount: 0,
        thresholdAdaptivePromotedWitnessedOrCount: 0,
        thresholdAdaptivePromotionChampionBlockCount: 0,
        thresholdStaticPairedWitnessComposeCount: 0
    });
}

function pickWinners(rows: CalibrationRow[], pick: (row: CalibrationRow) => number, higherIsBetter: boolean): CalibrationStrategy[] {
    if (rows.length === 0) return [];
    const values = rows.map(pick);
    const target = higherIsBetter ? Math.max(...values) : Math.min(...values);
    return rows
        .filter(row => Math.abs(pick(row) - target) < 1e-9)
        .map(row => row.strategyRequested)
        .sort();
}

function buildWinnerRecords(rows: CalibrationRow[]): WinnerRecord[] {
    const groups = new Map<string, CalibrationRow[]>();

    for (const row of rows) {
        const key = `${row.generations}:${row.mode}:${row.seed}`;
        const bucket = groups.get(key);
        if (bucket) bucket.push(row);
        else groups.set(key, [row]);
    }

    return [...groups.entries()]
        .map(([key, groupRows]) => {
            const [generationsRaw, modeRaw, seedRaw] = key.split(':');
            return {
                generations: Number(generationsRaw),
                mode: modeRaw as Mode,
                seed: Number(seedRaw),
                exactWinners: pickWinners(groupRows, row => row.exactAccuracy, true),
                boundaryWinners: pickWinners(groupRows, row => row.boundary.total, false),
                fitnessWinners: pickWinners(groupRows, row => row.fitness, true),
                fp4Winners: pickWinners(groupRows, row => row.boundary.fp4, false),
                fn5Winners: pickWinners(groupRows, row => row.boundary.fn5, false),
                balanceWinners: pickWinners(groupRows, row => row.boundaryGap, false),
                rows: groupRows
                    .map(row => ({
                        strategyRequested: row.strategyRequested,
                        strategyResolved: row.strategyResolved,
                        exactAccuracy: row.exactAccuracy,
                        boundaryTotal: row.boundary.total,
                        fp4: row.boundary.fp4,
                        fn5: row.boundary.fn5,
                        fitness: row.fitness,
                        lowerBoundary: row.lowerBoundary,
                        upperBoundary: row.upperBoundary,
                        boundaryGap: row.boundaryGap,
                        length: row.length
                    }))
                    .sort((a, b) => a.strategyRequested.localeCompare(b.strategyRequested))
            };
        })
        .sort((a, b) =>
            a.generations - b.generations
            || a.mode.localeCompare(b.mode)
            || a.seed - b.seed
        );
}

function buildAggregateSummary(rows: CalibrationRow[], winners: WinnerRecord[]): Record<CalibrationStrategy, AggregateSummary> {
    const strategies = [...new Set(rows.map(row => row.strategyRequested))].sort() as CalibrationStrategy[];
    const exactWinCounts = new Map<CalibrationStrategy, number>();
    const boundaryWinCounts = new Map<CalibrationStrategy, number>();
    const fitnessWinCounts = new Map<CalibrationStrategy, number>();

    for (const winner of winners) {
        for (const strategy of winner.exactWinners) {
            exactWinCounts.set(strategy, (exactWinCounts.get(strategy) || 0) + 1);
        }
        for (const strategy of winner.boundaryWinners) {
            boundaryWinCounts.set(strategy, (boundaryWinCounts.get(strategy) || 0) + 1);
        }
        for (const strategy of winner.fitnessWinners) {
            fitnessWinCounts.set(strategy, (fitnessWinCounts.get(strategy) || 0) + 1);
        }
    }

    return Object.fromEntries(
        strategies.map(strategy => {
            const strategyRows = rows.filter(row => row.strategyRequested === strategy);
            return [strategy, {
                runs: strategyRows.length,
                meanFitness: Number(mean(strategyRows.map(row => row.fitness)).toFixed(8)),
                medianFitness: Number(median(strategyRows.map(row => row.fitness)).toFixed(8)),
                meanExactAccuracy: Number(mean(strategyRows.map(row => row.exactAccuracy)).toFixed(8)),
                medianExactAccuracy: Number(median(strategyRows.map(row => row.exactAccuracy)).toFixed(8)),
                meanBoundaryTotal: Number(mean(strategyRows.map(row => row.boundary.total)).toFixed(4)),
                medianBoundaryTotal: Number(median(strategyRows.map(row => row.boundary.total)).toFixed(4)),
                meanFp4: Number(mean(strategyRows.map(row => row.boundary.fp4)).toFixed(4)),
                medianFp4: Number(median(strategyRows.map(row => row.boundary.fp4)).toFixed(4)),
                meanFn5: Number(mean(strategyRows.map(row => row.boundary.fn5)).toFixed(4)),
                medianFn5: Number(median(strategyRows.map(row => row.boundary.fn5)).toFixed(4)),
                meanLowerBoundary: Number(mean(strategyRows.map(row => row.lowerBoundary)).toFixed(4)),
                meanUpperBoundary: Number(mean(strategyRows.map(row => row.upperBoundary)).toFixed(4)),
                meanBoundaryGap: Number(mean(strategyRows.map(row => row.boundaryGap)).toFixed(4)),
                meanUpperMinusLower: Number(mean(strategyRows.map(row => row.upperMinusLower)).toFixed(4)),
                meanLength: Number(mean(strategyRows.map(row => row.length)).toFixed(2)),
                meanAdaptiveCrossShare: Number(mean(strategyRows.map(row => row.composeShare.cross)).toFixed(4)),
                meanAdaptiveMirroredCrossShare: Number(mean(strategyRows.map(row => row.composeShare.mirroredCross)).toFixed(4)),
                meanAdaptiveTriadWitnessShare: Number(mean(strategyRows.map(row => row.composeShare.triadWitness)).toFixed(4)),
                meanAdaptivePairedWitnessShare: Number(mean(strategyRows.map(row => row.composeShare.pairedWitness)).toFixed(4)),
                meanAdaptiveGuardedCrossShare: Number(mean(strategyRows.map(row => row.composeShare.guardedCross)).toFixed(4)),
                meanAdaptiveGuardedUpperRepairShare: Number(mean(strategyRows.map(row => row.composeShare.guardedUpperRepair)).toFixed(4)),
                meanAdaptiveWitnessedOrShare: Number(mean(strategyRows.map(row => row.composeShare.witnessedOr)).toFixed(4)),
                meanComposeDecisionRate: Number(mean(strategyRows.map(row => row.composeDecisionRate)).toFixed(4)),
                meanStaticPairedWitnessComposeRate: Number(mean(strategyRows.map(row => row.staticPairedWitnessComposeRate)).toFixed(4)),
                meanAdaptivePairedWitnessIncumbentRate: Number(mean(strategyRows.map(row => row.pairedWitnessIncumbentRate)).toFixed(4)),
                meanAdaptivePairedWitnessVetoRate: Number(mean(strategyRows.map(row => row.pairedWitnessVetoRate)).toFixed(4)),
                composeDebugTotals: sumComposeDebug(strategyRows),
                exactWins: exactWinCounts.get(strategy) || 0,
                boundaryWins: boundaryWinCounts.get(strategy) || 0,
                fitnessWins: fitnessWinCounts.get(strategy) || 0
            } satisfies AggregateSummary];
        })
    ) as Record<CalibrationStrategy, AggregateSummary>;
}

function main(): void {
    const n = parseNumber(process.argv[2], 9);
    const threshold = parseNumber(process.argv[3], 5);
    const horizons = parseNumberList(process.argv[4], [20, 60]);
    const seeds = parseNumberList(process.argv[5], [1337, 2026, 4242]);
    const modes = parseModeList(process.argv[6], ['bridge']);
    const strategies = parseStrategyList(process.argv[7], ['adaptive', 'adaptive-duel', 'witnessed-or', 'cross', 'mirrored-cross', 'triad-witness', 'coalition', 'paired-witness']);
    const selectionPolicy = parseSelectionPolicy(process.argv[8] || process.env.GCL_THRESHOLD_SELECTION_POLICY);

    const rows: CalibrationRow[] = [];

    for (const generations of horizons) {
        for (const strategy of strategies) {
            const payload = runBenchmark(n, threshold, generations, seeds, modes, selectionPolicy, strategy);
            rows.push(...flattenRows(payload, generations, selectionPolicy, strategy));
        }
    }

    const winners = buildWinnerRecords(rows);
    const aggregateByStrategy = buildAggregateSummary(rows, winners);

    console.log(JSON.stringify({
        problem: { n, threshold },
        selectionRequested: selectionPolicy,
        composeExperiment: true,
        horizons,
        seeds,
        modes,
        strategies,
        rows,
        winners,
        aggregateByStrategy
    }, null, 2));
}

try {
    main();
} catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exit(1);
}
