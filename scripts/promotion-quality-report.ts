#!/usr/bin/env ts-node

import fs from 'fs';

type StrategySummary = {
    runs: number;
    meanFitness: number;
    meanExactAccuracy: number;
    meanBoundaryTotal: number;
    meanFp4: number;
    meanFn5: number;
    meanBoundaryGap: number;
    exactWins: number;
    boundaryWins: number;
    fitnessWins: number;
    meanComposeDecisionRate?: number;
    meanAdaptivePairedWitnessIncumbentRate?: number;
    meanAdaptivePairedWitnessVetoRate?: number;
    composeDebugTotals?: {
        thresholdAdaptiveDuelCount?: number;
        thresholdAdaptivePairedWitnessCount?: number;
        thresholdAdaptivePairedWitnessIncumbentCount?: number;
        thresholdAdaptivePairedWitnessVetoCount?: number;
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
        thresholdAdaptiveCrossVetoCount?: number;
        thresholdAdaptiveMirroredCrossVetoCount?: number;
        thresholdAdaptiveTriadWitnessVetoCount?: number;
        thresholdAdaptiveGuardedUpperRepairVetoCount?: number;
        thresholdAdaptiveWitnessedOrVetoCount?: number;
    };
};

type CalibrationPayload = {
    selectionRequested?: string;
    horizons?: number[];
    seeds?: number[];
    modes?: string[];
    aggregateByStrategy: Record<string, StrategySummary>;
};

function readInput(): string {
    const file = process.argv[2];
    if (file && file !== '-') return fs.readFileSync(file, 'utf8');
    return fs.readFileSync(0, 'utf8');
}

function parseJsonPayload(raw: string): CalibrationPayload {
    const trimmed = raw.trim();
    try {
        return JSON.parse(trimmed) as CalibrationPayload;
    } catch {
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1)) as CalibrationPayload;
        throw new Error('calibration output did not contain parseable JSON');
    }
}

function round(value: number, digits: number = 8): number {
    return Number(value.toFixed(digits));
}

function rate(numerator: number, denominator: number): number {
    return Number((numerator / Math.max(1, denominator)).toFixed(4));
}

function main(): void {
    const payload = parseJsonPayload(readInput());
    const baseline = payload.aggregateByStrategy.adaptive;
    const adaptiveDuel = payload.aggregateByStrategy['adaptive-duel'];
    if (!baseline || !adaptiveDuel) {
        throw new Error('expected aggregateByStrategy.adaptive and aggregateByStrategy["adaptive-duel"]');
    }

    const totals = adaptiveDuel.composeDebugTotals || {};
    const exactDelta = adaptiveDuel.meanExactAccuracy - baseline.meanExactAccuracy;
    const boundaryDelta = adaptiveDuel.meanBoundaryTotal - baseline.meanBoundaryTotal;
    const fitnessDelta = adaptiveDuel.meanFitness - baseline.meanFitness;
    const fp4Delta = adaptiveDuel.meanFp4 - baseline.meanFp4;
    const fn5Delta = adaptiveDuel.meanFn5 - baseline.meanFn5;
    const shadowTotal = totals.thresholdAdaptiveShadowWinnerCount || 0;
    const promotedTotal = totals.thresholdAdaptivePromotedWinnerCount || 0;
    const championBlocks = totals.thresholdAdaptivePromotionChampionBlockCount || 0;
    const duelCount = totals.thresholdAdaptiveDuelCount || 0;
    const noRegression = exactDelta >= -0.005 && boundaryDelta <= 2 && fitnessDelta >= -0.005;
    const positiveSignal =
        (exactDelta > 0.001 && boundaryDelta <= 2)
        || (boundaryDelta < -2 && exactDelta >= -0.001)
        || (fitnessDelta > 0.005 && exactDelta >= -0.001 && boundaryDelta <= 2);
    const promotionActive = promotedTotal > 0;
    const recommendation =
        !noRegression ? 'hold-regression'
        : positiveSignal ? 'promote-candidate'
        : promotionActive ? 'keep-opt-in'
        : 'keep-shadow';

    console.log(JSON.stringify({
        recommendation,
        status: positiveSignal ? 'positive-use-case' : noRegression ? 'tie/no-regression' : 'regression',
        selectionRequested: payload.selectionRequested || null,
        horizons: payload.horizons || [],
        seeds: payload.seeds || [],
        modes: payload.modes || [],
        deltas: {
            exactAccuracy: round(exactDelta),
            boundaryTotal: round(boundaryDelta, 4),
            fitness: round(fitnessDelta),
            fp4: round(fp4Delta, 4),
            fn5: round(fn5Delta, 4),
            boundaryGap: round(adaptiveDuel.meanBoundaryGap - baseline.meanBoundaryGap, 4)
        },
        baseline: {
            runs: baseline.runs,
            meanExactAccuracy: baseline.meanExactAccuracy,
            meanBoundaryTotal: baseline.meanBoundaryTotal,
            meanFitness: baseline.meanFitness
        },
        adaptiveDuel: {
            runs: adaptiveDuel.runs,
            meanExactAccuracy: adaptiveDuel.meanExactAccuracy,
            meanBoundaryTotal: adaptiveDuel.meanBoundaryTotal,
            meanFitness: adaptiveDuel.meanFitness,
            meanComposeDecisionRate: adaptiveDuel.meanComposeDecisionRate || 0
        },
        promotionTelemetry: {
            duelCount,
            shadowTotal,
            promotedTotal,
            championBlocks,
            shadowRate: rate(shadowTotal, duelCount),
            promotedRate: rate(promotedTotal, duelCount),
            championBlockRate: rate(championBlocks, shadowTotal + promotedTotal + championBlocks),
            shadowByKind: {
                cross: totals.thresholdAdaptiveShadowCrossCount || 0,
                mirroredCross: totals.thresholdAdaptiveShadowMirroredCrossCount || 0,
                triadWitness: totals.thresholdAdaptiveShadowTriadWitnessCount || 0,
                guardedUpperRepair: totals.thresholdAdaptiveShadowGuardedUpperRepairCount || 0,
                witnessedOr: totals.thresholdAdaptiveShadowWitnessedOrCount || 0
            },
            promotedByKind: {
                cross: totals.thresholdAdaptivePromotedCrossCount || 0,
                mirroredCross: totals.thresholdAdaptivePromotedMirroredCrossCount || 0,
                triadWitness: totals.thresholdAdaptivePromotedTriadWitnessCount || 0,
                guardedUpperRepair: totals.thresholdAdaptivePromotedGuardedUpperRepairCount || 0,
                witnessedOr: totals.thresholdAdaptivePromotedWitnessedOrCount || 0
            },
            vetoByKind: {
                cross: totals.thresholdAdaptiveCrossVetoCount || 0,
                mirroredCross: totals.thresholdAdaptiveMirroredCrossVetoCount || 0,
                triadWitness: totals.thresholdAdaptiveTriadWitnessVetoCount || 0,
                guardedUpperRepair: totals.thresholdAdaptiveGuardedUpperRepairVetoCount || 0,
                witnessedOr: totals.thresholdAdaptiveWitnessedOrVetoCount || 0
            }
        }
    }, null, 2));
}

try {
    main();
} catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
}
