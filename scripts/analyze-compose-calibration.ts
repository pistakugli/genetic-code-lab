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
        thresholdAdaptiveCrossCount?: number;
        thresholdAdaptiveMirroredCrossCount?: number;
        thresholdAdaptiveTriadWitnessCount?: number;
        thresholdAdaptiveGuardedUpperRepairCount?: number;
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

type CalibrationPayload = {
    selectionRequested?: string;
    horizons?: number[];
    seeds?: number[];
    modes?: string[];
    aggregateByStrategy: Record<string, StrategySummary>;
};

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

function readInput(): string {
    const file = process.argv[2];
    if (file && file !== '-') return fs.readFileSync(file, 'utf8');
    return fs.readFileSync(0, 'utf8');
}

function round(value: number, digits: number = 8): number {
    return Number(value.toFixed(digits));
}

function classify(base: StrategySummary, duel: StrategySummary): {
    status: 'positive-use-case' | 'tie/no-regression' | 'regression' | 'inconclusive';
    reasons: string[];
} {
    const exactDelta = duel.meanExactAccuracy - base.meanExactAccuracy;
    const boundaryDelta = duel.meanBoundaryTotal - base.meanBoundaryTotal;
    const fitnessDelta = duel.meanFitness - base.meanFitness;
    const hasRouting = (duel.composeDebugTotals?.thresholdAdaptiveDuelCount || 0) > 0;
    const noMaterialRegression = exactDelta >= -0.005 && boundaryDelta <= 2 && fitnessDelta >= -0.005;
    const positiveSignal =
        (exactDelta > 0.001 && boundaryDelta <= 2)
        || (boundaryDelta < -2 && exactDelta >= -0.001)
        || (fitnessDelta > 0.005 && exactDelta >= -0.001 && boundaryDelta <= 2);

    const reasons: string[] = [];
    if (!hasRouting) reasons.push('adaptive-duel had no runtime duel decisions');
    if (exactDelta < -0.005) reasons.push(`mean exact regressed by ${round(exactDelta)}`);
    if (boundaryDelta > 2) reasons.push(`mean boundary total regressed by ${round(boundaryDelta, 4)}`);
    if (fitnessDelta < -0.005) reasons.push(`mean fitness regressed by ${round(fitnessDelta)}`);
    if (positiveSignal) reasons.push('adaptive-duel improved at least one capability metric without material regression');
    if (noMaterialRegression && !positiveSignal) reasons.push('adaptive-duel matched baseline within guard thresholds');

    if (!hasRouting) return { status: 'inconclusive', reasons };
    if (!noMaterialRegression) return { status: 'regression', reasons };
    if (positiveSignal) return { status: 'positive-use-case', reasons };
    return { status: 'tie/no-regression', reasons };
}

function main(): void {
    const payload = parseJsonPayload(readInput());
    const base = payload.aggregateByStrategy.adaptive;
    const duel = payload.aggregateByStrategy['adaptive-duel'];
    if (!base || !duel) {
        throw new Error('expected aggregateByStrategy.adaptive and aggregateByStrategy["adaptive-duel"]');
    }

    const exactDelta = duel.meanExactAccuracy - base.meanExactAccuracy;
    const boundaryDelta = duel.meanBoundaryTotal - base.meanBoundaryTotal;
    const fitnessDelta = duel.meanFitness - base.meanFitness;
    const fp4Delta = duel.meanFp4 - base.meanFp4;
    const fn5Delta = duel.meanFn5 - base.meanFn5;
    const boundaryGapDelta = duel.meanBoundaryGap - base.meanBoundaryGap;
    const classification = classify(base, duel);

    console.log(JSON.stringify({
        status: classification.status,
        reasons: classification.reasons,
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
            boundaryGap: round(boundaryGapDelta, 4)
        },
        baseline: {
            runs: base.runs,
            meanExactAccuracy: base.meanExactAccuracy,
            meanBoundaryTotal: base.meanBoundaryTotal,
            meanFitness: base.meanFitness,
            exactWins: base.exactWins,
            boundaryWins: base.boundaryWins,
            fitnessWins: base.fitnessWins
        },
        adaptiveDuel: {
            runs: duel.runs,
            meanExactAccuracy: duel.meanExactAccuracy,
            meanBoundaryTotal: duel.meanBoundaryTotal,
            meanFitness: duel.meanFitness,
            exactWins: duel.exactWins,
            boundaryWins: duel.boundaryWins,
            fitnessWins: duel.fitnessWins,
            meanComposeDecisionRate: duel.meanComposeDecisionRate || 0,
            meanAdaptivePairedWitnessIncumbentRate: duel.meanAdaptivePairedWitnessIncumbentRate || 0,
            meanAdaptivePairedWitnessVetoRate: duel.meanAdaptivePairedWitnessVetoRate || 0,
            composeDebugTotals: duel.composeDebugTotals || {}
        }
    }, null, 2));
}

try {
    main();
} catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
}
