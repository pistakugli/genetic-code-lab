#!/usr/bin/env ts-node

import fs from 'fs';

type DiagnoseThresholdBoundaryError = {
    ones: number;
    mask: string;
    expected: number;
    predicted: number;
};

type DiagnoseThresholdPayload = {
    n?: number;
    threshold?: number;
    mode?: string;
    seed?: number | null;
    selectionMethod?: string;
    fitness?: number;
    length?: number;
    byWeight?: Array<{ ones: number; total: number; accuracy: number }>;
    histogram?: Record<string, number>;
    boundaryErrors?: DiagnoseThresholdBoundaryError[];
};

type PairCount = {
    key: string;
    count: number;
};

function readInput(): string {
    const arg = process.argv[2];
    const looksNumeric = typeof arg === 'string' && arg.trim() !== '' && Number.isFinite(Number(arg));
    if (arg && arg !== '-' && !looksNumeric) {
        return fs.readFileSync(arg, 'utf8');
    }

    return fs.readFileSync(0, 'utf8');
}

function parsePayload(raw: string): DiagnoseThresholdPayload {
    const trimmed = raw.trim();
    if (!trimmed) throw new Error('No JSON input provided.');

    try {
        return JSON.parse(trimmed) as DiagnoseThresholdPayload;
    } catch {
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return JSON.parse(trimmed.slice(start, end + 1)) as DiagnoseThresholdPayload;
        }
        throw new Error('Input did not contain parseable JSON.');
    }
}

function zeroPadMask(mask: string): string {
    return mask;
}

function getBitPositions(mask: string, bitValue: '0' | '1'): number[] {
    const bits: number[] = [];
    for (let i = 0; i < mask.length; i++) {
        if (mask[i] === bitValue) bits.push(i);
    }
    return bits;
}

function collectPairCounts(masks: string[], bitValue: '0' | '1'): Map<string, number> {
    const counts = new Map<string, number>();
    for (const mask of masks) {
        const positions = getBitPositions(mask, bitValue);
        for (let i = 0; i < positions.length; i++) {
            for (let j = i + 1; j < positions.length; j++) {
                const key = `${positions[i]},${positions[j]}`;
                counts.set(key, (counts.get(key) || 0) + 1);
            }
        }
    }
    return counts;
}

function sortPairCounts(counts: Map<string, number>): PairCount[] {
    return Array.from(counts.entries())
        .map(([key, count]) => ({ key, count }))
        .sort((a, b) => b.count - a.count || a.key.localeCompare(b.key));
}

function formatPairKey(key: string): string {
    const [a, b] = key.split(',').map(Number);
    return `(${a}, ${b})`;
}

function topN<T>(items: T[], n: number): T[] {
    return items.slice(0, n);
}

function printSection(title: string): void {
    console.log(`\n${title}`);
}

function main(): void {
    const inputArg = process.argv[2];
    const looksNumeric = typeof inputArg === 'string' && inputArg.trim() !== '' && Number.isFinite(Number(inputArg));
    const limitArg = inputArg && inputArg !== '-' && !looksNumeric
        ? process.argv[3]
        : (looksNumeric ? inputArg : process.argv[3]);
    const limit = Number(limitArg || 10);
    const raw = readInput();
    const payload = parsePayload(raw);
    const boundaryErrors = Array.isArray(payload.boundaryErrors) ? payload.boundaryErrors : [];

    const byWeight = new Map<number, { total: number; fp: number; fn: number }>();
    const bitCounts = new Map<number, number>();
    const fpMasks: string[] = [];
    const fnMasks: string[] = [];

    for (const err of boundaryErrors) {
        const row = byWeight.get(err.ones) || { total: 0, fp: 0, fn: 0 };
        row.total++;
        if (err.expected === 0 && err.predicted === 1) {
            row.fp++;
            fpMasks.push(zeroPadMask(err.mask));
        } else if (err.expected === 1 && err.predicted === 0) {
            row.fn++;
            fnMasks.push(zeroPadMask(err.mask));
        }
        byWeight.set(err.ones, row);

        for (let i = 0; i < err.mask.length; i++) {
            if (err.mask[i] === '1') bitCounts.set(i, (bitCounts.get(i) || 0) + 1);
        }
    }

    const bitRanking = Array.from(bitCounts.entries())
        .map(([bit, count]) => ({ bit, count }))
        .sort((a, b) => b.count - a.count || a.bit - b.bit);

    const fpZeroPairs = sortPairCounts(collectPairCounts(fpMasks, '0'));
    const fnOnePairs = sortPairCounts(collectPairCounts(fnMasks, '1'));

    console.log('Threshold Boundary Analyzer');
    console.log(`input: n=${payload.n ?? 'unknown'} threshold=${payload.threshold ?? 'unknown'} mode=${payload.mode ?? 'unknown'} seed=${payload.seed ?? 'unknown'} selection=${payload.selectionMethod ?? 'unknown'}`);
    if (typeof payload.fitness === 'number') console.log(`fitness: ${payload.fitness}`);
    if (typeof payload.length === 'number') console.log(`length: ${payload.length}`);
    console.log(`boundary errors: ${boundaryErrors.length}`);

    printSection('Counts by Weight');
    if (byWeight.size === 0) {
        console.log('  none');
    } else {
        for (const ones of Array.from(byWeight.keys()).sort((a, b) => a - b)) {
            const row = byWeight.get(ones)!;
            console.log(`  weight ${ones}: total=${row.total} fp=${row.fp} fn=${row.fn}`);
        }
    }

    printSection('Top Bit Frequencies');
    if (bitRanking.length === 0) {
        console.log('  none');
    } else {
        for (const entry of topN(bitRanking, limit)) {
            console.log(`  bit ${entry.bit}: ${entry.count}`);
        }
    }

    printSection('FP Zero-Pairs');
    if (fpZeroPairs.length === 0) {
        console.log('  none');
    } else {
        for (const entry of topN(fpZeroPairs, limit)) {
            console.log(`  ${formatPairKey(entry.key)}: ${entry.count}`);
        }
    }

    printSection('FN One-Pairs');
    if (fnOnePairs.length === 0) {
        console.log('  none');
    } else {
        for (const entry of topN(fnOnePairs, limit)) {
            console.log(`  ${formatPairKey(entry.key)}: ${entry.count}`);
        }
    }
}

try {
    main();
} catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exit(1);
}
