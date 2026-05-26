import path from 'path';
import { spawnSync } from 'child_process';

type DiagnosePayload = {
    n?: number;
    threshold?: number;
    mode?: string;
    seed?: number | null;
    selectionMethod?: string;
    fitness?: number;
    length?: number;
    byWeight?: Array<{ ones: number; total: number; accuracy: number }>;
    histogram?: Record<string, number>;
    boundaryErrors?: Array<{
        ones: number;
        mask: string;
        expected: number;
        predicted: number;
    }>;
};

const repoRoot = path.resolve(__dirname, '..');
const diagnoseScript = path.join(repoRoot, 'scripts', 'diagnose-threshold.ts');
const analyzerScript = path.join(repoRoot, 'scripts', 'analyze-threshold-boundaries.ts');

function parseNumberArg(value: string | undefined, fallback: number): number {
    if (value === undefined) return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function runCommand(command: string, args: string[], input?: string, stderrMode: 'ignore' | 'inherit' = 'ignore'): string {
    const result = spawnSync(command, args, {
        cwd: repoRoot,
        encoding: 'utf8',
        input,
        maxBuffer: 20 * 1024 * 1024,
        stdio: ['ignore', 'pipe', stderrMode]
    });

    if (result.status !== 0) {
        if (result.stdout) process.stdout.write(result.stdout);
        if (result.stderr) process.stderr.write(result.stderr);
        throw new Error(`${command} ${args.join(' ')} exited with code ${result.status ?? 'unknown'}`);
    }

    return result.stdout;
}

function parsePayload(raw: string): DiagnosePayload {
    const trimmed = raw.trim();
    if (!trimmed) throw new Error('diagnose-threshold produced empty output');
    return JSON.parse(trimmed) as DiagnosePayload;
}

function countBoundaryErrors(payload: DiagnosePayload): { total: number; fp4: number; fn5: number } {
    const boundaryErrors = payload.boundaryErrors || [];
    let fp4 = 0;
    let fn5 = 0;
    for (const err of boundaryErrors) {
        if (err.expected === 0 && err.predicted === 1 && err.ones === 4) fp4++;
        if (err.expected === 1 && err.predicted === 0 && err.ones === 5) fn5++;
    }
    return { total: boundaryErrors.length, fp4, fn5 };
}

function main(): void {
    const seed = parseNumberArg(process.argv[2], 1337);
    const selectionMethod = (process.argv[3] || 'lexicase').toLowerCase();
    const limit = parseNumberArg(process.argv[4], 8);

    const diagnoseArgs = [
        'ts-node',
        diagnoseScript,
        '9',
        '5',
        '140',
        'bridge',
        String(seed),
        selectionMethod,
        'dump-errors'
    ];
    const diagnoseStdout = runCommand('npx', diagnoseArgs, undefined, 'ignore');
    const payload = parsePayload(diagnoseStdout);
    const counts = countBoundaryErrors(payload);

    const analyzerArgs = ['ts-node', analyzerScript, '-', String(limit)];
    const analyzerStdout = runCommand('npx', analyzerArgs, diagnoseStdout, 'ignore').trim();

    const byWeight4 = payload.byWeight?.find(row => row.ones === 4);
    const byWeight5 = payload.byWeight?.find(row => row.ones === 5);

    console.log('Majority-9 boundary summary');
    console.log(`replay: n=9 threshold=5 generations=140 mode=bridge seed=${seed} selection=${selectionMethod}`);
    console.log(`score: fitness=${payload.fitness?.toFixed(8) ?? 'n/a'} length=${payload.length ?? 'n/a'} weight4=${byWeight4?.accuracy?.toFixed(4) ?? 'n/a'} weight5=${byWeight5?.accuracy?.toFixed(4) ?? 'n/a'}`);
    console.log(`boundary errors: total=${counts.total} fp@4=${counts.fp4} fn@5=${counts.fn5}`);
    console.log('');
    console.log(analyzerStdout);
}

try {
    main();
} catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exit(1);
}
