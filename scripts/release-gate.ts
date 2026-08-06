#!/usr/bin/env ts-node

import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';

type StepOptions = {
    env?: NodeJS.ProcessEnv;
    timeoutMs?: number;
    captureStdout?: boolean;
};

type AnalysisReport = {
    status: 'positive-use-case' | 'tie/no-regression' | 'regression' | 'inconclusive';
    deltas: {
        exactAccuracy: number;
        boundaryTotal: number;
        fitness: number;
    };
};

type PromotionQualityReport = {
    recommendation: 'hold-regression' | 'promote-candidate' | 'keep-opt-in' | 'keep-shadow';
    status: 'positive-use-case' | 'tie/no-regression' | 'regression';
    deltas: {
        exactAccuracy: number;
        boundaryTotal: number;
        fitness: number;
    };
    promotionTelemetry: {
        duelCount: number;
        shadowTotal: number;
        promotedTotal: number;
        championBlocks: number;
        promotedByKind: Record<string, number>;
    };
};

const repoRoot = path.resolve(__dirname, '..');
const tsNode = path.join(repoRoot, 'node_modules', '.bin', 'ts-node');
const workDir = path.join(os.tmpdir(), 'gcl-release-gate');
const calibrationPath = path.join(workDir, 'promotion-calibration.json');
const analysisPath = path.join(workDir, 'promotion-analysis.json');
const qualityPath = path.join(workDir, 'promotion-quality.json');

function parseJsonPayload<T>(raw: string): T {
    const trimmed = raw.trim();
    try {
        return JSON.parse(trimmed) as T;
    } catch {
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) return JSON.parse(trimmed.slice(start, end + 1)) as T;
        throw new Error('command output did not contain parseable JSON');
    }
}

function printHeader(name: string): void {
    console.log(`\n[release-gate] ${name}`);
}

function runStep(name: string, command: string, args: string[], options: StepOptions = {}): string {
    printHeader(name);
    const env = { ...process.env, ...(options.env || {}) };
    if (options.env) {
        for (const [key, value] of Object.entries(options.env)) {
            if (value === undefined) delete env[key];
        }
    }
    const result = spawnSync(command, args, {
        cwd: repoRoot,
        env,
        encoding: 'utf8',
        timeout: options.timeoutMs || 10 * 60 * 1000,
        maxBuffer: 256 * 1024 * 1024
    });

    if (!options.captureStdout && result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);

    if (result.error) throw result.error;
    if (result.status !== 0) {
        if (options.captureStdout && result.stdout) process.stdout.write(result.stdout);
        throw new Error(`${name} failed with exit code ${result.status ?? 'unknown'}`);
    }
    return result.stdout || '';
}

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) throw new Error(message);
}

function assertNoRegression(report: AnalysisReport, source: string): void {
    assert(report.status !== 'regression', `${source} reported regression`);
    assert(report.status !== 'inconclusive', `${source} was inconclusive`);
    assert(report.deltas.exactAccuracy >= -0.000001, `${source} exact accuracy regressed: ${report.deltas.exactAccuracy}`);
    assert(report.deltas.boundaryTotal <= 0.000001, `${source} boundary total regressed: ${report.deltas.boundaryTotal}`);
    assert(report.deltas.fitness >= -0.000001, `${source} fitness regressed: ${report.deltas.fitness}`);
}

function main(): void {
    fs.mkdirSync(workDir, { recursive: true });

    runStep('build', 'npm', ['run', 'build'], { timeoutMs: 2 * 60 * 1000 });
    // Measured at 126.5 seconds for 172 tests on this machine, against a two-minute budget -
    // so the gate had never reached step three, and every claim that it protects the eighty
    // steps below was unverified. Raised from measurement with headroom, not to a round number
    // that happens to pass.
    runStep('unit/integration tests', 'npm', ['test', '--', '--runInBand'], { timeoutMs: 6 * 60 * 1000 });
    // tsconfig.json includes only src/**, so the build step above never type checked the
    // benchmark and gate entrypoints under scripts/. They were checked only when ts-node ran
    // them, which let a type error hide in any gate that had not been invoked recently.
    runStep('scripts typecheck', 'npm', ['run', 'typecheck:scripts'], { timeoutMs: 3 * 60 * 1000 });
    runStep('default threshold policy invariant', 'npm', ['run', 'verify:default-threshold-policy'], { timeoutMs: 3 * 60 * 1000 });
    runStep('Santa Fe trail verifier', 'npm', ['run', 'verify:santa-fe-trail'], { timeoutMs: 2 * 60 * 1000 });
    runStep('delegated citation integrity', 'npm', ['run', 'verify:delegated-citations'], { timeoutMs: 3 * 60 * 1000 });
    runStep('artifact verdict consistency', 'npm', ['run', 'verify:artifact-verdicts'], { timeoutMs: 3 * 60 * 1000 });
    runStep('artifact provenance', 'npm', ['run', 'verify:artifact-provenance'], { timeoutMs: 3 * 60 * 1000 });
    // Four classes of defect that each reached a commit during one day of work on the
    // Jacobian route, none of which any other verifier could see: a swallowed failure, an
    // artifact outliving its generator, an access audit made of self-written literals, and a
    // run writing outside its own lane.
    runStep('solver integrity', 'npm', ['run', 'verify:solver-integrity'], { timeoutMs: 3 * 60 * 1000 });
    runStep('variant distinctness', 'npm', ['run', 'verify:variant-distinctness'], { timeoutMs: 3 * 60 * 1000 });
    runStep('semantic archive composition benchmark', 'npm', ['run', 'benchmark:semantic-archive'], { timeoutMs: 2 * 60 * 1000 });
    runStep('closed-loop archive emergence benchmark', 'npm', ['run', 'benchmark:archive-emergence'], { timeoutMs: 3 * 60 * 1000 });
    runStep('multi-island NN archive benchmark', 'npm', ['run', 'benchmark:island-nn-archive:wide'], { timeoutMs: 10 * 60 * 1000 });
    runStep('distributed worker-island NN benchmark', 'npm', ['run', 'benchmark:distributed-island-nn:wide'], { timeoutMs: 10 * 60 * 1000 });
    runStep('NN symbolic distillation benchmark', 'npm', ['run', 'benchmark:nn-distill'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN circuit distillation benchmark', 'npm', ['run', 'benchmark:nn-circuit-distill'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN ReLU circuit distillation benchmark', 'npm', ['run', 'benchmark:nn-relu-circuit-distill'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-blackbox-relu-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN deep black-box ReLU 1D decompile benchmark', 'npm', ['run', 'benchmark:nn-deep-blackbox-relu1d-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-deep-blackbox-relu2d-region-graph'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN randomized deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-deep-blackbox-relu2d-region-graph:randomized'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-noisy-deep-blackbox-relu2d-region-graph'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN adaptive-adversarial deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN compact deep black-box ReLU 2D boundary benchmark', 'npm', ['run', 'benchmark:nn-compact-deep-blackbox-relu2d-boundary'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN adaptive-budget deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN boundary-refined deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN subcell-boundary adaptive deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN topology-boundary deep black-box ReLU 2D region-graph benchmark', 'npm', ['run', 'benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN deep black-box ReLU ND region-sampler benchmark', 'npm', ['run', 'benchmark:nn-deep-blackbox-relu-nd-region-sampler'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN noisy adaptive-adversarial deep black-box ReLU ND region-sampler benchmark', 'npm', ['run', 'benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler'], { timeoutMs: 6 * 60 * 1000 });
    runStep('NN adaptive-budget noisy deep black-box ReLU ND region-sampler benchmark', 'npm', ['run', 'benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN ND adjacency-geometry deep black-box ReLU region-sampler benchmark', 'npm', ['run', 'benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN ND adjacency-geometry 5D sweep benchmark', 'npm', ['run', 'benchmark:nn-nd-adjacency-geometry-5d-sweep'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN ND adjacency-geometry 6D robustness benchmark', 'npm', ['run', 'benchmark:nn-nd-adjacency-geometry-6d-robustness'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN ND adjacency-geometry 6D baseline court benchmark', 'npm', ['run', 'benchmark:nn-nd-adjacency-geometry-6d-baselines'], { timeoutMs: 6 * 60 * 1000 });
    runStep('NN active-subspace FFN decompiler benchmark', 'npm', ['run', 'benchmark:nn-active-subspace-ffn-decompiler'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN active-subspace FFN 4096D stress benchmark', 'npm', ['run', 'benchmark:nn-active-subspace-ffn-4096d-stress'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy active-subspace FFN benchmark', 'npm', ['run', 'benchmark:nn-active-subspace-ffn-noisy'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN GELU FFN structure finder benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-structure-finder'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN rank-adaptive mixed GELU FFN structure benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-mixed-structure'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN near-parallel mixed GELU FFN family benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-near-parallel-family'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN GELU FFN cross-decomposition invariance benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-cross-decomposition-invariance'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN GELU FFN formalization negative-case benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-negative-cases'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN GELU FFN rank-cap negative guard benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-rank-cap-negative'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN noisy mixed GELU FFN structure benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-noisy-mixed-structure'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN mixed SwiGLU FFN structure benchmark', 'npm', ['run', 'benchmark:nn-swiglu-ffn-mixed-structure'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN noisy mixed SwiGLU FFN structure benchmark', 'npm', ['run', 'benchmark:nn-swiglu-ffn-noisy-mixed-structure'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN shared-gate cancellation SwiGLU canonical benchmark', 'npm', ['run', 'benchmark:nn-swiglu-ffn-shared-gate-canonical'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN GELU FFN source-neuron provenance verifier benchmark', 'npm', ['run', 'benchmark:nn-gelu-ffn-provenance-verifier'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN local transformer MLP hook benchmark', 'npm', ['run', 'benchmark:nn-local-transformer-mlp-hook'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN real GPT-2 GELU FFN source-structure benchmark', 'npm', ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN real GPT-2 GELU FFN top-5 blind source-structure benchmark', 'npm', ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure:top5'], { timeoutMs: 7 * 60 * 1000 });
    runStep('NN real GPT-2 GELU FFN top-10 blind source-structure benchmark', 'npm', ['run', 'benchmark:nn-gpt2-gelu-ffn-real-structure:top10'], { timeoutMs: 12 * 60 * 1000 });
    runStep('NN real GPT-2 GELU FFN source-extractor benchmark', 'npm', ['run', 'benchmark:nn-gpt2-gelu-ffn-source-extractor'], { timeoutMs: 2 * 60 * 1000 });
    runStep('NN live GPT-2 ordered residual path circuit finder benchmark', 'npm', ['run', 'benchmark:nn-live-gpt2-ordered-residual-path-circuit-finder:gpu'], { timeoutMs: 12 * 60 * 1000 });
    runStep('NN live Pythia ordered residual path circuit finder benchmark', 'npm', ['run', 'benchmark:nn-live-pythia-ordered-residual-path-circuit-finder:gpu'], { timeoutMs: 15 * 60 * 1000 });
    runStep('NN live Llama SwiGLU ordered residual path circuit finder benchmark', 'npm', ['run', 'benchmark:nn-live-llama-swiglu-ordered-residual-path-circuit-finder:gpu'], { timeoutMs: 30 * 60 * 1000 });
    runStep('NN live Mistral SwiGLU ordered residual path circuit finder benchmark', 'npm', ['run', 'benchmark:nn-live-mistral-swiglu-ordered-residual-path-circuit-finder:gpu'], { timeoutMs: 45 * 60 * 1000 });
    runStep('NN live ordered path stability benchmark', 'npm', ['run', 'benchmark:nn-live-ordered-path-stability:gpu'], { timeoutMs: 45 * 60 * 1000 });
    // The strongest result the project has and nothing was guarding it. 64 units of layer 15
    // recovered from oracle calls alone - gate direction, up direction and gate norm together
    // - checked against a held-out reconstruction that needs no weights. Runs in ninety
    // seconds, so there is no reason for it to sit outside the gate.
    runStep('NN live Jacobian boundary recovery benchmark', 'npm', ['run', 'benchmark:nn-live-jacobian-boundary-recovery'], { timeoutMs: 15 * 60 * 1000 });
    runStep('NN 2D black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-blackbox-relu2d-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN 3D black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-blackbox-relu3d-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN randomized 3D black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-blackbox-relu3d-randomized'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN 4D random-direction line-probe ReLU benchmark', 'npm', ['run', 'benchmark:nn-blackbox-relu-nd-line-probe'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN randomized 4D/5D line-probe ReLU benchmark', 'npm', ['run', 'benchmark:nn-blackbox-relu-nd-line-probe-randomized'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy 4D/5D line-probe ReLU benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu-nd-line-probe'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy adversarial ND line-probe ReLU benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu-nd-adversarial'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN noisy adaptive-adversary ND line-probe ReLU benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu-nd-adaptive-adversary'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN noisy black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy 2D black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu2d-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy 3D black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu3d-decompile'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy randomized 3D black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu3d-randomized'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy 3D cancellation hard-case decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu3d-cancellation'], { timeoutMs: 4 * 60 * 1000 });
    runStep('NN noisy 3D adversarial sweep benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu3d-adversarial'], { timeoutMs: 5 * 60 * 1000 });
    runStep('NN noisy 2D crossing black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu2d-crossing'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy 2D near-parallel black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu2d-near-parallel'], { timeoutMs: 3 * 60 * 1000 });
    runStep('NN noisy 2D near-collision black-box ReLU decompile benchmark', 'npm', ['run', 'benchmark:nn-noisy-blackbox-relu2d-near-collision'], { timeoutMs: 3 * 60 * 1000 });
    runStep('GCL hybrid sparse basis benchmark', 'npm', ['run', 'benchmark:gcl-hybrid:gate'], { timeoutMs: 10 * 60 * 1000 });

    const routingGate = runStep('default compose routing gate', tsNode, [
        path.join(repoRoot, 'scripts', 'verify-compose-routing-gate.ts'),
        '1'
    ], { timeoutMs: 3 * 60 * 1000, captureStdout: true });
    const routingReport = parseJsonPayload<{ pass: boolean; adaptive: { shadowWinners: { total: number }; promotedWinners: { total: number } } }>(routingGate);
    assert(routingReport.pass, 'default compose routing gate did not pass');
    assert(routingReport.adaptive.shadowWinners.total > 0, 'adaptive-duel shadow opt-out produced no shadow evidence');
    assert(routingReport.adaptive.promotedWinners.total === 0, 'adaptive-duel shadow opt-out promoted winners');
    fs.writeFileSync(path.join(workDir, 'default-routing-gate.json'), JSON.stringify(routingReport, null, 2));

    const defaultPromotionEnv: NodeJS.ProcessEnv = {
        GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION: undefined,
        GCL_THRESHOLD_ADAPTIVE_DUEL_COMMIT: undefined
    };
    const calibration = runStep('default promotion calibration', tsNode, [
        path.join(repoRoot, 'scripts', 'compose-routing-calibration.ts'),
        '9',
        '5',
        '5',
        '1337,2026,4242',
        'bridge',
        'adaptive,adaptive-duel',
        'weighted'
    ], { env: defaultPromotionEnv, timeoutMs: 10 * 60 * 1000, captureStdout: true });
    fs.writeFileSync(calibrationPath, calibration);

    const analysisRaw = runStep('promotion calibration analysis', tsNode, [
        path.join(repoRoot, 'scripts', 'analyze-compose-calibration.ts'),
        calibrationPath
    ], { timeoutMs: 2 * 60 * 1000, captureStdout: true });
    fs.writeFileSync(analysisPath, analysisRaw);
    const analysis = parseJsonPayload<AnalysisReport>(analysisRaw);
    assertNoRegression(analysis, 'promotion calibration analysis');

    const qualityRaw = runStep('promotion quality report', tsNode, [
        path.join(repoRoot, 'scripts', 'promotion-quality-report.ts'),
        calibrationPath
    ], { timeoutMs: 2 * 60 * 1000, captureStdout: true });
    fs.writeFileSync(qualityPath, qualityRaw);
    const quality = parseJsonPayload<PromotionQualityReport>(qualityRaw);

    assert(quality.recommendation === 'promote-candidate', `promotion quality report did not recommend promotion: ${quality.recommendation}`);
    assert(quality.status === 'positive-use-case', `promotion quality report did not find a positive use case: ${quality.status}`);
    assert(quality.promotionTelemetry.duelCount > 0, 'promotion run had no adaptive duel decisions');
    assert(quality.promotionTelemetry.shadowTotal > 0, 'promotion run had no shadow winners');
    assert(quality.promotionTelemetry.promotedTotal > 0, 'promotion run had no default promoted winners');
    assert(quality.promotionTelemetry.championBlocks > 0, 'promotion run did not exercise champion-level blocking');
    assertNoRegression({ status: quality.status, deltas: quality.deltas }, 'promotion quality report');

    console.log(JSON.stringify({
        pass: true,
        artifacts: {
            defaultRoutingGate: path.join(workDir, 'default-routing-gate.json'),
            calibration: calibrationPath,
            analysis: analysisPath,
            quality: qualityPath
        },
        promotion: {
            recommendation: quality.recommendation,
            status: quality.status,
            deltas: quality.deltas,
            telemetry: quality.promotionTelemetry
        }
    }, null, 2));
}

try {
    main();
} catch (error) {
    console.error(`\n[release-gate] FAILED: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
}
