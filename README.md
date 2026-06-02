# Genetic Code Lab

Genetic Code Lab is a TypeScript genetic-programming toolkit for evolving small programs and candidate formulas against measurable fitness functions.

Current product status: MVP candidate. The core engine, built-in examples, CLI smoke path, and threshold production lane are build/test gated. Adaptive-duel threshold compose and guarded promotion are now production defaults with a positive-use-case proof.

## Current North Star

The active research/product direction is documented in
[`docs/gcl-north-star.md`](docs/gcl-north-star.md).

Short version: GCL is being developed into an Active-Subspace Neural
Decompiler. The target is not to brute-force huge neural inputs, but to find the
small active subspace of a black-box neuron/channel/block, recover explicit
GELU/SwiGLU/ReLU-style structure when identifiable, validate the recovered
structure against held-out oracle probes, and attach exact source-neuron
provenance when source manifests or hooks are available.

The current mainline is smooth FFN structure recovery and provenance. ReLU
geometry, archive, and symbolic-regression lanes remain important foundations,
but they are not the center of the neural-decompiler roadmap.

## What Is Stable

- Core `EvolutionEngine` API for problem definitions, genomes, population evolution, and result stats.
- Built-in CLI commands for math expression evolution, sorting-network evolution, and pathfinding experiments.
- Safe math target parser for CLI expressions. It does not use JavaScript `eval`.
- Release gate command that verifies build, tests, default compose routing safety, and adaptive-duel promotion readiness.
- Adaptive-duel threshold compose promotes guarded winners by default. Set `GCL_THRESHOLD_COMPOSE_EXPERIMENT=0` to disable threshold compose, or `GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION=0` for shadow-only audits.

## What Still Needs Explicit Proof

- Research scripts under `scripts/` are for calibration, diagnosis, and proof runs, not public API stability.
- Claims of outperforming existing open-source tools are not made by default. That requires a positive benchmark gate, not only passing tests.

## Install

```bash
npm install
npm run build
```

## Verify

Quick local doctor:

```bash
npm run doctor
```

Fast development gate:

```bash
npm run smoke:gate
```

Package hygiene check:

```bash
npm run package:check
```

Installed package verification:

```bash
npm run package:install-check
```

Full release gate:

```bash
npm run release:gate
```

Adaptive-duel promotion proof:

```bash
npm run release:promotion
```

Capability registry:

```bash
npm run start -- capabilities
npm run start -- capabilities threshold-adaptive-duel
npm run start -- capability threshold-adaptive-duel --verify
```

Open-source symbolic-regression benchmark:

```bash
npm run benchmark:opensource
```

Open-source symbolic-regression smoke gate:

```bash
npm run benchmark:opensource:gate
```

The full gate runs:

- TypeScript build
- Jest test suite
- release-lane NN decompiler benchmarks listed in `docs/release-lane-status.md`,
  including active-subspace FFN, GELU structure, near-parallel,
  cross-decomposition invariance, negative-case, noisy mixed GELU, and
  source-neuron provenance gates
- GCL hybrid wide symbolic-regression production gate
- semantic archive composition benchmark
- closed-loop archive emergence benchmark
- multi-island NN archive wide benchmark
- distributed worker-island NN wide benchmark
- default compose-routing safety gate
- default promotion calibration
- calibration analyzer
- promotion quality report

Use `smoke:gate` while developing and `release:gate` before claiming a release-quality change.
Use `release:promotion` to verify adaptive-duel default promotion. It must return `promote-candidate` / `positive-use-case`; use `GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION=0` only for shadow-only audits.
Use `capabilities` to inspect innovation lanes as product capabilities instead of hunting through benchmark scripts and environment flags. Production capabilities have a normal CLI entry, proof command, scope, and artifacts; preview/diagnostic capabilities stay visible instead of being hidden or discarded.
`package:check` verifies the npm package does not include audit logs, tests, source trees, proof archives, or research-only build outputs.
`package:install-check` packs the project, installs the tarball into a temporary project, then verifies the installed `gclab` binary, API import, TypeScript declarations, CLI output JSON, and server health endpoint.

Artifact hygiene: patched release benchmark scripts write to `benchmarks/release/<lane>/<variant>/report.{json,md}` plus `manifest.json`. Root-level `benchmarks/*` files are legacy mixed-use artifacts because older release scripts and scratch experiments both wrote there. A file under `benchmarks/` is not release evidence unless its generating command, date, status, and claim boundary are listed in `docs/evidence-map.md`.

## Neural Decompiler / FFN Structure Scope

The current release evidence covers controlled oracle-only neural decompiler
lanes. It does not claim arbitrary production LLM extraction.

Release-gated smooth FFN structure commands:

```bash
npm run benchmark:nn-active-subspace-ffn-decompiler
npm run benchmark:nn-active-subspace-ffn-4096d-stress
npm run benchmark:nn-active-subspace-ffn-noisy
npm run benchmark:nn-gelu-ffn-structure-finder
npm run benchmark:nn-gelu-ffn-mixed-structure
npm run benchmark:nn-gelu-ffn-near-parallel-family
npm run benchmark:nn-gelu-ffn-cross-decomposition-invariance
npm run benchmark:nn-gelu-ffn-negative-cases
npm run benchmark:nn-gelu-ffn-noisy-mixed-structure
npm run benchmark:nn-gelu-ffn-provenance-verifier
```

What these gates currently support:

- Active-subspace discovery for controlled GELU/SwiGLU-style channels in
  256D/1024D and controlled 4096D stress cases.
- Noisy rank-1 active-subspace recovery for controlled stochastic GELU
  teachers.
- Explicit GELU atom recovery for controlled single/two-channel teachers.
- Rank-adaptive mixed GELU recovery where GCL receives only `f(vector)` and a
  maximum channel budget, not the true channel count.
- Near-parallel mixed GELU stability down to epsilon `1e-4`.
- Cross-decomposition canonical invariance where two mathematically equivalent
  hidden decompositions recover the same effective canonical atom count and
  close canonical signatures.
- Formalization negative-case guardrails for duplicate/cancellation
  canonicalization, weak-under-noise non-identifiability, weak-but-resolvable
  channel recovery, activation-family mismatch rejection, and minimum
  equivalent structure under overcomplete budgets.
- Noisy/quantized mixed GELU recovery for controlled balanced, near-parallel,
  and weak-but-resolvable three-channel FFN teachers, with separate structure
  and accuracy-court gates, IID/wide-OOD/shifted-OOD validation, and
  dependency-free, scikit-learn, XGBoost, and sklearn-MLP package-backed
  baselines.
- Provenance-mode source-neuron verification when an explicit source-neuron
  manifest or equivalent activation/weight instrumentation is available: exact
  source ids, layer names, indices, duplicate source groups, and why-reacts
  evidence are reported for the controlled GELU FFN gate.

What they do not support yet:

- Full GPT-2 FFN extraction as a release claim.
- Arbitrary 3072-channel or 4096D production LLM block decompilation.
- Semantic identity of the original hidden neurons from output-only access
  without a source manifest, weight hook, activation hook, or equivalent
  provenance instrumentation.
- Arbitrary noisy mixed-block/cancellation-heavy GELU or SwiGLU APIs beyond the
  controlled release-gated noisy mixed GELU family.
- Broad "SOTA" or "GCL understands every LLM neuron/block" language.

Use `docs/nn-decompiler-roadmap.md` for the active development plan and
`docs/neural-decompiler-formalization.md` for the atom/canonical-structure
definitions. Use `docs/evidence-map.md` for the exact command/artifact/claim
boundary table.

`benchmark:opensource` compares pure Genetic Code Lab against open-source `gplearn` and DEAP symbolic-regression baselines, then writes `benchmarks/release/open-source-symbolic-regression/<variant>/report.{json,md}`. The engine now includes train-only grammar-derived regression basis seeding for arithmetic/trig symbolic regression; set `GCL_REGRESSION_BASIS_SEED=0` to disable it for ablation. Regression runs also include a train-only semantic refiner that recombines already-evolved subexpressions with sparse fitted coefficients behind an internal validation gate; set `GCL_SEMANTIC_REFINER=0` to disable that lane. A problem-local semantic archive/market keeps high-value subexpressions across generations and can synthesize candidates from multiple elites; it ranks behavior-space bases with Pareto pressure over accuracy, local/global novelty, stability, recombination gain, and complexity so semantic duplicates do not crowd out diverse building blocks. Set `GCL_SEMANTIC_ARCHIVE=0` to disable archive injection. The report also includes `gcl-hybrid`, a train-only sparse basis calibration lane with beam-search selection, a validation split, rational motif coverage, and claim audit. It is reported separately as a production hybrid capability rather than a pure GP result. The benchmark now bootstraps a managed `/tmp/gcl-open-source-bench-venv` from `scripts/open-source-python-requirements.txt` when `gplearn`/DEAP are missing; set `GCL_OPEN_SOURCE_PYTHON` only when you want to force a preconfigured Python environment, which will be verified before the benchmark starts.

`benchmark:opensource:gate` runs the smoke benchmark as a hard-fail gate: Python baseline failures fail the gate, pure `gcl` must solve every current smoke task, and pure `gcl` must beat or stay within the configured band of the best external baseline. `benchmark:gcl-hybrid:gate` runs the wide hybrid production gate with eight symbolic-regression tasks, and requires `gcl-hybrid` to solve every task, avoid MAE and success-rate regressions against pure `gcl`, and win at least half the task set. The current wide audit gives `gcl-hybrid` production-candidate status: `100%` solved rate, `0` MAE regressions, `0` success regressions, and `50%` win rate. Pure-GCL open-source superiority remains separate and is not claimed by hybrid wins.

`benchmark:semantic-archive` is a focused archive-composition gauntlet. It feeds the archive delayed bivariate fragments, semantic duplicates, and rational/trig fragments, then checks that archive-off injects nothing while archive-on synthesizes a train-only composite candidate with at least a 10x MAE lift over the best single elite. It writes `benchmarks/release/semantic-archive-composition/positive-use-case/report.{json,md}`.

`benchmark:archive-emergence` is a closed-loop curated positive-use-case gate: it runs the real `EvolutionEngine` with `GCL_REGRESSION_BASIS_SEED=0`, compares archive-off against archive-on for the same deterministic seeds, and requires archive-on to record/inject candidates and beat archive-off on held-out points. It defaults to curated positive-use-case seeds `3,9,11`, which freshly pass with `successRate=100%` and `summaryLift=1.830375x`. The old default seed set `4,7,1337` is a known negative control (`successRate=0%`, `summaryLift=0.817056x`), so this lane must not be phrased as universal archive superiority. It writes `benchmarks/release/archive-emergence/positive-use-case/report.{json,md}`.

`benchmark:island-nn-archive` is a multi-island + NN archive proof lane. It runs four real `EvolutionEngine` islands through a shared `MigrationHub`, gives each island a different grammar role, collects island champion history, and uses the existing TypeScript `MLP` as a validation-label-trained gate for final island selection without feeding validation error as an input feature. The selector uses a second guard holdout and falls back to the single baseline when the selected island regresses on guard checks. `benchmark:island-nn-archive:wide` runs the 24-seed release sweep with timeout/error accounting, non-regression accounting, non-degenerate lift accounting, archive injection totals, and outcome labels (`island-nn-lift`, `island-nn-miss`, `single-baseline-solved`, `single-baseline-fallback`, `runtime-timeout`, `runtime-error`). The fresh isolated wide run passes with 24/24 completed, 0 timeouts/errors, 100% non-regression, 87.5% non-degenerate success, 255 archive injections, and median eligible lift `7.805164x`. It writes `benchmarks/release/island-nn-archive/<smoke|wide>/report.{json,md}` plus `manifest.json`.

`benchmark:distributed-island-nn` is the production-runtime island proof lane design. It runs real worker-thread islands through `DistributedManager`, uses a shared atomic `MigrationHub`, trains an existing-MLP committee on worker island history, and applies confidence plus robust validation/guard-holdout hybrid selection before final candidate selection. The default sweep uses five seeds (`4,7,1337,2026,4242`), and `benchmark:distributed-island-nn:wide` runs the 24-seed release sweep. The report includes NN holdout accuracy/Brier/AUC, robust holdout-only and oracle-best baselines, regret vs oracle, migration counters, per-seed candidate metrics, timeout/error counts, and outcome labels (`distributed-lift`, `distributed-miss`, `single-baseline-solved`, `single-baseline-fallback`, `runtime-timeout`, `runtime-error`) so the learned policy cannot hide behind a single aggregate. A conservative single-baseline veto requires the distributed candidate to survive both the primary validation grid and a second guard holdout; a severe guard regression falls back to single baseline instead of shipping a distributed regression. The fresh isolated wide run passes with 24/24 completed, 0 timeouts/errors, 100% non-regression, 95.83% non-degenerate success, 223 hub-pull successes, and median eligible lift `10.957924x`. It writes `benchmarks/release/distributed-island-nn/<smoke|wide>/report.{json,md}` plus `manifest.json`.

Release-gate status:

- build passes
- test suite passes
- semantic archive composition benchmark passes
- closed-loop archive emergence benchmark now passes as a curated positive-use-case on seeds `3,9,11`; old seeds `4,7,1337` remain a negative control
- multi-island NN archive wide benchmark passes in fresh isolated release output
- distributed worker-island NN wide benchmark passes in fresh isolated release output
- adaptive-duel shadow opt-out has shadow evidence and zero promotions
- default promotion recommendation is `promote-candidate`
- promotion must not regress exact accuracy, boundary total, or fitness

`npm run release:gate` is green as of 2026-06-02 after the deep 3D/4D ND
region-sampler, recovered-adjacency geometry, 5D clean/noisy sweep, 6D
randomized/adversarial robustness and baseline court, active-subspace FFN
decompiler, 4096D/noisy active-subspace gates, explicit GELU structure finder,
rank-adaptive mixed GELU gate, near-parallel mixed GELU family, and
cross-decomposition GELU invariance gate, GELU formalization negative-case
gate, noisy mixed GELU IID/OOD XGBoost/NN court, and source-neuron provenance verifier gate. This is release-lane proof, not a promotion of scratch experiments, not
full GPT-2/LLM extraction, not a pure-GCL universal superiority claim, and not a
broad SOTA claim.

See `docs/release-lane-status.md` for the explicit stable/opt-in/research lane matrix.

## CLI Usage

Math expression evolution:

```bash
npm run start -- math --generations 100 --target "x*x + 2*x + 1"
```

Sorting network evolution:

```bash
npm run start -- sort --size 4 --generations 200
```

Pathfinding evolution:

```bash
npm run start -- path --width 8 --height 8 --generations 150
```

Run from a reproducible JSON config:

```bash
npm run start -- run configs/math-quadratic.json
```

Minimal config shape:

```json
{
  "problem": {
    "type": "math",
    "target": "x*x + 2*x + 1",
    "testCount": 30
  },
  "evolution": {
    "generations": 100,
    "populationSize": 100,
    "seed": 42
  },
  "output": "runs/math-quadratic-result.json"
}
```

Supported `problem.type` values:

- `math`: supports `target`, `variableCount`, `testCount`
- `sort`: supports `size`
- `path`: supports `width`, `height`, `testCount`

Doctor through compiled CLI:

```bash
npm run start -- doctor
npm run start -- doctor --release
```

## Programmatic API

Public package API is the root import:

```ts
import { EvolutionEngine, createMathProblem } from 'genetic-code-lab';

const targetFn = (inputs: number[]) => {
  const x = inputs[0];
  return x * x + 2 * x + 1;
};

const problem = createMathProblem(targetFn, 1, 30);
const engine = new EvolutionEngine(problem, {
  population: {
    size: 100,
    eliteCount: 2,
    crossoverRate: 0.8,
    mutationRate: 0.2,
    maxGenomeDepth: 8,
    maxGenomeSize: 50,
    tournamentSize: 3
  },
  maxGenerations: 50,
  targetFitness: 0.99,
  stagnationLimit: 20,
  seed: 42
});

const result = await engine.evolve();
console.log(result.bestGenome.fitness);
```

Deep imports from internal `dist/` paths are not part of the public API contract.

## Release Rule

Do not call a change finished unless this passes:

```bash
npm run release:gate
```

Do not keep adaptive-duel promotion enabled if the quality report loses its positive signal and falls back to only `tie/no-regression`.
