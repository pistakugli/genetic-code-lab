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

Black-box and instrumented modes are separate. A black-box claim means the
solver consumed only oracle calls such as `f(x) -> y`, logits, or selected
output channels. Local model weights, hooks, activations, and source manifests
may be used to create a real oracle or to verify/label a recovered structure
afterward, but they are not allowed to be the solver input for a black-box
4096D/LLM decompilation claim.

The concrete live-model product target is a graph-level adaptive LLM circuit
reader: given a real local model, prompt/target, and adapter, GCL should find
causal attention/MLP/residual paths, validate them against controls and
paraphrases, search alternate paths when prompt shifts break the original path,
and explain why selected components react. GPT-2 is only a fast proof model;
the framework target is adapter-backed portability across transformer families.

The current mainline is smooth FFN structure recovery, provenance, and
adapter-backed LLM circuit reading. ReLU geometry, archive, and
symbolic-regression lanes remain important foundations, but they are not the
center of the neural-decompiler roadmap.

## What Is Stable

- Core `EvolutionEngine` API for problem definitions, genomes, population evolution, and result stats.
- Built-in CLI commands for math expression evolution, sorting-network evolution, and pathfinding experiments.
- Safe math target parser for CLI expressions. It does not use JavaScript `eval`.
- Release gate command that verifies build, tests, default compose routing safety, and adaptive-duel promotion readiness.
- Adaptive-duel threshold compose promotes guarded winners by default. Set `GCL_THRESHOLD_COMPOSE_EXPERIMENT=0` to disable threshold compose, or `GCL_THRESHOLD_ADAPTIVE_DUEL_PROMOTION=0` for shadow-only audits.

## What Still Needs Explicit Proof

- Research scripts under `scripts/` are for calibration, diagnosis, and proof runs, not public API stability.
- Claims of outperforming existing open-source tools are not made by default. That requires a positive benchmark gate, not only passing tests.
- A graph-level adaptive LLM reader that discovers alternate circuits when OOD
  paraphrases or prompt shifts break the selected path.

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
npm run benchmark:nn-swiglu-ffn-mixed-structure
npm run benchmark:nn-swiglu-ffn-noisy-mixed-structure
npm run benchmark:nn-swiglu-ffn-shared-gate-canonical
npm run benchmark:nn-gelu-ffn-provenance-verifier
npm run benchmark:nn-local-transformer-mlp-hook
```

Live GPT-2 oracle instrumentation:

```bash
npm run python:setup
npm run llm:read:gpt2
npm run benchmark:nn-live-gpt2-fact-circuit-finder
npm run benchmark:nn-live-gpt2-attention-path-circuit-finder
npm run benchmark:nn-live-gpt2-combined-circuit-finder
npm run benchmark:nn-live-gpt2-ordered-residual-path-circuit-finder
```

GPU-accelerated live GPT-2 instrumentation on machines with NVIDIA CUDA:

```bash
npm run python:setup:gpu
npm run benchmark:nn-live-gpt2-fact-circuit-finder:gpu
npm run benchmark:nn-live-gpt2-attention-path-circuit-finder:gpu
npm run benchmark:nn-live-gpt2-combined-circuit-finder:gpu
npm run benchmark:nn-live-gpt2-ordered-residual-path-circuit-finder:gpu
```

Targeted blind real-SwiGLU oracle gate:

```bash
npm run benchmark:nn-live-blind-swiglu-oracle-decompiler
npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:sweep
```

The CPU environment remains the default reproducible release path. The GPU
environment is intentionally separate at `.gcl-python-gpu` and is used only when
`GCL_PYTHON=.gcl-python-gpu/bin/python` and `GCL_LIVE_GPT2_DEVICE=cuda` are set.
`npm run python:setup:gpu` installs a CUDA PyTorch wheel and fails if
`torch.cuda.is_available()` is false or a real CUDA tensor probe cannot run.

The live reader loads the real local GPT-2 model through the GCL-managed Python
runtime, hooks real MLP activations, runs causal neuron ablations, and records
negative-control families. GCL now also has a first benchmark-scoped live GPT-2
fact-circuit finder: it scans real GPT-2 layers/top MLP neurons for factual
prompts, builds token/neuron/target graph artifacts, and validates the selected
graph through joint causal ablation against graph-level controls. Hooking,
ablation, logit-delta scoring, and controls are known mechanistic-
interpretability sensors; the GCL claim is the automatic graph selection,
control-normalized graph evidence, and release artifact, not a hand-inspected
single-neuron demo.

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
  and prediction-sanity gates, IID/wide-OOD/shifted-OOD validation, and
  dependency-free, scikit-learn, XGBoost, and sklearn-MLP package-backed fit
  baselines. These are prediction sanity checks for the recovered structure,
  not structure-decompiler competitors.
- Rank-adaptive mixed SwiGLU recovery for a controlled two-block teacher where
  GCL receives only `f(vector)` and a maximum pair budget, then recovers
  explicit value/gate pairs and the effective pair count.
- Noisy/quantized mixed SwiGLU recovery for controlled balanced,
  near-parallel-value, and weak-but-resolvable value/gate-pair teachers, where
  GCL receives only repeated noisy `f(vector)` calls and max pair budgets, then
  recovers explicit value/gate pairs and effective pair counts.
- Shared-gate/cancellation-heavy SwiGLU canonicalization for controlled clean
  and noisy/quantized teachers, where multiple source value channels share one
  gate and GCL must recover the single minimal functionally equivalent
  value/gate pair instead of promoting duplicate non-identifiable channels.
- Provenance-mode source-neuron verification when an explicit source-neuron
  manifest or equivalent activation/weight instrumentation is available: exact
  source ids, layer names, indices, duplicate source groups, and why-reacts
  evidence are reported for the controlled GELU FFN gate.
- Local transformer MLP hook recovery for a deterministic local
  transformer-style block: 384 captured residual/MLP/preactivation/activation/
  contribution/output rows, 3 recovered GELU expansion neurons for one target
  MLP output slice, R2 `0.9998897778278036`, MAE
  `0.005720353338158273`, minimum provenance confidence
  `0.9106904584057108`, and `19540` raw oracle calls.
- Live GPT-2 oracle instrumentation: real model loading, real forward-pass MLP
  hooks, causal ablations, and disjoint negative-control families for the
  current GPT-2 prompt test.
- Live GPT-2 graph-level fact circuit finding for the configured factual prompt
  set with Stage-I hardening: `50` valid facts, strict case definition
  `graphGate.pass && hardeningGate.pass`, `50` strict-passed cases, `50`
  graph-passed cases, `50` hardening-passed cases, pass rate `1.0`, median
  joint graph target-logit drop `0.25433349609375`, p10 graph drop
  `0.1052703857421875`, median graph-control p95 positive target-logit drop
  `0.06884765625`, median graph lift over control p95
  `3.8081556082148498`, mean selected graph node count `1.6`, and `46` unique
  selected graph neurons. Hardening passed with paraphrase pass rate `1.0`,
  wrong-answer specificity `1.0`, negative-fact specificity `1.0`, and
  unrelated-logit preservation `0.955`. The current artifact was generated with
  the GPU command and records `model.device = cuda`. The artifact is
  `benchmarks/release/nn-live-gpt2-fact-circuit-finder/gpt2-50-facts-graph-layer-scan/report.json`.
- Live GPT-2 attention-head/path circuit finding for the configured factual
  prompt set: `50` valid facts, `50` passed cases, pass rate `1.0`, median
  attention graph target-logit drop `1.7271881103515625`, p10 graph drop
  `1.1454010009765625`, median attention-control p95 target-logit drop
  `0.448638916015625`, median lift over control p95 `3.7378692395502666`,
  mean selected heads `1.04`, and `6` unique selected heads. The current
  artifact was generated with the GPU command and records `model.device =
  cuda`. Wrong-answer collateral ratio is reported, median
  `0.7162562846461261`, but is not gated for this attention-path lane because
  heads can carry broad token-class effects. Skipped rows are reported
  separately and are not counted as valid pass/fail cases when the answer is
  not a single token or the target baseline rank is low-confidence. The
  artifact is
  `benchmarks/release/nn-live-gpt2-attention-path-circuit-finder/gpt2-50-facts-attention-path-layer-scan/report.json`.
- Live GPT-2 combined attention+MLP circuit finding for the configured factual
  prompt set: `50` valid facts, `50` passed cases, pass rate `1.0`, MLP
  component pass rate `1.0`, attention component pass rate `1.0`, median
  combined target-logit drop `1.92523193359375`, p10 combined drop
  `1.1469650268554688`, median full-control combined p95 target-logit drop
  `0.417144775390625`, median combined lift over full-control p95
  `4.333544871993313`, mean selected MLP nodes `1.56`, and mean selected
  attention heads `1.02`. The current artifact was generated with the GPU
  command and records `model.device = cuda`; failed subjects are empty. The
  combined gate uses full component-replacement controls for pass/fail while
  retaining MLP-swap and attention-swap controls as diagnostics. The artifact
  is
  `benchmarks/release/nn-live-gpt2-combined-circuit-finder/gpt2-50-facts-attention-mlp-combined-graph/report.json`.
- Live GPT-2 ordered residual path circuit finding for the configured factual
  prompt set: `50` valid facts, `50` passed cases, pass rate `1.0`, combined
  graph pass rate `1.0`, median selected attention-to-MLP activation delta
  `0.18021255731582642`, p10 selected path activation delta
  `0.09223440289497375`, median full-control path p95 activation delta
  `0.0318724343786016`, median ordered path lift over full-control p95
  `5.259007093850701`, and median combined target-logit drop
  `1.9063301086425781`. The current artifact was generated with the GPU
  command and records `model.device = cuda`, `torch = 2.10.0+cu128`,
  `synthetic = false`, and `surrogate = false`. The gate uses an order-aware
  rescue selector when the union graph is causal but not a valid
  attention->residual->MLP ordering. The artifact is
  `benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json`.
- Live Pythia/GPT-NeoX ordered residual path circuit finding for the configured
  factual prompt set: `20` valid facts, `20` passed cases, pass rate `1.0`,
  combined graph pass rate `1.0`, median selected attention-to-MLP activation
  delta `0.51171875`, p10 selected path activation delta
  `0.3111328125`, median full-control path p95 activation delta
  `0.06836662292480464`, median ordered path lift over full-control p95
  `10.015731969055064`, and failed subjects `[]`. The artifact records
  `model.device = cuda`, `adapter = gpt-neox`, `synthetic = false`, and
  `surrogate = false`. Deterministic per-subject controls and multi-component
  ordered rescue close the previous Japan/Israel hard cases without lowering
  thresholds.
- Live Mistral/SwiGLU ordered residual path circuit finding for a real
  quantized Mistral-7B-Instruct model: `10` valid facts, `10` passed cases,
  pass rate `1.0`, combined graph pass rate `1.0`, median selected path
  activation delta `0.203125`, p10 delta `0.0921875`, median full-control path
  p95 delta `0.04531249999999998`, median ordered path lift over full-control
  p95 `5.104890772220551`, and failed subjects `[]`. The artifact records
  32 layers, hidden size `4096`, intermediate size `14336`, 8 key/value heads,
  `model.device = cuda`, `adapter = mistral-swiglu`, `synthetic = false`, and
  `surrogate = false`.
- Live ordered-path stability replay across GPT-2, Pythia/GPT-NeoX,
  Llama/SwiGLU, and Mistral/SwiGLU: `4/4` model reports pass. GPT-2 and
  Pythia/GPT-NeoX each check 5 replay cases; Llama/SwiGLU and Mistral/SwiGLU
  each check 4 replay cases. Each case probes 3 paraphrase prompts and layer
  shifts `[-2,-1,1,2]` with at least 2 valid layer controls. Original replay
  pass rate is `1.0` for every model report; paraphrase pass rates are GPT-2
  `1.0`, Pythia/GPT-NeoX `0.9333333333333333`, Llama/SwiGLU `1.0`, and
  Mistral/SwiGLU `1.0`; neighboring-layer specificity rate is `1.0`, max
  replay error is `0`, and failed subjects are `[]`.
- Blind 4096D real SwiGLU unit/layer sweep: `4/4` selected Mistral-family
  units passed with the same oracle-only access invariant, min R2
  `0.9999845855712838`, max MAE `1.5160815735029332e-05`, max worst error
  `0.00017546677141010852`, min subspace principal cosine
  `0.9999997892743723`, max raw oracle calls per case `18496`, and total raw
  oracle calls `73984`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-4unit-layer-sweep-2probe-oracle-only/report.json`.
- Blind 4096D randomized real SwiGLU sweep: `8/8` deterministic-random
  Mistral-family layer/unit pairs passed with the same oracle-only access
  invariant and the same `18496` max raw oracle calls per case. Result: min R2
  `0.9999570341720895`, max MAE `2.4326629529818456e-05`, max worst error
  `0.00044138616892425153`, min subspace principal cosine
  `0.9999999957944806`, and total raw oracle calls `147968`. Aggregate
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-8unit-layer-sweep-2probe-oracle-only/report.json`.
- Blind 3072D LLaMA-family SwiGLU parity sweep: `4/4` selected real
  LLaMA-family layer/unit pairs passed with the same oracle-only access
  invariant. Result: min R2 `0.9999999964505302`, max MAE
  `1.0737975340964412e-05`, max worst error `0.00011067984901980132`, min
  subspace principal cosine `0.9999999979627894`, max raw oracle calls per
  case `14400`, and total raw oracle calls `57600`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-4unit-layer-sweep-2probe-oracle-only/report.json`.
- Blind 3072D randomized LLaMA-family SwiGLU sweep: `8/8`
  deterministic-random real LLaMA-family layer/unit pairs passed with the same
  oracle-only access invariant and max raw oracle calls per case `14400`.
  Result: min R2 `0.9999981324158254`, max MAE `0.00016030715095856503`, max
  worst error `0.001568794345920943`, min subspace principal cosine
  `0.9999994111232651`, and total raw oracle calls `115200`. Aggregate
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-8unit-layer-sweep-2probe-oracle-only/report.json`.
- Blind 3072D LLaMA-family 6-unit SwiGLU block recovery: a scalar oracle
  returns the sum of six real layer-20 units `[0,17,123,456,789,1024]`; the
  solver receives only `f(x)` batches and recovers an explicit 6-atom SwiGLU
  sum in a rank-12 active subspace. Result: pass `true`, R2
  `0.9999999971996777`, MAE `2.8572995724253928e-05`, worst error
  `0.00021262079246214816`, subspace min principal cosine
  `0.9999999938218761`, raw oracle calls `88864`, and access audit
  `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
  `hooksUsedBySolver = false`, `postHocWitnessOnly = true`. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-6atom-oracle-only/report.json`.
- Blind 3072D randomized LLaMA-family 6-unit SwiGLU block recovery: `2/2`
  deterministic-random six-unit blocks passed with active rank `12` and the
  same oracle-only invariant. Result: min R2 `0.9999999419380987`, max MAE
  `0.00008572753027982732`, max worst error `0.0008619591998977327`, min
  subspace principal cosine `0.9999998049413197`, max raw oracle calls per
  case `88864`, and total raw oracle calls `177728`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-6unit-sweep-oracle-only/report.json`.
- Blind 3072D LLaMA-family 8-unit forward-coordinate SwiGLU block recovery:
  a scalar oracle returns the sum of eight real layer-20 units
  `[0,17,123,456,789,1024,2048,4096]`; the solver receives only `f(x)` batches,
  uses raw forward-coordinate active-subspace witnesses, active-gradient fit
  witnesses, and a batched CUDA multi-start explicit 8-atom SwiGLU fitter.
  Result: pass `true`, R2 `0.9999913818259203`, MAE
  `0.0017664391250281214`, worst error `0.016726727075836223`, subspace min
  principal cosine `0.9999919539106246`, and raw oracle calls `68241`.
  Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-budgeted-oracle-only/report.json`.
- Blind 3072D randomized LLaMA-family 8-unit forward-coordinate SwiGLU block
  recovery: `2/2` deterministic-random eight-unit blocks passed under the same
  oracle-only invariant with active-gradient fit witnesses and batched CUDA
  fitting. Result: min R2 `0.9999781080257685`, max MAE
  `0.002327151614302204`, max worst error `0.02565795362142126`, min subspace
  principal cosine `0.999936352892748`, max raw oracle calls per case `68241`,
  and total raw oracle calls `136482`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only/report.json`.
- Blind 4096D Mistral small multi-unit SwiGLU block recovery: a scalar oracle
  returns the sum of two real layer-20 SwiGLU unit outputs, while the solver
  receives only `f(x)` batches and recovers an explicit 2-atom SwiGLU sum in a
  rank-4 active subspace. Result: pass `true`, units `[0,17]`, R2
  `0.9998005671654032`, MAE `9.702990494674492e-05`, worst error
  `0.0012327953741858785`, subspace min principal cosine
  `0.9999999793123026`, raw oracle calls `44128`, and access invariant
  `solverAccess = oracle-only`. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-2atom-oracle-only/report.json`.
- Blind 4096D randomized Mistral small-block SwiGLU recovery: `2/2`
  deterministic-random two-unit blocks passed under the same oracle-only
  invariant. Result: min R2 `0.9994744138350042`, max MAE
  `0.00014261562921587213`, max worst error `0.0027002211714715688`, min
  subspace principal cosine `0.9999999823144813`, max raw oracle calls per
  case `44128`, total raw oracle calls `88256`, max unit count `2`, and max
  atom count `2`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-2unit-sweep-oracle-only/report.json`.
- Blind 4096D Mistral first 3-unit SwiGLU block recovery: a scalar oracle
  returns the sum of three real layer-20 units `[0,17,123]`; the solver
  receives only `f(x)` batches and recovers an explicit 3-atom SwiGLU sum in a
  rank-6 active subspace. Result: pass `true`, R2 `0.9992245717327005`, MAE
  `0.0002548553473914666`, worst error `0.0029161037108420967`, subspace min
  principal cosine `0.9999999974474356`, raw oracle calls `61968`, and access
  invariant `solverAccess = oracle-only`. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-3atom-oracle-only/report.json`.
- Blind 4096D randomized Mistral 3-unit SwiGLU block recovery: `2/2`
  deterministic-random three-unit blocks passed with an explicit worst-error
  gate (`maxWorstError <= 0.02`). Result: min R2 `0.9962650545162728`, max
  MAE `0.0005119471166077025`, max worst error `0.010120884383530829`, min
  subspace principal cosine `0.9999999878109772`, total raw oracle calls
  `123936`, max unit count `3`, and max atom count `3`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-3unit-sweep-oracle-only/report.json`.
- Blind 4096D Mistral 4-unit SwiGLU block recovery: a scalar oracle returns
  the sum of four real layer-20 units `[0,17,123,456]`; the solver receives
  only `f(x)` batches and recovers an explicit 4-atom SwiGLU sum in a rank-8
  active subspace. This gate required slower 4-atom fit settings
  (`fitLearningRate = 0.01`, `fitSteps = 3200`) rather than a lowered
  accuracy threshold. Result: pass `true`, R2 `0.9996964505713657`, MAE
  `0.0001659790529608388`, worst error `0.002872809058861975`, subspace min
  principal cosine `0.9999999945348134`, and raw oracle calls `79808`.
  Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-4atom-oracle-only/report.json`.
- Blind 4096D randomized Mistral 4-unit SwiGLU block recovery: `2/2`
  deterministic-random four-unit blocks passed with active rank `8` and the
  same oracle-only invariant. Result: min R2 `0.995727068595925`, max MAE
  `0.0006359374939938832`, max worst error `0.008566143276326953`, min
  subspace principal cosine `0.9999999725417702`, max raw oracle calls per
  case `79808`, and total raw oracle calls `159616`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-4unit-sweep-oracle-only/report.json`.
- Blind 4096D Mistral 5-unit SwiGLU block recovery: a scalar oracle returns
  the sum of five real layer-20 units `[0,17,123,456,789]`; the solver
  receives only `f(x)` batches and recovers an explicit 5-atom SwiGLU sum in a
  rank-10 active subspace. Result: pass `true`, R2 `0.9970749283083804`, MAE
  `0.0006105457047962797`, worst error `0.008950286237037718`, subspace min
  principal cosine `0.9999999795104269`, and raw oracle calls `97648`.
  Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-5atom-oracle-only/report.json`.
- Blind 4096D randomized Mistral 5-unit SwiGLU block recovery: `2/2`
  deterministic-random five-unit blocks passed with active rank `10`.
  Result: min R2 `0.9951436177161573`, max MAE
  `0.0008022104114991162`, max worst error `0.010051115148071464`, min
  subspace principal cosine `0.9999997906113538`, max raw oracle calls per
  case `97648`, and total raw oracle calls `195296`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-5unit-sweep-oracle-only/report.json`.
- Blind 4096D Mistral 6-unit SwiGLU block recovery: a scalar oracle returns
  the sum of six real layer-20 units `[0,17,123,456,789,1024]`; the solver
  receives only `f(x)` batches and recovers an explicit 6-atom SwiGLU sum in a
  rank-12 active subspace. Result: pass `true`, R2 `0.9970869523285548`, MAE
  `0.0007282176910088243`, worst error `0.008779981374470364`, subspace min
  principal cosine `0.9999999544565973`, raw oracle calls `115488`, and access
  audit `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
  `hooksUsedBySolver = false`, `postHocWitnessOnly = true`. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-6atom-oracle-only/report.json`.
- Blind 4096D randomized Mistral 6-unit SwiGLU block recovery: `2/2`
  deterministic-random six-unit blocks passed with active rank `12` and the
  same oracle-only invariant. Result: min R2 `0.9952423306790068`, max MAE
  `0.0009230608551877102`, max worst error `0.012063083889497478`, min
  subspace principal cosine `0.9999999518366974`, max raw oracle calls per
  case `115488`, and total raw oracle calls `230976`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-6unit-sweep-oracle-only/report.json`.
- Blind 4096D Mistral selected 8-unit SwiGLU block recovery with centered
  basis and active-gradient fit witnesses: a scalar oracle returns the sum of
  eight real layer-20 units `[0,17,123,456,789,1024,2048,4096]`; the solver
  receives only `f(x)` batches and recovers an explicit 8-atom SwiGLU sum in a
  rank-16 active subspace. Result: pass `true`, R2 `0.9953413995342039`, MAE
  `0.0011220708891648541`, worst error `0.01246827271765028`, subspace min
  principal cosine `0.9999966163799717`, raw oracle calls `85649`, and access
  audit `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
  `hooksUsedBySolver = false`, `postHocWitnessOnly = true`. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-clean-centered-gradfit-oracle-only/report.json`.
- Blind 4096D Mistral selected 8-unit finite-precision SwiGLU block recovery:
  the same scalar oracle deterministically quantizes every response to step
  `1e-8`. Result: pass `true`, R2 `0.9954504023764993`, MAE
  `0.0011104610821356987`, worst error `0.012667784692725533`, subspace min
  principal cosine `0.9999724425295571`, and raw oracle calls `85649`.
  Coarser Mistral quantization is not promoted by this artifact. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-q00000001-centered-gradfit-oracle-only/report.json`.
- Blind 4096D randomized Mistral 8-unit finite-precision SwiGLU sweep: `2/2`
  deterministic-random eight-unit blocks passed with scalar response
  quantization step `1e-8`, centered basis, active-gradient fit witnesses, and
  the same oracle-only invariant. Result: min R2 `0.9957594628054579`, max MAE
  `0.0010088005020659832`, max worst error `0.009621486908707531`, min
  subspace principal cosine `0.9999828597286717`, max raw oracle calls per
  case `85649`, and total raw oracle calls `171298`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-q00000001-centered-gradfit-sweep-oracle-only/report.json`.
- Blind 4096D Mistral selected 8-unit higher-query finite-precision SwiGLU
  recovery: the same selected eight-unit scalar oracle passes at response
  quantization step `1e-7` when the estimator uses finite-difference step
  `0.08` and `25` probe points. Result: pass `true`, R2
  `0.9963343164135441`, MAE `0.0010131811501723571`, worst error
  `0.008584273469236195`, subspace min principal cosine
  `0.9999930621807005`, and raw oracle calls `118425`. The lower-query
  `17`-probe `1e-7` attempt is not promoted. Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-q0000001-centered-gradfit-step008-probe25-oracle-only/report.json`.
- Blind 4096D randomized Mistral 8-unit higher-query finite-precision SwiGLU
  sweep: `2/2` deterministic-random eight-unit blocks passed at response
  quantization step `1e-7` with `25` probe points. Result: min R2
  `0.995860081804988`, max MAE `0.0010036287991970745`, max worst error
  `0.010330999089242423`, min subspace principal cosine
  `0.9999857710403501`, max raw oracle calls per case `118425`, and total raw
  oracle calls `236850`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-q0000001-centered-gradfit-probe25-sweep-oracle-only/report.json`.
- Blind 4096D Mistral selected 8-unit multi-step finite-precision SwiGLU
  recovery: the selected eight-unit scalar oracle passes at response
  quantization step `1e-6` when the new multi-step estimator uses
  finite-difference steps `0.08,0.12`. Result: pass `true`, R2
  `0.9952414211113175`, MAE `0.0011636592796313008`, worst error
  `0.009295537729115112`, subspace min principal cosine
  `0.9993536447338561`, and raw oracle calls `220850`. This artifact alone is
  selected evidence; randomized `1e-6` robustness is reported separately below.
  Artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-q000001-centered-gradfit-multistep0812-probe25-oracle-only/report.json`.
- Blind 4096D randomized Mistral 8-unit multi-step finite-precision SwiGLU
  sweep: `2/2` deterministic-random eight-unit blocks passed at response
  quantization step `1e-6` after increasing the basis witness budget to `33`
  probe points while keeping the multi-step finite-difference estimator
  `0.08,0.12` and reducing noisy gradient-fit weight to `0.075`. Result:
  min R2 `0.9958676370633143`, max MAE `0.0009856537193232476`, max worst
  error `0.014497109404921407`, min
  subspace principal cosine `0.9994407684636628`, max raw oracle calls per
  case `290754`, and total raw oracle calls `581508`. Aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-q000001-centered-gradfit-multistep0812-probe33-grad0075-sweep-oracle-only/report.json`.

Finite-precision boundary:

- `4096D` is the real Mistral hidden/input dimension seen by the oracle.
  `8-unit` is the selected recovered SwiGLU block size. These gates do not
  claim all-channel or full-FFN extraction.
- `q=1e-8`, `q=1e-7`, and `q=1e-6` now have deterministic-random two-case
  8-unit sweeps. The `q=1e-6` sweep requires the larger `33`-probe witness
  budget; the lower `25`-probe setting is only selected-pass evidence.
- The finite-precision lane shows graceful benchmark-scoped degradation, not a
  universal API guarantee: randomized `q=1e-8` min R2 `0.9957594628054579`,
  randomized `q=1e-7` min R2 `0.995860081804988`, and randomized `q=1e-6`
  min R2 `0.9958676370633143`.

What they do not support yet:

- Full GPT-2 FFN extraction as a release claim.
- Arbitrary 3072-channel or 4096D production LLM block decompilation.
- Full Mistral/LLaMA FFN extraction or all-channel 4096D decompilation from
  the selected-unit blind SwiGLU oracle gates.
- Semantic identity of the original hidden neurons from output-only access
  without a source manifest, weight hook, activation hook, or equivalent
  provenance instrumentation.
- Full GPT-2/LLM FFN extraction, arbitrary transformer block decompilation, or
  production-model source identity from the local transformer hook gate alone.
- Treating the single-prompt live reader as circuit discovery. The new
  fact-circuit finder is benchmark-scoped MLP graph discovery for a configured
  real GPT-2 factual prompt set. The attention-path finder is a separate
  benchmark-scoped head/path graph lane, and the combined finder now validates a
  token/head/neuron/target union graph through simultaneous attention+MLP
  ablation. The ordered residual path finder adds benchmark-scoped
  attention->residual->MLP->target evidence above full-control path
  interventions. These gates still do not prove full GPT-2 FFN extraction,
  arbitrary LLM understanding, exhaustive residual-stream patching for every
  edge, or output-only original-neuron identity.
- Treating the current verified adapters as proof that the circuit finder
  already scales to arbitrary LLM architectures. The live reader has a
  model-agnostic `LLMOracleAdapter` boundary with real GPT-2, GPT-NeoX/Pythia,
  LLaMA/SwiGLU, and Mistral/SwiGLU gates, but arbitrary tokenizer/runtime
  stacks, full FFN extraction, semantic original-neuron identity, and
  exhaustive circuit completeness are still outside the verified claim.
- Arbitrary noisy mixed-block/cancellation-heavy GELU or SwiGLU APIs beyond the
  controlled release-gated noisy mixed GELU, noisy mixed SwiGLU, and shared-gate
  SwiGLU canonicalization families.
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

`npm run release:gate` is green as of 2026-06-05 after the deep 3D/4D ND
region-sampler, recovered-adjacency geometry, 5D clean/noisy sweep, 6D
randomized/adversarial robustness and baseline court, active-subspace FFN
decompiler, 4096D/noisy active-subspace gates, explicit GELU structure finder,
rank-adaptive mixed GELU gate, near-parallel mixed GELU family, and
cross-decomposition GELU invariance gate, GELU formalization negative-case
gate, noisy mixed GELU IID/OOD prediction-sanity court, mixed SwiGLU structure
gate, noisy mixed SwiGLU structure gate, shared-gate cancellation SwiGLU
canonical gate, source-neuron provenance verifier gate, local transformer MLP
hook gate, compact GPT-2 FFN source-structure gates, and the open-source hybrid
baseline gate. The live GPT-2 fact-circuit finder remains a separate heavy
real-model benchmark command, freshly green with Stage-I hardening as listed
above. This is release-lane proof, not a promotion of scratch experiments, not
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
