# NN Decompiler Roadmap

## What this is for

A trained network is a black box: you can run it and see what comes out, but you
cannot say what any part of it is doing. This project exists to remove that. The
goal is to read a neural network the way a decompiler reads a binary — to take a
piece of an LLM you can only query, and come back with the units it is made of,
what each one responds to, and a reconstruction faithful enough that being wrong
about it would show.

Read completely, an LLM's network stops being a black box. That is the target
the work is aimed at, and it is not reached yet. What has been reached is
recorded in Direction 1b with its scope stated: real weights of LLaMA-3.2-3B
layer 15, blocks up to 1024 of the layer's 8192 units, solver holding oracle
access only, every unit recovered to a cosine above 0.99 against the truth it
never saw. The scope matters as much as the number — the oracle is configured
with the block's own units, so those results describe a block the harness
isolates, not a layer. The run that removes that qualifier is the one over all
8192 units, where naming every unit says nothing that naming none would not.

`docs/gcl-north-star.md` holds the full charter, including what may not be
claimed. Two lines from it govern everything below: a black-box claim requires a
solver that reads no weights, no hooks, and no original neuron ids; and no
result here removes black-box behaviour universally, only for what was measured.

The rest of this roadmap is the discipline that keeps the distance between the
target and the measurement visible. A lane becomes a release claim only when it
has a command, artifact, gate, and clear boundary.

## Focus Reset

Read `docs/gcl-north-star.md` before adding new neural-decompiler work. The
current mainline is Active-Subspace Neural Decompilation:

```text
high-dimensional black-box channel/block
-> active subspace discovery
-> explicit atom/canonical structure recovery
-> provenance mapping when source manifests or hooks exist
-> held-out fidelity, query budget, baselines, and negative cases
```

Do not let new 7D/8D ReLU probes, GPT-2 fixture experiments, or scratch
surrogate results replace the mainline. They are useful only when they improve
the active-subspace structure finder, provenance verifier, baseline court, or
failure-generation machinery.

## Engineering Rule

Problems must not be silenced or bypassed. If a benchmark reveals a failure,
the failure becomes an explicit test or benchmark variant, and the implementation
must be fixed or the limitation must remain visible as a failing/non-promoted
gate. Do not make the task easier just to preserve a pass.

## Immediate Pivot: Model-Agnostic LLM Adapter Circuit Framework

The live GPT-2 reader is the first fast proof gate, not the destination.
Hooking activations, ablating neurons, measuring logit deltas, and comparing
negative controls are known mechanistic-interpretability techniques. GCL must
use them as sensors for automatic graph discovery, not as the final claim. The
destination is a typed model-agnostic LLM adapter framework that can run the
same circuit-search protocol across compatible transformer families.

The first graph-level gate now exists:
`npm run benchmark:nn-live-gpt2-fact-circuit-finder`. It loads the real GPT-2
runtime, scans configured factual prompts across selected layers/top MLP
neurons, emits token/neuron/target graph artifacts, and validates selected
graphs through joint causal ablation above graph-level controls.

Current gate artifact:

```text
benchmarks/release/nn-live-gpt2-fact-circuit-finder/gpt2-50-facts-graph-layer-scan/report.json
```

Current result:

- pass `true`;
- strict case definition `graphGate.pass && hardeningGate.pass`;
- 50 valid facts, 50 strict-passed cases, 50 graph-passed cases, 50
  hardening-passed cases, pass rate `1.0`;
- median joint graph target-logit drop `0.25433349609375`;
- p10 joint graph target-logit drop `0.1052703857421875`;
- median graph-control p95 positive target-logit drop `0.06884765625`;
- median graph lift over control p95 `3.8081556082148498`;
- mean selected graph node count `1.6`;
- hardening pass rates: paraphrase `1.0`, wrong-answer specificity `1.0`,
  negative-fact specificity `1.0`, unrelated-logit preservation `0.955`;
- primary layers `[6,9,11]`, adaptive fallback layers `[0..11]` on strict
  primary failures; the current report used fallback on Russia only.
- 46 unique selected graph neurons.
- current artifact records `model.device = cuda` after the explicit GPU
  benchmark command.

The first output/logit-only whole-model behavior gate also exists:
`npm run benchmark:nn-live-blackbox-llm-behavior-reader:gpu`. It loads real
GPT-2 as a local oracle, but the reader receives only prompt strings, target
token ids, and next-token logits. It builds a behavior graph from subject-swap
controls, adaptive capital-template paraphrase search, answer-token
specificity, relation-control diagnostics, and query-budget accounting.

Current output-only artifact:

```text
benchmarks/release/nn-live-blackbox-llm-behavior-reader/gpt2-output-only-10-facts-behavior-graph/report.json
```

Current output-only result:

- pass `true`;
- 10 valid facts, 10 passed cases, 0 failed cases, pass rate `1.0`;
- gate threshold requires min valid cases `10` and min pass rate `1.0`;
- query count `240`;
- median target rank `2`, max target rank `10`;
- median subject-specificity lift `6.850424766540527`;
- min subject-specificity lift `5.2599224090576175`;
- median answer-specificity margin `3.499279022216797`;
- min answer-specificity margin `2.0098876953125`;
- median relation-template margin `3.4460906982421875`;
- min relation-template margin `1.2060317993164062`;
- median paraphrase retention rate `1.0`;
- access audit: `solverAccess = prompt-to-next-token-logits-only`,
  `weightsUsedBySolver = false`, `hooksUsedBySolver = false`,
  `activationsUsedBySolver = false`.

This output-only rail is separate from the hook-based circuit finder. It may
claim benchmark-scoped prompt-intervention behavior evidence, not hidden-neuron
identity, source provenance, full FFN extraction, or arbitrary LLM
understanding.

The next production path is:

1. Extract a model-agnostic `LLMOracleAdapter` layer
   - Current status: DONE for GPT-2 adapter parity. The live fact-circuit tool
     now uses `tools/llm_reader/adapters/base.py` and
     `tools/llm_reader/adapters/gpt2.py`; the full 50-fact GPT-2 graph gate
     still passes with `graphGate.pass && hardeningGate.pass`.
   - Keep GPT-2 as the fast real-model proof gate, but move GPT-2-specific
     model/tokenizer/layer/hook knowledge behind an adapter.
   - The circuit finder must ask the adapter for blocks, attention heads, MLP
     components, activation family, token positions, logits, hooks,
     interventions, warnings, model identity, tokenizer identity, and budget
     accounting.
   - First target adapters: `gpt2` parity with the current 50/50 gate, then a
     small HuggingFace GPT-Neo/Pythia-style model, then LLaMA/Mistral-style
     SwiGLU MLPs.
   - No adapter may replace the real model with a synthetic teacher or a fitted
     surrogate.
2. Add attention-head and path graph recovery
   - Current status: DONE for first CUDA GPT-2 attention-head/path graph gate.
     `npm run benchmark:nn-live-gpt2-attention-path-circuit-finder:gpu` scans
     real GPT-2 attention heads, builds token/head/target path evidence, and
     validates selected heads through causal attention-path ablation above
     attention-head controls.
   - Emit comparable token/head/neuron/target graph edges with causal effect,
     collateral damage, and control-normalized effect.
3. Add combined attention+MLP graph recovery
   - Current status: DONE for first CUDA GPT-2 combined graph gate.
     `npm run benchmark:nn-live-gpt2-combined-circuit-finder:gpu` discovers
     selected MLP nodes and attention heads, builds one token/head/neuron/target
     graph, and validates the union through simultaneous attention+MLP causal
     ablation above combined controls.
4. Scale model coverage through adapter gates
   - GPT-2 remains the fast GELU proof gate.
   - Pythia/GPT-Neo adapters must prove the framework is not GPT-2-specific.
   - LLaMA/Mistral adapters must prove SwiGLU MLP/value-gate handling without
     weakening graph controls, query budgets, or claim boundaries.
5. Expand the output/logit-only behavior rail
   - Current status: DONE for first GPT-2 10-fact output-only behavior gate.
   - Next gates: larger fact set, OOD paraphrase families, and Pythia/LLaMA/
     Mistral parity through the same prompt-to-logits-only solver boundary.
   - This rail must not receive hooks, hidden activations, weights, source ids,
     or intervention handles.
6. Scale the fact set
   - Run the same graph search over larger factual sets and then larger local
     transformer models only after adapter parity, GPT-2 graph hardening, and
     path recovery gates hold.

This pivot forbids promoting another single-prompt hook-and-ablate report as an
innovation. The innovation target is stable automatic graph evidence from a
real model oracle.

## SOTA Seriousness Checklist

The NN decompiler roadmap is now ordered by dependency, not by excitement. A
later phase may not be promoted while an earlier phase has an unresolved hard
failure.

Phase 0: shallow clean/noisy geometry floor.

- Status: release-gated for 1D, 2D, 3D, randomized 3D, clean 4D/5D line probes,
  noisy 4D/5D line probes, generated adversarial noisy ND, and adaptive
  adversarial noisy ND.
- Purpose: prove that the oracle/probing layer can recover observable
  piecewise-linear geometry before deep composition is attempted.

Phase 1: deep 1D and 2D output-equivalent recovery.

- Status: release-gated for `benchmark:nn-deep-blackbox-relu1d-decompile`,
  `benchmark:nn-deep-blackbox-relu2d-region-graph`, and
  `benchmark:nn-deep-blackbox-relu2d-region-graph:randomized`.
- Purpose: recover the observable folded function first. Original hidden
  weights and exact layer attribution remain out of scope for output-only
  access.

Phase 2: deep 2D hardening.

- Status: release-gated for `benchmark:nn-noisy-deep-blackbox-relu2d-region-graph`
  and `benchmark:nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph`.
- Current proof: repeated noisy/quantized oracle calls on deterministic folded
  2D seed sweep plus adaptive generated near-parallel, offset-collision,
  cancellation, low-amplitude, boundary-grazing, and mixed-oblique folded
  teachers, with median aggregation and explicit query-budget accounting.
- Required gates: held-out accuracy, region/adjacency evidence, query budget,
  first-layer-only baseline gap, and failure artifact if a random/adversarial
  teacher breaks the solver.

Phase 3: compact/adaptive/topology boundary extraction.

- Status: compact-boundary extraction, the first adaptive-budget probe
  selector, boundary-refined local switching, subcell-boundary adaptive
  validation, and topology-boundary recovered-adjacency geometry are
  release-gated for
  `benchmark:nn-compact-deep-blackbox-relu2d-boundary` and
  `benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph`, and
  `benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph`, and
  `benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph`,
  and `benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph`.
- Current boundary: oracle-guided subcell validation selects the lowest passing
  boundary-aware grid and records too-coarse failed candidates; topology-
  boundary geometry clips every recovered adjacency edge to an affine equality
  line inside an expanded contact corridor with no fallback segments. It is
  still recovered-adjacency topology, not topology-perfect true-region merging.
- Next target after topology-boundary geometry: the same boundary discipline in
  deep ND folded recovery.
- Required gates: same accuracy as fixed grid with fewer oracle calls or a
  strictly better boundary artifact.

Phase 4: deep ND folded recovery.

- Status: first release-gated foundation via
  `benchmark:nn-deep-blackbox-relu-nd-region-sampler`, now hardened with
  `benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler`
  and budget-hardened with
  `benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler`,
  then geometry-hardened with
  `benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler`,
  `benchmark:nn-nd-adjacency-geometry-5d-sweep`, and
  `benchmark:nn-nd-adjacency-geometry-6d-robustness`, plus the dependency-free
  baseline court `benchmark:nn-nd-adjacency-geometry-6d-baselines`.
- Current proof: controlled 3D/4D folded two-layer ReLU teachers recovered as
  output-equivalent affine-region samplers from oracle-only random local-gradient
  probes, with query-free kNN region assignment, noisy/quantized repeated-call
  aggregation, adaptive region-vote/local-affine KNN selection, sequential
  noisy holdout budget promotion with rejected-candidate accounting, query-
  budget accounting, recovered ND adjacency geometry from cross-region contact
  witnesses plus affine-equality hyperplanes, a 5D clean/noisy hard-profile
  sweep, a 12-teacher 6D clean/noisy randomized/adversarial robustness gate
  with zero high-tier selections, and a 12-teacher 6D baseline court where GCL
  beats the measured dependency-free baseline families on MAE.
- Required gates: deterministic randomized ND teachers, noisy/adaptive
  adversarial hard-profile sweeps, query-budget accounting, and baseline
  comparison against shallow/linear approximations. Next hardening step is a
  generated adaptive-adversarial 6D teacher pool, then stronger package-backed
  external baselines and harder topology-perfect ND cell-complex audits.

Phase 5: layer attribution only with extra observability.

- Activation-access lanes may claim layer attribution when hidden activations
  are explicitly available.
- Output-only lanes must keep the claim at equivalent-function recovery unless
  an identifiability assumption is stated and gated.

Phase 6: Active-Subspace Neural Decompiler for smooth LLM activation and FFN recovery.

- Status: first foundation release-gated with
  `benchmark:nn-active-subspace-ffn-decompiler`, extended to 4096D stress with
  `benchmark:nn-active-subspace-ffn-4096d-stress`, and now covered for the
  previously observed noisy rank-1 failure mode with
  `benchmark:nn-active-subspace-ffn-noisy`. The first explicit GELU atom
  structure finder is now release-gated with
  `benchmark:nn-gelu-ffn-structure-finder`, and the first rank-adaptive mixed
  GELU structure gate is release-gated with
  `benchmark:nn-gelu-ffn-mixed-structure`. Near-parallel mixed-channel
  stability is now release-gated with
  `benchmark:nn-gelu-ffn-near-parallel-family`, and canonical structure
  invariance under equivalent hidden decompositions is release-gated with
  `benchmark:nn-gelu-ffn-cross-decomposition-invariance`. Noisy/quantized
  mixed GELU structure recovery is release-gated with
  `benchmark:nn-gelu-ffn-noisy-mixed-structure`. The first
  mixed SwiGLU value/gate structure gate is release-gated with
  `benchmark:nn-swiglu-ffn-mixed-structure`, noisy mixed SwiGLU recovery is
  release-gated with `benchmark:nn-swiglu-ffn-noisy-mixed-structure`, and
  shared-gate/cancellation-heavy SwiGLU canonicalization is release-gated with
  `benchmark:nn-swiglu-ffn-shared-gate-canonical`. The first full-slice
  rank-cap negative guard is release-gated with
  `benchmark:nn-gelu-ffn-rank-cap-negative`, and the local real GPT-2 full-FFN
  rank-cap audit is available through
  `research:nn-gpt2-gelu-ffn-full-rank-cap-negative`. The first
  source-neuron provenance verifier is release-gated with
  `benchmark:nn-gelu-ffn-provenance-verifier` for cases where source manifests
  or equivalent instrumentation are available. The first local transformer MLP
  hook gate is release-gated with `benchmark:nn-local-transformer-mlp-hook`,
  using captured residual/MLP/preactivation/activation/contribution/output rows
  from a deterministic local transformer-style block. The first compact real
  GPT-2 top-3, top-5, and top-10 output-only source-structure gates are
  release-gated with `benchmark:nn-gpt2-gelu-ffn-real-structure`,
  `benchmark:nn-gpt2-gelu-ffn-real-structure:top5`, and
  `benchmark:nn-gpt2-gelu-ffn-real-structure:top10`, and the first compact real
  GPT-2 top-20 instrumented source-graph extractor is release-gated with
  `benchmark:nn-gpt2-gelu-ffn-source-extractor`.
- Direction: GCL should not brute-force 4096D. It should query a high-
  dimensional black-box channel, discover the low-dimensional active subspace
  where the function actually changes, then recover explicit structure when the
  activation family is identifiable, such as
  `c + sum_j a_j * gelu(w_j dot x + b_j)`. Fit-only approximators are allowed
  only as diagnostics or baselines; they are not promoted as neural structure
  finders.
- Scope: GELU, SiLU, and SwiGLU-style LLM feed-forward channels are not
  piecewise-linear kink geometry. The ReLU recovered-adjacency gates provide
  oracle discipline, query accounting, and benchmark hygiene, but they do not
  prove GELU/SwiGLU recovery by themselves.
- Current command:
  `npm run benchmark:nn-active-subspace-ffn-decompiler`.
- Stage A: controlled high-dimensional single-channel recovery.
  - GELU teacher: `y = a * gelu(w dot x + b) + c`.
  - SiLU/SwiGLU teacher: `y = (u dot x + bu) * silu(g dot x + bg)`.
  - GCL receives only `f(vector)` oracle calls.
  - Current foundation gate: controlled 256D/1024D teachers, held-out R2 above
    `0.995` for single-channel GELU and above `0.99` for two-direction SwiGLU,
    explicit active-subspace projection evidence, query budget below `50000`,
    and a win against mean/random-subspace/coordinate-subspace baselines.
  - Foundation result: 4/4 passed, max MAE `0.0005708747054871747`, min R2
    `0.999979343109252`, min selected gradient-energy ratio
    `0.9999995561224265`, min true-direction projection score
    `0.9999999887415834`, max raw oracle calls `45360`.
  - Current 4096D stress gate: `npm run
    benchmark:nn-active-subspace-ffn-4096d-stress`, controlled 4096D GELU and
    SwiGLU-style teachers, held-out R2 above `0.999`, explicit true-direction
    projection evidence, query budget below `115000`, and a win against
    mean/random-subspace/coordinate-subspace baselines.
  - 4096D result: 2/2 passed, max MAE `0.0008334928131969248`, min R2
    `0.9998586505688375`, min selected gradient-energy ratio
    `0.9999999998896398`, min true-direction projection score
    `0.9999999998732209`, max raw oracle calls `103504`.
  - Current noisy rank-1 robustness gate: `npm run
    benchmark:nn-active-subspace-ffn-noisy`, controlled stochastic 128D/256D
    GELU teachers with validation against held-out noiseless truth, held-out R2
    above `0.99`, direction projection above `0.998`, explicit query budgets,
    and a win against mean/random-subspace/coordinate-subspace baselines.
  - Noisy result: 2/2 passed, max MAE `0.04567550692434054`, min R2
    `0.993172477248586`, min true-direction projection score
    `0.9980221387236469`, max raw oracle calls `8720`.
- Stage B: explicit GELU structure finder.
  - Current command: `npm run benchmark:nn-gelu-ffn-structure-finder`.
  - Current gate: controlled 96D single/two-channel GELU teachers, oracle calls
    only, recovered input directions, input biases, output weights, held-out
    output metrics, structural parameter error thresholds, query budget, and a
    mean-output baseline check.
  - Current result: 2/2 passed, max MAE `0.002229589390805771`, min R2
    `0.9999894428635375`, min direction projection `0.9999995030788708`, max
    weight-norm error `0.044204547411414996`, max bias error
    `0.006098221054376296`, max raw oracle calls `6088`.
  - Research bridge: `npm run research:nn-gpt2-gelu-ffn-structure-finder`
    runs the same structural solver on one local real GPT-2 layer-0 FFN JSON
    fixture. Current research result: 4/4 cases passed for one single 768D
    GELU neuron plus top-1/top-3 truncated output-channel sums, min R2
    `0.9950578044301109`, max MAE `0.04830918330347917`, min direction
    projection `0.9998685252655729`. This is not release-gated because the
    fixture files live outside the npm package, and it must remain a fixture
    check rather than the center of the roadmap.
  - Important correction: this lane is a structure finder, not a local KNN or
    polynomial surrogate. The earlier active-subspace FFN lane remains useful
    for output fidelity and active-rank discovery, but it does not satisfy the
    deeper "find the neuron structure" goal by itself.
- Stage C: synthetic FFN block recovery.
  - Multiple GELU/SwiGLU channels are summed into a block output slice.
  - This remains the primary bridge from controlled smooth channels toward real
    FFN block recovery. It should strengthen the generic GCL mechanism, not
    overfit a single GPT-2 fixture.
  - Current command: `npm run benchmark:nn-gelu-ffn-mixed-structure`.
  - Current clean mixed-GELU result: controlled 32D three-channel GELU block,
    GCL sees only `f(vector)` plus a max channel budget `4`, not the true
    channel count. The gate passed with effective channel count `3`, duplicate
    pairs `0`, MAE `0.0014707898276633694`, R2 `0.9999899685243602`, min
    direction projection `0.9998572617229938`, max weight-norm error
    `0.03168688372064543`, max bias error `0.009782376019626393`, and raw
    oracle calls `14458`.
  - Current near-parallel family command: `npm run
    benchmark:nn-gelu-ffn-near-parallel-family`.
  - Near-parallel family result: epsilons `1e-1`, `1e-2`, `1e-3`, `1e-4`;
    effective channel counts `3,3,3,3`; duplicate pair counts `0,0,0,0`;
    near-parallel pair counts `0,1,1,1`; max MAE
    `0.002510633422272873`; min R2 `0.999883779624714`; min direction
    projection `0.9993621598125698`; max raw oracle calls `18248`.
  - Current cross-decomposition invariance command: `npm run
    benchmark:nn-gelu-ffn-cross-decomposition-invariance`.
  - Cross-decomposition result: two equivalent 16D GELU hidden decompositions
    with hidden atom counts `3` and `6`, oracle delta `4.441e-16`, recovered
    effective counts `3,3`, canonical atom counts `3,3`, signature distance
    `0.052726404715341284`, max atom distance `0.07144167430917188`, max MAE
    `0.0019376881321900733`, min R2 `0.9999767700614473`, model-agreement MAE
    `0.0021717078319734404`, and max raw oracle calls `25710`.
  - Current noisy mixed GELU command: `npm run
    benchmark:nn-gelu-ffn-noisy-mixed-structure`.
  - Noisy mixed GELU result: 3/3 controlled stochastic/quantized cases passed,
    covering balanced, near-parallel, and weak-but-resolvable three-channel
    teachers. Max MAE `0.003728882150765997`, min R2
    `0.9997459345781228`, max worst error `0.04908248358529477`, min
    direction projection `0.9987338205891503`, effective channel counts
    `3,3,3`, canonical atom counts `3,3,3`, max GCL/best-prediction-baseline MAE ratio
    `0.08120754460430137`, max distribution MAE `0.005876071181397475`,
    min distribution R2 `0.9995038453250289`, max distribution worst error
    `0.09735167309682635`, max distribution GCL/best-prediction-baseline MAE ratio
    `0.08567143575266337`, max raw oracle calls `155012`, with separate
    `structureRecoveryPass` and `predictionSanityCourtPass` gates. The
    prediction-sanity court now covers
    IID uniform, wide-OOD, and shifted-OOD distributions, dependency-free
    baselines, scikit-learn Ridge/polynomial Ridge/RandomForest/ExtraTrees/
    HistGradientBoosting/PLS active-subspace polynomial Ridge, XGBoost, and
    sklearn MLPRegressor. These are fit baselines used to validate output value
    of the recovered structure; they are not structure-decompiler competitors.
  - Current mixed SwiGLU command: `npm run
    benchmark:nn-swiglu-ffn-mixed-structure`.
  - Mixed SwiGLU result: controlled 24D two-block SwiGLU FFN output slice,
    GCL sees only `f(vector)` plus a max pair budget `3`, not the true pair
    count or value/gate directions. The gate passed with effective pair count
    `2`, duplicate pair count `0`, MAE `0.0008717937894303614`, R2
    `0.9999961336476628`, min value-direction projection
    `0.9999996446143484`, min gate-direction projection
    `0.9999996288279506`, mean pair projection `0.9999997837062355`,
    max gate-norm error `0.008049915513471673`, max gate-bias error
    `0.005227321558675863`, and raw oracle calls `15456`.
  - Current noisy mixed SwiGLU command: `npm run
    benchmark:nn-swiglu-ffn-noisy-mixed-structure`.
  - Noisy mixed SwiGLU result: controlled stochastic/quantized family passed
    3/3 cases covering balanced, near-parallel-value, and weak-resolvable
    value/gate-pair teachers. Effective pair counts recovered as `2,2,3`,
    max MAE `0.003938356995627657`, min R2 `0.999874503247406`, max worst
    error `0.03733335221578926`, min value-direction projection
    `0.999310139471901`, min gate-direction projection `0.9998244527769446`,
    min mean pair projection `0.9998467737576421`, max distribution MAE
    `0.007389576137780751`, min distribution R2 `0.999874503247406`, max
    distribution worst error `0.09266107798318846`, and max raw oracle calls
    `345960`.
  - Current shared-gate/cancellation-heavy SwiGLU command: `npm run
    benchmark:nn-swiglu-ffn-shared-gate-canonical`.
  - Shared-gate/cancellation-heavy result: controlled clean/noisy shared-gate
    SwiGLU hard-case passed 2/2 teachers. The true source pair counts were
    `3,3`, but the canonical effective pair counts were `1,1`; GCL collapsed
    the observable shared-gate function into one minimal value/gate pair instead
    of promoting duplicate non-identifiable source channels. Min canonical
    value-direction projection `0.9992111582432534`, min shared-gate projection
    `0.9999470719484649`, noisy cancellation-heavy case cancellation ratio
    `0.9032502910798934`, max MAE `0.00241722715125567`, min R2
    `0.9995956095908356`, max distribution MAE `0.0038654875223167406`, min
    distribution R2 `0.9992352419312791`, max distribution worst error
    `0.07707410825476324`, and max raw oracle calls `276672`.
  - Gate: explicit atom recovery where identifiable, plus output-equivalent
    fallback only when non-identifiability is stated and gated; must include
    per-channel/subspace evidence, held-out accuracy, query budget, and failure
    accounting for collinear/cancelling gates.
  - Important boundary: the cross-decomposition gate proves clean canonical
    structure recovery under an equivalence class. It does not prove semantic
    identity of the original hidden neurons, noisy invariance, or full LLM FFN
    extraction.
  - Current provenance command: `npm run
    benchmark:nn-gelu-ffn-provenance-verifier`.
  - Provenance result: 2/2 controlled 24D source-manifest GELU FFN cases passed,
    including distinct one-to-one source-neuron mapping and duplicate-split
    source-group mapping. Max MAE `0.004659687044604257`, min R2
    `0.9998833857503274`, max model-agreement MAE `0.004659687044604257`,
    min provenance confidence `0.9050399718933542`, max raw oracle calls
    `18184`.
  - Provenance boundary: this is exact source identity only in instrumented
    provenance mode. Output-only recovery remains canonical functional
    structure, not original-neuron identity without source manifests or hooks.
  - Formal contract: `docs/neural-decompiler-formalization.md` defines atom,
    functional equivalence, canonical structure, canonical signature, allowed
    claim levels, mandatory negative cases, and the scaling stress ladder.
  - Required benchmark shape: controlled randomized mixed-channel teachers across
    dimensions, channel counts, rank caps, noisy/quantized outputs, collinear
    directions, weak channels, and cancellation profiles.
  - Implementation subplan:
    1. DONE: Add a rank-adaptive mixed-channel GELU structure finder that receives only
       `f(vector)` and a maximum channel budget, then returns explicit recovered
       atoms, an effective channel count, pruned weak atoms, duplicate/collision
       diagnostics, held-out metrics, and raw oracle-call accounting.
    2. DONE: Add a release gate for controlled mixed GELU blocks where GCL does not know
       the true channel count. The gate must fail if output accuracy is achieved
       without enough recovered atom structure.
    3. DONE for clean near-parallel: Add a deterministic epsilon-family gate for
       two near-parallel GELU channels and require stable direction recovery,
       effective count recovery, duplicate diagnostics, near-parallel
       diagnostics, output fit, and query-budget accounting.
    4. DONE for clean canonical invariance: Add a cross-decomposition gate where
       two mathematically equivalent hidden decompositions must recover the same
       canonical structure signature, with failure preserved if the solver picks
       a larger atom basis.
    5. DONE for first negative-case enforcement: Add
       `benchmark:nn-gelu-ffn-negative-cases`, where the correct result is
       sometimes "surrogate" or "non-identifiable", not forced atom recovery.
       The current gate covers duplicate split/cancellation canonicalization,
       weak channels below noise, weak-but-resolvable channels above noise that
       must recover exactly three atoms, activation-family mismatch, and
       overcomplete budgets.
    6. DONE for first source identity bridge: Add
       `benchmark:nn-gelu-ffn-provenance-verifier`, where recovered GELU atoms
       are mapped to exact source ids/layers/indices and duplicate source
       groups only when a source-neuron manifest or equivalent instrumentation
       is available. The benchmark must record why each mapped source reacts.
    7. DONE for controlled noisy/quantized mixed GELU: Add
       `benchmark:nn-gelu-ffn-noisy-mixed-structure`, where repeated median
       queries recover balanced, near-parallel, and weak-but-resolvable
       three-channel structures under stochastic output noise and finite
       precision. The gate separates structure recovery from prediction-sanity
       output fit and records prediction-baseline winners.
    8. DONE for scikit-learn package-backed prediction-sanity court: the noisy mixed GELU gate
       now compares against mean/ridge/random-GELU dependency-free baselines
       plus scikit-learn Ridge, polynomial Ridge, RandomForest, ExtraTrees,
       HistGradientBoosting, and PLS active-subspace polynomial Ridge. The gate
       still reports structure evidence separately from output fit.
    9. DONE for distribution and stronger package-backed prediction-sanity court: the same
       noisy mixed GELU gate now evaluates IID uniform, wide-OOD, and shifted-OOD
       truth sets and requires the recovered GCL structure to beat the best
       prediction baseline on every distribution while covering XGBoost and
       sklearn MLPRegressor. This is not a direct decompiler-vs-XGBoost claim.
    10. DONE for controlled clean mixed SwiGLU: Add
       `benchmark:nn-swiglu-ffn-mixed-structure`, where GCL receives a maximum
       pair budget and must recover explicit value/gate pairs, effective pair
       count, held-out metrics, structure projections, and query budget.
    11. DONE for controlled noisy/adversarial mixed SwiGLU: Add
       `benchmark:nn-swiglu-ffn-noisy-mixed-structure`, where repeated
       median queries recover explicit value/gate pairs and effective pair
       counts across balanced, near-parallel-value, and weak-resolvable
       stochastic/quantized teachers.
    12. DONE for shared-gate/cancellation-heavy SwiGLU canonicalization: Add
       `benchmark:nn-swiglu-ffn-shared-gate-canonical`, where source channels
       share one gate and may cancel heavily. The gate distinguishes
       non-identifiable source-pair identity from the recoverable minimal
       functionally equivalent value/gate pair instead of silently promoting an
       arbitrary factorization.
    13. DONE for the first local transformer MLP hook: Add
       `benchmark:nn-local-transformer-mlp-hook`. Capture real residual-stream
       inputs and MLP output channels/slices from a local transformer block,
       then run explicit structure recovery with source/hook provenance where
       available and hard-fail extraction/tokenization/data-shape failures.
    14. DONE for compact real GPT-2 top-3 output-only source-structure:
       Add `benchmark:nn-gpt2-gelu-ffn-real-structure`, where GCL receives
       oracle calls over a real GPT-2 layer-0 top-3 output-channel slice,
       recovers explicit GELU atoms, and maps them through functional
       provenance.
    15. DONE for compact real GPT-2 top-5 output-only source-structure:
       Add `benchmark:nn-gpt2-gelu-ffn-real-structure:top5`, using
       full-budget candidate selection plus source-blind fixed-direction scalar
       refinement. During recovery GCL receives only `f(vector)` oracle calls;
       source weights/manifests are used only after recovery for provenance.
    16. DONE for compact real GPT-2 top-8/top-10 output-only source-structure:
       Add `benchmark:nn-gpt2-gelu-ffn-real-structure:top8` and
       `benchmark:nn-gpt2-gelu-ffn-real-structure:top10`. The first top-8 run
       exposed that output-only fidelity was not enough because two weaker
       channels had poor provenance alignment; the solver was strengthened with
       source-blind active-basis gradient-consistency training. Top-8 and
       top-10 then recovered every compact source group from oracle calls only.
    17. DONE for compact real GPT-2 top-20 instrumented source extraction:
       Add `benchmark:nn-gpt2-gelu-ffn-source-extractor`, where GCL builds an
       exact source graph from real GPT-2 source weights or equivalent hook
       manifests, validates IID/OOD held-out outputs, and verifies every atom
       against exact GPT-2 source ids with strict parameter provenance. This is
       deliberately separate from output-only recovery.
    18. DONE for randomized real-LLaMA 16-unit source-blind structure recovery:
       Add `benchmark:nn-live-blind-swiglu-layer-expansion:llama-16unit-residual-block-randomized`.
       The solver recovers two deterministic-random real LLaMA-family 3072D
       16-unit scalar SwiGLU blocks from oracle values only. A gate-norm plus
       tail-responsibility weak-block selector repairs under-resolved atoms;
       source weights remain post-hoc and are used only for exact one-to-one
       gate/up assignment. Current gate, regenerated 2026-08-02: `2/2` pass,
       min R2 `0.9999855916796467`, max worst error `0.006249561164756855`,
       min paired atom joint cosine `0.9999610978737026`, and all `16/16`
       atoms above `0.99` in both blocks. Gate directions are measured from
       where the function's second derivative breaks rather than fitted, after
       an earlier reading of `1/2` traced to a nonlinear fit recovering two
       sources `0.0095` apart as two copies of their bisector.
    19. DONE for fresh structure-first real-Mistral 16-atom recovery without a
       random nonlinear initial fit:
       Add `benchmark:nn-live-blind-swiglu-asymptotic-cold-start`. The gate
       reconstructs the 4096D active subspace and explicit gate/up pairs from
       a new `1e-6`-quantized scalar-oracle run, forbids fit-snapshot input and
       nonlinear random initial atom fitting, and hard-gates `16/16` post-hoc pairs at
       joint cosine `0.99` under `1.5M` oracle calls. Current result: R2
       `0.9995787659698843`, MAE `0.00000616945597616291`, worst error
       `0.000032305729901930614`, subspace minimum cosine
       `0.9996985570973159`, paired minimum `0.9972788963034496`, and raw
       oracle calls `1279238`.
    20. DONE for process-separated, access-attested real-Mistral cold start:
       Add `benchmark:nn-live-blind-swiglu-isolated-cold-start`. The
       orchestrator sends the teacher manifest only to the oracle server over a
       private inherited file descriptor. The dedicated solver receives a
       narrow argv plus an opaque eval-only Unix binary endpoint, and the
       post-hoc source verifier starts only after the recovered artifact is
       frozen and the solver has exited. The gate requires matching hash-chained
       solver/server query transcripts, an immutable frozen artifact, and a
       non-empty hash-stamped `strace` witness with no model, weight, or control-
       file access by the solver. Current run: R2 `0.9995787659698843`, all
       `16/16` paired atoms above `0.99`, and `1279238/1500000` oracle calls.
    21. DONE for cross-family process-separated cold start: Add
       `benchmark:nn-live-blind-swiglu-isolated-cold-start-court`. It runs the
       unchanged opaque solver protocol against two real model families and two
       oracle modes: Mistral 4096D layer `15` through a
       `down-proj-output-channel` oracle and LLaMA 3072D layer `9` through a
       `selected-unit-sum` oracle. Each solver runs inside a private randomized
       work directory, so no output path carries a model or layer name and
       `solverSourceLocatorMatches` stays empty. Per case the court re-checks
       the post-hoc source-unit witness, transcript digest match, frozen-artifact
       immutability, verifier-after-solver ordering, current runtime source
       hashes, and a hash-stamped `strace` witness with zero model/weight and
       zero control-file access. Current aggregate: pass `true`, cases `2/2`,
       min R2 `0.9995787659698843`, max MAE `0.0003064998140072596`, max worst
       error `0.0019215374240817429`, `16/16` paired atoms above `0.99` per
       case, max `1279238/1500000` calls per case, and `2298770` total calls.
       This answers the previous "widen the process-separated court across model
       families" hardening item for the fixed-layer case; randomized early/
       middle/late layer selection is still open.
    22. NEXT: Add scaling stress as a separate ladder: `768D -> 4096D -> 16384D`,
       then multi-layer smooth composition, always with query budget and
       negative-case behavior. A dimension jump is not a structure claim by
       itself.
    23. DONE for full-slice/rank-cap negative guarding: Add
       `benchmark:nn-gelu-ffn-rank-cap-negative`, where a self-contained
       synthetic 768D/3072-channel full-slice teacher must report
       `insufficient-rank` under a too-small rank cap instead of promoting a
       weak structure claim. Add
       `research:nn-gpt2-gelu-ffn-full-rank-cap-negative` as a local audit over
       the uncommitted full GPT-2 fixture.
    24. NEXT: Only after blind top-K is efficient should full FFN slices be used
       as research stress tests. They must not replace the generic mixed-
       channel gate.
- Stage D: local transformer MLP hook.
  - Current command: `npm run benchmark:nn-local-transformer-mlp-hook`.
  - Current result: deterministic local transformer-style block, `384` captured
    residual/MLP/preactivation/activation/contribution/output hook rows, one
    target MLP output slice, source expansion neurons `3`, recovered effective
    channels `3`, hook MAE `0.005720353338158273`, hook R2
    `0.9998897778278036`, worst hook error `0.014246698868125185`, min
    provenance confidence `0.9106904584057108`, hook preactivation/activation/
    output integrity errors `0`/`0`/`2.220446049250313e-16`, and raw oracle
    calls `19540`.
  - Gate: recover explicit structure first, verify hook-row integrity, then map
    recovered atoms through source/hook provenance. All extraction,
    tokenization, data-shape, preactivation, activation, and target-output
    mismatches are hard failures.
- Claim boundary: GCL may claim local instrumented transformer-style MLP source
  mapping for this gate. It may not claim full GPT-2/LLM neuron decompilation,
  exact GELU/SwiGLU parameter recovery for arbitrary production blocks, or
  mechanistic understanding of production transformer blocks from this gate
  alone.
- Stage E: compact real GPT-2 FFN source structure.
  - Current command: `npm run benchmark:nn-gpt2-gelu-ffn-real-structure`.
  - Current top-3 result: compact fixture generated from real GPT-2 layer-0 FFN
    weights for output channel `0`, top-3 source channels `1614,2452,2053`,
    dimension `768`; recovered effective channel count `3`, all source groups
    mapped, MAE `0.04964509934587569`, R2 `0.9942876980025065`, worst error
    `0.1585760339175079`, min provenance confidence `0.7918710775184039`,
    min direction cosine `0.9998877789545968`, and raw oracle calls `116988`.
  - Current top-5 command: `npm run
    benchmark:nn-gpt2-gelu-ffn-real-structure:top5`.
  - Current top-5 result: compact fixture generated from real GPT-2 layer-0 FFN
    weights for output channel `0`, top-5 source channels
    `1614,2452,2053,486,2172`, dimension `768`; recovered effective channel
    count `5`, all source groups mapped, MAE `0.010216135944799058`, R2
    `0.9998249080496286`, worst error `0.06915410304531516`, min provenance
    confidence `0.9433099991784843`, min direction cosine
    `0.9997811523380504`, and raw oracle calls `64820`.
  - Current top-8 command: `npm run
    benchmark:nn-gpt2-gelu-ffn-real-structure:top8`.
  - Current top-8 result: compact fixture generated from real GPT-2 layer-0 FFN
    weights for output channel `0`, top-8 source channels
    `1614,2452,2053,486,2172,1981,137,257`, dimension `768`;
    recovered effective channel count `8`, all source groups mapped, MAE
    `0.010762522489570131`, R2 `0.9998280690875673`, worst error
    `0.07145730981694887`, min provenance confidence
    `0.9651104040584654`, min direction cosine `0.9998577803120876`, and
    raw oracle calls `213512`.
  - Current top-10 command: `npm run
    benchmark:nn-gpt2-gelu-ffn-real-structure:top10`.
  - Current top-10 result: compact fixture generated from real GPT-2 layer-0 FFN
    weights for output channel `0`, top-10 source channels
    `1614,2452,2053,486,2172,1981,137,257,1554,2211`, dimension `768`;
    recovered effective channel count `10`, all source groups mapped, MAE
    `0.01549465009745577`, R2 `0.9996487749720989`, worst error
    `0.10184722252510248`, min provenance confidence `0.8397038910070366`,
    min direction cosine `0.998244983491102`, and raw oracle calls `219268`.
  - Current full-slice/rank-cap negative command: `npm run
    benchmark:nn-gelu-ffn-rank-cap-negative`.
  - Current full-slice/rank-cap negative result: self-contained synthetic
    768D/3072-channel GELU full-slice teacher with 48 observable source groups
    and rank cap 12 correctly returned `insufficient-rank`; effective channels
    `11`, source coverage `0.3290814157449527`, MAE
    `1.4944170756394612`, R2 `0.8744173553254918`, worst error
    `7.696315351510922`, and raw oracle calls `79172`.
  - Current local real GPT-2 full-rank audit command: `npm run
    research:nn-gpt2-gelu-ffn-full-rank-cap-negative`.
  - Current local real GPT-2 full-rank audit result: uncommitted full GPT-2
    layer-0 FFN fixture with 3072 channels and rank cap 20 correctly returned
    `insufficient-rank`; top-rank contribution coverage
    `0.034907769543268366`, capped-vs-full MAE `4.982383447755312`,
    R2 `-0.0014440317377399925`, worst error `16.26971003902171`, sample count
    `96`.
  - Gate: recover explicit GELU atoms from oracle calls over the compact real
    GPT-2 FFN slice, then verify exact source channel ids through functional
    provenance: direction, activation trace, contribution trace, and why-reacts
    evidence. Strict parameter identity is not claimed because the first strict
    run exposed real GELU scale/output-weight gauge drift.
  - Next hard case: adaptive blind top-K expansion that can turn the current
    insufficient-rank diagnostic into progressively larger recovered source
    structure without crossing into unsupported full-3072 claims.
- Stage F: compact real GPT-2 FFN source extractor.
  - Current command: `npm run benchmark:nn-gpt2-gelu-ffn-source-extractor`.
  - Current result: compact fixture generated from real GPT-2 layer-0 FFN
    weights for output channel `0`, top-20 source channels
    `1614,2452,2053,486,2172,1981,137,257,1554,2211,638,1984,2733,1784,2547,1311,1476,2836,763,1169`,
    dimension `768`; exact source graph built, strict provenance verified, all
    source groups mapped, all recovered atoms mapped, max MAE
    `4.561527660225847e-17`, min R2 `1`, max worst error
    `1.7763568394002505e-15`, min provenance confidence
    `0.9999999999999998`, and raw validation oracle calls `3600`.
  - Gate: use real GPT-2 source weights or equivalent hook/source manifests to
    build an explicit GELU source graph, then validate exact held-out outputs
    across IID GPT-2 activation scale, wide-OOD, and shifted-OOD distributions.
  - Claim boundary: this is exact instrumented source extraction, not output-
    only source identity, not blind full-FFN decompilation, and not arbitrary
    3072-channel GPT-2 extraction.
- Stage G: live GPT-2 reader as oracle instrumentation.
  - Current command:
    `npx ts-node src/cli/index.ts explain-llm --model gpt2 --prompt "The capital of France is" --layer 0 --top-neurons 5 --top-logits 5 --output artifacts/llm-reader/gpt2-layer0-france.json`.
  - Current local proof: the tool loads real GPT-2 through the project-managed
    Python runtime, hooks real MLP activations, reports causal ablation effects
    on next-token logits, and records negative controls. The hardened
    50-control artifact is
    `artifacts/llm-reader/gpt2-layer0-france-negative-controls-50.json`.
  - Current hardening: random-active same-amplitude, random-inactive,
    off-prompt, and same-neuron other-position controls are explicit; random
    and off-prompt control families are disjoint and audited through
    `crossFamilyNeuronOverlap`.
  - Claim boundary: this is real model instrumentation and causal measurement,
    not the GCL invention by itself. It may not be called automatic circuit
    discovery. That proof now lives in Stage H and remains benchmark-scoped.
- Stage H: live GPT-2 MLP fact-circuit graph finder.
  - Current command: `npm run benchmark:nn-live-gpt2-fact-circuit-finder`.
  - Current artifact:
    `benchmarks/release/nn-live-gpt2-fact-circuit-finder/gpt2-50-facts-graph-layer-scan/report.json`.
  - Current result: pass `true`, strict case definition
    `graphGate.pass && hardeningGate.pass`, 50 valid facts, 50 strict-passed
    cases, 50 graph-passed cases, 50 hardening-passed cases, pass rate `1.0`,
    median joint graph target-logit drop `0.25433349609375`, p10 graph drop
    `0.1052703857421875`, median graph-control p95 positive target-logit drop
    `0.06884765625`, median graph lift over control p95
    `3.8081556082148498`, mean selected graph node count `1.6`, and 46 unique
    selected graph neurons. The current artifact records `model.device = cuda`.
  - Output: recovered token/neuron/target circuit graphs with token-support
    edges, neuron-target causal-drop edges, selected-graph joint ablation, and
    graph-level negative controls.
  - Claim boundary: this proves benchmark-scoped MLP graph circuit discovery
    for the configured real GPT-2 factual prompt set. It does not prove
    attention-head/path recovery, full GPT-2 FFN extraction, arbitrary LLM
    understanding, or output-only original-neuron identity.
- Stage I: graph hardening with paraphrases and negative facts.
  - DONE for release gate: the fact-circuit benchmark now validates paraphrased
    prompts, wrong-answer targets, negative facts, unrelated factual answer
    logits, graph-level controls, explicit failedSubjects artifacts, and strict
    per-case gating.
  - Current hardening result: case-level pass `50/50`, aggregate pass `true`,
    paraphrase pass rate `1.0`, wrong-answer specificity `1.0`, negative-fact
    specificity `1.0`, unrelated-logit preservation `0.955`, over `80`, `150`,
    `100`, and `400` validation items respectively.
- Stage J: model-agnostic LLM adapter framework.
  - Current status: Gate 1 is implemented for GPT-2 adapter parity. The
    monolithic fact-circuit reader now goes through `LLMOracleAdapter` /
    `GPT2OracleAdapter` for real model loading, tokenizer/model identity, MLP
    component lookup, activation-family metadata, activation ablation, negative
    control selection, and runtime dependency metadata.
  - Verified gate: the full 50-fact GPT-2 graph benchmark still passes with
    strict case definition `graphGate.pass && hardeningGate.pass`.
  - Runtime acceleration lane: CPU remains the default release path, while GPU
    execution is explicit through `.gcl-python-gpu`, `npm run python:setup:gpu`,
    and `npm run benchmark:nn-live-gpt2-fact-circuit-finder:gpu`. The GPU setup
    must prove CUDA through `torch.cuda.is_available()` and a real CUDA tensor
    operation before any GPU benchmark claim is accepted.
  - Gate 2: Pythia/GPT-Neo GELU adapters must run the same graph-search and
    hardening protocol with model/tokenizer identity, layer/component mapping,
    controls, warnings, and budgets recorded.
  - Gate 3: LLaMA/Mistral SwiGLU adapters must extend component addressing and
    FFN value/gate handling without weakening graph controls, query budgets, or
    claim boundaries.
  - Claim boundary: cross-model adapter support proves protocol portability only
    when each adapter has a command, artifact, gate, and negative controls. It
    does not prove arbitrary LLM understanding.
- Stage K: attention-head and path graph recovery.
  - Current status: first CUDA gate implemented.
  - Current command: `npm run benchmark:nn-live-gpt2-attention-path-circuit-finder:gpu`.
  - Current artifact:
    `benchmarks/release/nn-live-gpt2-attention-path-circuit-finder/gpt2-50-facts-attention-path-layer-scan/report.json`.
  - Current result: pass `true`, 50 valid facts, 50 passed cases, pass rate
    `1.0`, median attention graph target-logit drop `1.7271881103515625`, p10
    graph drop `1.1454010009765625`, median attention-control p95 target-logit
    drop `0.448638916015625`, median lift over control p95
    `3.7378692395502666`, mean selected head count `1.04`, unique selected
    heads `6`, median wrong-answer collateral ratio `0.7162562846461261`,
    skipped rows `33`, `model.device = cuda`.
  - Output: recovered token/head/neuron/target graph edges with edge/effect
    scores, collateral-damage metrics, control-normalized effect, and stability
    metrics.
  - Claim boundary: this proves benchmark-scoped attention-head/path graph
    discovery, not answer-specific factual identity. Wrong-answer collateral is
    reported but not gated because attention heads can carry broad task or
    token-class effects. Skipped rows are explicit and excluded from valid
    pass/fail counts when the answer is not a single token or the target
    baseline rank is low-confidence. It is not attention-to-MLP composition
    recovery yet.
- Stage K2: combined attention+MLP graph recovery.
  - Current status: first CUDA gate implemented.
  - Current command: `npm run benchmark:nn-live-gpt2-combined-circuit-finder:gpu`.
  - Current artifact:
    `benchmarks/release/nn-live-gpt2-combined-circuit-finder/gpt2-50-facts-attention-mlp-combined-graph/report.json`.
  - Current result: pass `true`, 50 valid facts, 50 passed cases, pass rate
    `1.0`, MLP component pass rate `1.0`, attention component pass rate
    `1.0`, median combined target-logit drop `1.92523193359375`, p10
    combined drop `1.1469650268554688`, median full-control combined p95
    target-logit drop `0.417144775390625`, median combined lift over full-
    control p95 `4.333544871993313`, mean selected MLP nodes `1.56`, mean
    selected attention heads `1.02`, `model.device = cuda`.
  - Output: one token/head/neuron/target union graph with simultaneous
    attention+MLP ablation evidence, full component-replacement gate controls,
    diagnostic MLP-swap and attention-swap controls, component pass rates, and
    explicit failed subject reporting (`[]` in the current artifact).
  - Claim boundary: this proves benchmark-scoped combined graph discovery for
    the configured GPT-2 factual prompt set. It is not ordered path-patching
    through every residual stream edge, not full GPT-2 circuit recovery, not
    full FFN extraction, not arbitrary LLM understanding, and not output-only
    original-neuron identity.
- Stage K3: ordered residual path recovery.
  - Current status: first CUDA GPT-2 gate implemented, Pythia/GPT-NeoX
    second-adapter gate strengthened to 20/20, and first real LLaMA-family and
    Mistral-family SwiGLU preview gates implemented.
  - Current command: `npm run benchmark:nn-live-gpt2-ordered-residual-path-circuit-finder:gpu`.
  - Second-adapter command:
    `npm run benchmark:nn-live-pythia-ordered-residual-path-circuit-finder:gpu`.
  - LLaMA-family SwiGLU command:
    `npm run benchmark:nn-live-llama-swiglu-ordered-residual-path-circuit-finder:gpu`.
  - Mistral-family SwiGLU command:
    `npm run benchmark:nn-live-mistral-swiglu-ordered-residual-path-circuit-finder:gpu`.
  - Ordered-path stability command:
    `npm run benchmark:nn-live-ordered-path-stability:gpu`.
  - Current artifact:
    `benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json`.
  - Second-adapter artifact:
    `benchmarks/release/nn-live-pythia-ordered-residual-path-circuit-finder/pythia-70m-20-facts-ordered-residual-path/report.json`.
  - LLaMA-family SwiGLU artifact:
    `benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-20-facts-ordered-residual-path/report.json`.
  - Mistral-family SwiGLU artifact:
    `benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-10-facts-ordered-residual-path/report.json`.
  - Ordered-path stability artifact:
    `benchmarks/release/nn-live-ordered-path-stability/real-model-paraphrase-layer-replay/report.json`.
  - Current result: pass `true`, 50 valid facts, 50 passed cases, pass rate
    `1.0`, combined graph pass rate `1.0`, median selected path activation
    delta `0.18021255731582642`, p10 selected path activation delta
    `0.09223440289497375`, median full-control path p95 activation delta
    `0.0318724343786016`, median ordered path lift over full-control p95
    `5.259007093850701`, median combined target-logit drop
    `1.9063301086425781`, skipped rows `33`, `model.device = cuda`.
  - Second-adapter result: pass `true`, real EleutherAI/pythia-70m-deduped /
    GPT-NeoX, 20 valid facts, 20 passed cases, pass rate `1.0`, combined graph
    pass rate `1.0`, median selected path activation delta
    `0.51171875`, p10 selected path activation delta
    `0.3111328125`, median full-control path p95 activation delta
    `0.06836662292480464`, median ordered path lift
    `10.015731969055064`, median combined target-logit drop `2`, failed
    subjects `[]`, `adapter = gpt-neox`, `model.device = cuda`.
    Deterministic per-subject controls and multi-component ordered rescue
    close the previous Japan/Israel hard cases without threshold weakening.
  - LLaMA-family SwiGLU result: pass `true`, real
    `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`,
    `LlamaForCausalLM`, 28 layers, hidden size `3072`, intermediate size
    `8192`, 20 valid facts, 20 passed cases, pass rate `1.0`, combined graph
    pass rate `1.0`, median selected path activation delta `0.046875`, p10
    selected path activation delta `0.023046875`, median full-control path p95
    activation delta `0.0017847249284386607`, median ordered path lift
    `39.35359036498139`, median combined target-logit drop `0.9375`, failed
    subjects `[]`, skipped cases `1`,
    `adapter = llama-swiglu`, `activationFamily = swiglu`,
    `model.device = cuda`, `synthetic = false`, and `surrogate = false`.
    The addressable MLP unit is the real
    `SiLU(gate_proj(x))*up_proj(x)` pair before `down_proj`, not a GELU-style
    compatibility fiction.
  - Mistral-family SwiGLU result: pass `true`, real
    `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`, `MistralForCausalLM`, 32
    layers, hidden size `4096`, intermediate size `14336`, 8 key/value heads,
    10 valid facts, 10 passed cases, pass rate `1.0`, combined graph pass rate
    `1.0`, median selected path activation delta `0.203125`, p10 selected path
    activation delta `0.0921875`, median full-control path p95 activation
    delta `0.04531249999999998`, median ordered path lift
    `5.104890772220551`, median combined target-logit drop `0.34375`, failed
    subjects `[]`, skipped cases `10`, `adapter = mistral-swiglu`,
    `activationFamily = swiglu`, `model.device = cuda`, `synthetic = false`,
    and `surrogate = false`. The addressable MLP unit is the real
    `SiLU(gate_proj(x))*up_proj(x)` pair before `down_proj`.
  - Ordered-path stability result: suite pass `true`, 4/4 model reports pass
    across GPT-2, Pythia/GPT-NeoX, Llama/SwiGLU, and Mistral/SwiGLU. GPT-2 and
    Pythia/GPT-NeoX each check 5 cases; Llama/SwiGLU and Mistral/SwiGLU each
    check 4 cases. Each case probes 3 paraphrase prompts and layer shifts
    `[-2,-1,1,2]` with at least 2 valid layer controls. Original replay pass
    rate is `1.0` for all model reports; paraphrase pass rates are GPT-2
    `1.0`, Pythia/GPT-NeoX `0.9333333333333333`, Llama/SwiGLU `1.0`, and
    Mistral/SwiGLU `1.0`; neighboring-layer specificity rate is `1.0`, max
    replay error is `0`, and failed subjects are `[]`.
  - Output: attention->residual->MLP->target path records with selected
    attention heads, selected MLP nodes, path activation deltas, full-control
    path deltas, combined target-logit evidence, and order-aware rescue
    attempts when a causal union graph is not a valid attention-before-MLP
    path.
  - Claim boundary: these gates prove benchmark-scoped ordered residual path
    evidence across real GPT-2, GPT-NeoX/Pythia, LLaMA-family SwiGLU, and
    Mistral-family SwiGLU runtimes, plus replay/paraphrase/layer-control
    stability for configured cases. They do not prove exhaustive residual
    patching for every edge, full-model circuit recovery, full FFN extraction,
    arbitrary LLM understanding, arbitrary OOD paraphrase robustness, semantic
    original-neuron identity, or output-only original-neuron identity.
- Stage K4: graph-level adaptive LLM circuit reader.
  - Target: turn benchmark-scoped path evidence into a real reader workflow
    that takes a prompt/target and returns a circuit family with stable edges,
    prompt-specific edges, alternate paths, controls, and why-reacts evidence.
  - Non-negotiable mode split: if this stage uses weights, hooks, cached
    activations, or source manifests to choose the circuit, it is an
    instrumented reader/witness gate, not a black-box decompiler gate. A
    black-box 4096D/LLM claim requires a blind solver path that receives only
    oracle calls/logits/channels and uses weights/hooks only after recovery for
    evaluation or provenance.
  - Required behavior:
    1. run real model runtimes only through adapters with model/tokenizer/
       component identity recorded;
    2. scan candidate attention heads, MLP units, residual positions, layers,
       and token positions without hard-coding GPT-2-only assumptions into the
       circuit-finder core;
    3. build a graph containing token, head, neuron/MLP-unit, residual, and
       target-logit nodes;
    4. validate graph edges with causal ablation, ordered replay, negative
       controls, neighboring-layer controls, random active/inactive controls,
       off-prompt controls, and same-component other-position controls;
    5. run paraphrase and OOD paraphrase probes as first-class gates;
    6. when a paraphrase breaks the selected path, search for an alternate path
       and report the circuit change instead of hiding the failure or calling
       the original path universal;
    7. emit why-reacts traces: trigger probes, preactivation, activation,
       contribution, residual-path delta, and target-logit effect;
    8. record query/runtime budgets, skipped cases, failed probes, runtime
       warnings, and control-family overlap audits.
  - First gate: OOD paraphrase stress for ordered paths. The current stability
    gate covers configured paraphrase families; the next gate must include
    harder prompt transforms and require alternate-path discovery when the
    original path no longer carries the evidence.
  - Claim boundary: this stage is still a reader for selected prompts/targets
    and configured model families. Passing it would not prove full LLM
    understanding, semantic original-neuron identity from output-only access,
    complete circuit enumeration, or arbitrary production-model decompilation.
- Stage K5: blind 4096D LLM-channel decompiler gate.
  - Target: use a real local transformer block only as an oracle provider,
    then hide weights, hooks, layer internals, and source neuron ids from the
    solver while it recovers active subspace and functional structure for a
    selected 4096D channel/logit/block output.
  - Required behavior:
    1. construct an oracle from a real local HF transformer component;
    2. expose to the solver only `f(x)` calls and budget metadata;
    3. forbid direct reads of weights, hidden activations, manifests, or source
       ids in the solver process;
    4. recover active subspace, candidate atoms/gates, and held-out functional
       fidelity;
    5. run post-hoc witness comparison against hooks/weights only after blind
       recovery has produced its structure;
    6. report `solverAccess = oracle-only` and fail the gate if that invariant
       is violated.
  - First gate: single selected 4096D MLP channel or target-logit projection
    from a local HF transformer, with the solver run in oracle-only mode and
    witness comparison separated into a post-recovery audit file.
  - Current first sweep gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:sweep`.
    The gate builds real local
    `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` SwiGLU oracles for selected
    layer/unit pairs, then runs the solver in a separate parent process that
    receives only batch `f(x)` outputs. The post-hoc witness is requested only
    after recovery. Aggregate artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-4unit-layer-sweep-2probe-oracle-only/report.json`.
    Result: pass `true`, cases `4/4`, dimension `4096`, active rank `2`, total
    raw oracle calls `73984`, max raw oracle calls per case `18496`, min
    validation R2 `0.9999845855712838`, max validation MAE
    `1.5160815735029332e-05`, max worst error `0.00017546677141010852`, min
    subspace principal cosine `0.9999997892743723`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
    Per-case artifacts include layer `8`/unit `123`, layer `16`/unit `2048`,
    layer `20`/unit `0`, and the harder layer `24`/unit `17`, where the
    stricter case passed with R2 `0.9999985693061243` and MAE
    `4.612942860071953e-6` under the lower-query 2-probe budget. The earlier
    2-probe failure exposed a solver bug: centered covariance collapses rank
    when gradient witness count equals active rank. The fix uses raw second
    moments in that regime instead of hiding the failure or raising the budget.
  - Current randomized hardening gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:randomized`.
    The gate uses deterministic-random real Mistral-family layer/unit selection
    with the same oracle-only access invariant and the same 2-probe query
    budget. Aggregate artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-8unit-layer-sweep-2probe-oracle-only/report.json`.
    Result: pass `true`, cases `8/8`, dimension `4096`, active rank `2`, total
    raw oracle calls `147968`, max raw oracle calls per case `18496`, min
    validation R2 `0.9999570341720895`, max validation MAE
    `2.4326629529818456e-05`, max worst error `0.00044138616892425153`, min
    subspace principal cosine `0.9999999957944806`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current LLaMA parity gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama`.
    The gate uses the same oracle-only solver against real local
    `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` SwiGLU units, with
    expected hidden dimension `3072` and intermediate size `8192`. Aggregate
    artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-4unit-layer-sweep-2probe-oracle-only/report.json`.
    Result: pass `true`, cases `4/4`, dimension `3072`, active rank `2`, total
    raw oracle calls `57600`, max raw oracle calls per case `14400`, min
    validation R2 `0.9999999964505302`, max validation MAE
    `1.0737975340964412e-05`, max worst error `0.00011067984901980132`, min
    subspace principal cosine `0.9999999979627894`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized LLaMA parity gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-randomized`.
    The gate uses deterministic-random real LLaMA-family layer/unit selection
    with the same oracle-only access invariant and the same 2-probe query
    budget. Aggregate artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-8unit-layer-sweep-2probe-oracle-only/report.json`.
    Result: pass `true`, cases `8/8`, dimension `3072`, active rank `2`, total
    raw oracle calls `115200`, max raw oracle calls per case `14400`, min
    validation R2 `0.9999981324158254`, max validation MAE
    `0.00016030715095856503`, max worst error `0.001568794345920943`, min
    subspace principal cosine `0.9999994111232651`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current small multi-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi`.
    The gate uses a real local Mistral-family scalar oracle that returns the
    sum of two layer-20 SwiGLU unit outputs, units `[0,17]`. The solver receives
    only batch `f(x)` outputs, recovers a rank-4 active subspace, and fits an
    explicit 2-atom SwiGLU sum. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-2atom-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `2`, atomCount `2`, raw
    oracle calls `44128`, validation R2 `0.9998005671654032`, validation MAE
    `9.702990494674492e-05`, max worst error `0.0012327953741858785`, subspace
    min principal cosine `0.9999999793123026`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized small-block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles; each oracle returns the sum of two same-layer SwiGLU unit outputs.
    The solver receives only batch `f(x)` outputs, recovers rank-4 active
    subspaces, and fits explicit 2-atom SwiGLU sums. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-2unit-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `2`, max atomCount `2`,
    total raw oracle calls `88256`, max raw oracle calls per case `44128`, min
    validation R2 `0.9994744138350042`, max validation MAE
    `0.00014261562921587213`, max worst error `0.0027002211714715688`,
    subspace min principal cosine `0.9999999823144813`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current first 3-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-3unit`.
    The gate uses a real local Mistral-family scalar oracle that returns the
    sum of three layer-20 SwiGLU unit outputs, units `[0,17,123]`. The solver
    receives only batch `f(x)` outputs, recovers a rank-6 active subspace, and
    fits an explicit 3-atom SwiGLU sum. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-3atom-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `3`, atomCount `3`, raw
    oracle calls `61968`, validation R2 `0.9992245717327005`, validation MAE
    `0.0002548553473914666`, max worst error `0.0029161037108420967`, subspace
    min principal cosine `0.9999999974474356`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized 3-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-3unit-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles; each oracle returns the sum of three same-layer SwiGLU unit
    outputs. The solver receives only batch `f(x)` outputs, recovers rank-6
    active subspaces, and fits explicit 3-atom SwiGLU sums. This gate also
    enforces `maxWorstError <= 0.02`, because R2 alone hid larger tail errors
    in the first manual inspection. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-3unit-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `3`, max atomCount `3`,
    total raw oracle calls `123936`, max raw oracle calls per case `61968`,
    min validation R2 `0.9962650545162728`, max validation MAE
    `0.0005119471166077025`, max worst error `0.010120884383530829`,
    subspace min principal cosine `0.9999999878109772`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current first 4-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-4unit`.
    The gate uses a real local Mistral-family scalar oracle that returns the
    sum of four layer-20 SwiGLU unit outputs, units `[0,17,123,456]`. The
    solver receives only batch `f(x)` outputs, recovers a rank-8 active
    subspace, and fits an explicit 4-atom SwiGLU sum. This case failed before
    hardening at R2 `< 0.995`; it passed after the 4+ atom fit path switched to
    float64 fitting, residual-start search, and slower fit settings
    (`fitLearningRate = 0.01`, `fitSteps = 3200`). Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-4atom-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `4`, atomCount `4`, raw
    oracle calls `79808`, validation R2 `0.9996964505713657`, validation MAE
    `0.0001659790529608388`, max worst error `0.002872809058861975`, subspace
    min principal cosine `0.9999999945348134`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized 4-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-4unit-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles; each oracle returns the sum of four same-layer SwiGLU unit
    outputs. The solver receives only batch `f(x)` outputs, recovers rank-8
    active subspaces, and fits explicit 4-atom SwiGLU sums. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-4unit-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `4`, max atomCount `4`,
    total raw oracle calls `159616`, max raw oracle calls per case `79808`, min
    validation R2 `0.995727068595925`, max validation MAE
    `0.0006359374939938832`, max worst error `0.008566143276326953`,
    subspace min principal cosine `0.9999999725417702`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current first 5-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-5unit`.
    The gate uses a real local Mistral-family scalar oracle that returns the
    sum of five layer-20 SwiGLU unit outputs, units `[0,17,123,456,789]`. The
    solver receives only batch `f(x)` outputs, recovers a rank-10 active
    subspace, and fits an explicit 5-atom SwiGLU sum. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-5atom-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `5`, atomCount `5`, raw
    oracle calls `97648`, validation R2 `0.9970749283083804`, validation MAE
    `0.0006105457047962797`, max worst error `0.008950286237037718`, subspace
    min principal cosine `0.9999999795104269`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized 5-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-5unit-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles; each oracle returns the sum of five same-layer SwiGLU unit
    outputs. The solver receives only batch `f(x)` outputs, recovers rank-10
    active subspaces, and fits explicit 5-atom SwiGLU sums. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-5unit-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `5`, max atomCount `5`,
    total raw oracle calls `195296`, max raw oracle calls per case `97648`, min
    validation R2 `0.9951436177161573`, max validation MAE
    `0.0008022104114991162`, max worst error `0.010051115148071464`,
    subspace min principal cosine `0.9999997906113538`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current first 6-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-6unit`.
    The gate uses a real local Mistral-family scalar oracle that returns the
    sum of six layer-20 SwiGLU unit outputs, units `[0,17,123,456,789,1024]`.
    The solver receives only batch `f(x)` outputs, recovers a rank-12 active
    subspace, and fits an explicit 6-atom SwiGLU sum. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-6atom-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `6`, atomCount `6`, raw
    oracle calls `115488`, validation R2 `0.9970869523285548`, validation MAE
    `0.0007282176910088243`, max worst error `0.008779981374470364`, subspace
    min principal cosine `0.9999999544565973`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget 6-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-6unit-forward-coordinate`.
    This keeps the same real local Mistral-family scalar oracle and the same
    six units, but uses forward-coordinate active-subspace witnesses instead
    of central-coordinate witnesses. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-6atom-forward-coordinate-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `6`, atomCount `6`, raw
    oracle calls `62253`, validation R2 `0.9970121491993152`, validation MAE
    `0.0007410395113780754`, max worst error `0.00931293073078479`, subspace
    min principal cosine `0.9999994812351087`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized 6-unit block gate: DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-6unit-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles; each oracle returns the sum of six same-layer SwiGLU unit outputs.
    The solver receives only batch `f(x)` outputs, recovers rank-12 active
    subspaces, and fits explicit 6-atom SwiGLU sums. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-6unit-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `6`, max atomCount `6`,
    total raw oracle calls `230976`, max raw oracle calls per case `115488`,
    min validation R2 `0.9952423306790068`, max validation MAE
    `0.0009230608551877102`, max worst error `0.012063083889497478`,
    subspace min principal cosine `0.9999999518366974`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget randomized 6-unit block gate: DONE as a targeted
    benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-6unit-forward-coordinate-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles and forward-coordinate active-subspace witnesses. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-6unit-forward-coordinate-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `6`, max atomCount `6`,
    total raw oracle calls `124506`, max raw oracle calls per case `62253`,
    min validation R2 `0.9952220102880769`, max validation MAE
    `0.0009253719069818783`, max worst error `0.012323574241066268`,
    subspace min principal cosine `0.9999991210977408`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget selected 8-unit block gate: DONE as a targeted
    benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-8unit-forward-coordinate`.
    The gate uses a real local Mistral-family scalar oracle that returns the
    sum of eight layer-20 SwiGLU unit outputs, units
    `[0,17,123,456,789,1024,2048,4096]`. The solver receives only batch `f(x)`
    outputs, recovers a rank-16 active subspace, and fits an explicit 8-atom
    SwiGLU sum using forward-coordinate witnesses and explicit fit restart
    budgeting. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-budgeted-oracle-only/report.json`.
    Result: pass `true`, dimension `4096`, unitCount `8`, atomCount `8`, raw
    oracle calls `81297`, validation R2 `0.9956625486864737`, validation MAE
    `0.0010698511933124055`, max worst error `0.010449555792974501`,
    subspace min principal cosine `0.9999966163799717`, fit restarts `160`,
    and access audit `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget randomized 8-unit block gate: DONE as a targeted
    benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:multi-8unit-forward-coordinate-randomized`.
    The gate uses two deterministic-random real local Mistral-family scalar
    oracles; each oracle returns the sum of eight same-layer SwiGLU unit
    outputs. The solver receives only batch `f(x)` outputs, recovers rank-16
    active subspaces, and fits explicit 8-atom SwiGLU sums using
    forward-coordinate witnesses and explicit fit restart budgeting. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `8`, max atomCount `8`,
    total raw oracle calls `162594`, max raw oracle calls per case `81297`,
    min validation R2 `0.9965490748792153`, max validation MAE
    `0.0008981839822401561`, max worst error `0.010140033299851102`,
    subspace min principal cosine `0.9999975326762489`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current LLaMA-family 6-unit parity block gate: DONE as a targeted benchmark
    with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-multi-6unit`.
    The gate uses a real local LLaMA-family scalar oracle that returns the sum
    of six layer-20 SwiGLU unit outputs, units `[0,17,123,456,789,1024]`.
    The solver receives only batch `f(x)` outputs, recovers a rank-12 active
    subspace, and fits an explicit 6-atom SwiGLU sum. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-6atom-oracle-only/report.json`.
    Result: pass `true`, dimension `3072`, unitCount `6`, atomCount `6`, raw
    oracle calls `88864`, validation R2 `0.9999999971996777`, validation MAE
    `2.8572995724253928e-05`, max worst error `0.00021262079246214816`,
    subspace min principal cosine `0.9999999938218761`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget LLaMA-family 6-unit parity block gate: DONE as a
    targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-multi-6unit-forward-coordinate`.
    This keeps the same real local LLaMA-family scalar oracle and six units,
    but uses forward-coordinate active-subspace witnesses. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-6atom-forward-coordinate-oracle-only/report.json`.
    Result: pass `true`, dimension `3072`, unitCount `6`, atomCount `6`, raw
    oracle calls `48941`, validation R2 `0.9999975420086665`, validation MAE
    `0.0008233366167048662`, max worst error `0.006377951229655121`,
    subspace min principal cosine `0.9999960757027645`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current randomized LLaMA-family 6-unit parity block gate: DONE as a
    targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-multi-6unit-randomized`.
    The gate uses two deterministic-random real local LLaMA-family scalar
    oracles; each oracle returns the sum of six same-layer SwiGLU unit outputs.
    The solver receives only batch `f(x)` outputs, recovers rank-12 active
    subspaces, and fits explicit 6-atom SwiGLU sums. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-6unit-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `6`, max atomCount `6`,
    total raw oracle calls `177728`, max raw oracle calls per case `88864`, min
    validation R2 `0.9999999419380987`, max validation MAE
    `0.00008572753027982732`, max worst error `0.0008619591998977327`,
    subspace min principal cosine `0.9999998049413197`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget randomized LLaMA-family 6-unit parity block gate:
    DONE as a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-multi-6unit-forward-coordinate-randomized`.
    The gate uses two deterministic-random real local LLaMA-family scalar
    oracles and forward-coordinate active-subspace witnesses. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-6unit-forward-coordinate-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `6`, max atomCount `6`,
    total raw oracle calls `97882`, max raw oracle calls per case `48941`, min
    validation R2 `0.9999880522847288`, max validation MAE
    `0.0012813828353731324`, max worst error `0.012340230224598514`,
    subspace min principal cosine `0.9999574008675185`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget LLaMA-family selected 8-unit block gate: DONE as a
    targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-multi-8unit-forward-coordinate`.
    The gate uses a real local LLaMA-family scalar oracle that returns the sum
    of eight layer-20 SwiGLU unit outputs, units
    `[0,17,123,456,789,1024,2048,4096]`. The solver receives only batch `f(x)`
    outputs, uses raw forward-coordinate active-subspace witnesses, active
    oracle-gradient fit witnesses, and a batched CUDA multi-start explicit
    8-atom SwiGLU fitter. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-budgeted-oracle-only/report.json`.
    Result: pass `true`, dimension `3072`, unitCount `8`, atomCount `8`, raw
    oracle calls `68241`, validation R2 `0.9999913818259203`, validation MAE
    `0.0017664391250281214`, max worst error `0.016726727075836223`,
    subspace min principal cosine `0.9999919539106246`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current query-budget randomized LLaMA-family 8-unit block gate: DONE as a
    targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-oracle-decompiler:llama-multi-8unit-forward-coordinate-randomized`.
    The gate uses two deterministic-random real local LLaMA-family scalar
    oracles; each oracle returns the sum of eight same-layer SwiGLU unit
    outputs. The solver receives only batch `f(x)` outputs, uses raw
    forward-coordinate active-subspace witnesses, active oracle-gradient fit
    witnesses, and batched CUDA multi-start explicit 8-atom SwiGLU fitting.
    Artifact:
    `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, max unitCount `8`, max atomCount `8`,
    total raw oracle calls `136482`, max raw oracle calls per case `68241`,
    min validation R2 `0.9999781080257685`, max validation MAE
    `0.002327151614302204`, max worst error `0.02565795362142126`,
    subspace min principal cosine `0.999936352892748`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current Mistral-family 16-unit layer-expansion gate: DONE as a targeted
    benchmark with `npm run benchmark:nn-live-blind-swiglu-layer-expansion`.
    The default runner now starts from a cheaper 16-unit probe profile and
    relies on the oracle decompiler's adaptive-rescue ledger when that probe
    misses gate quality, instead of manually starting every default run at the
    full high-budget profile. Artifact:
    `benchmarks/release/nn-live-blind-swiglu-layer-expansion/mistral-layer20-16unit-expansion-oracle-only/report.json`.
    Result: pass `true`, model family Mistral, dimension `4096`, layer `20`,
    unitCount `16`, active rank `32`, raw oracle calls `188193`, validation R2
    `0.9969881717700748`, validation MAE `0.0013780126849811608`, worst error
    `0.009548484562662135`, subspace min principal cosine
    `0.9999979444870536`, and access audit `solverAccess = oracle-only`,
    `weightsUsedBySolver = false`, `hooksUsedBySolver = false`,
    `postHocWitnessOnly = true`. The per-case ledger records initial probe R2
    `0.9932383815355237`, then adaptive-rescue R2
    `0.9969881717700748`.
  - Current randomized Mistral-family 16-unit layer-expansion gate: DONE as a
    targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-layer-expansion:mistral-16unit-randomized`.
    Artifact:
    `benchmarks/release/nn-live-blind-swiglu-layer-expansion/mistral-4096d-randomized-2block-16unit-layer-expansion-oracle-only/report.json`.
    Result: pass `true`, cases `2/2`, real Mistral-family dimension `4096`,
    layers `1` and `11`, unitCount `16`, active rank `32`, total raw oracle
    calls `376386`, min validation R2 `0.9968086229936512`, max validation MAE
    `0.0013585229495551043`, max worst error `0.018289335513027943`, subspace
    min principal cosine `0.999989791877386`, and access audit
    `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
    `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
  - Current selected LLaMA-family 16-unit layer-expansion parity gate: DONE as
    a targeted benchmark with
    `npm run benchmark:nn-live-blind-swiglu-layer-expansion:llama-16unit`.
    Artifact:
    `benchmarks/release/nn-live-blind-swiglu-layer-expansion/llama-layer20-16unit-expansion-oracle-only/report.json`.
    Result: pass `true`, model family LLaMA, dimension `3072`, layer `20`,
    unitCount `16`, active rank `32`, raw oracle calls `129953`, validation R2
    `0.996028807883924`, validation MAE `0.013147489844684133`, worst error
    `0.1197387336033246`, subspace min principal cosine
    `0.9996660748268255`, and access audit `solverAccess = oracle-only`,
    `weightsUsedBySolver = false`, `hooksUsedBySolver = false`,
    `postHocWitnessOnly = true`. The runner now uses LLaMA high-atom defaults
    for this profile instead of the earlier Mistral-shaped scale/profile
    defaults; thresholds were not lowered.
  - Current randomized LLaMA-family 16-unit function-and-structure gate: DONE
    with
    `npm run benchmark:nn-live-blind-swiglu-layer-expansion:llama-16unit-residual-block-randomized`.
    Artifact:
    `benchmarks/release/nn-live-blind-swiglu-layer-expansion/llama-3072d-randomized-2block-16unit-structure-gated-residual-block-oracle-only/report.json`.
    Result of the 2026-08-02 regeneration: pass `true`, cases `2/2`, total raw
    oracle calls `461398`, min R2 `0.9999855916796467`, max MAE
    `0.0007727211891917342`, max worst error `0.006249561164756855`, min
    subspace principal cosine `0.9999530973154566`, min exact one-to-one paired
    gate/up joint cosine `0.9999610978737026`, and all `16/16` atoms above
    `0.99` in both blocks. No threshold was moved.

    What closed it. The failing block was not a weak-atom problem, a collapsed
    pair, or a restart-selection problem: all three were tested and measured
    dead. Units 3627 and 3679 are `0.0095` apart in gate direction, and the
    nonlinear fit had recovered them as two copies of their bisector - the
    projections of atoms 12 and 14 onto the sources' span had cosine `0.999838`,
    both at 41 degrees - while the surface fit held at R2 `0.9986`. No amount of
    further optimisation finds that, because the output loss has nothing to say
    about which factorisation produced it.

    A gate direction is not something to optimise for; it is where the
    function's second derivative breaks, and that is measurable with oracle
    calls. Running the existing asymptotic gate stitch as a check after the
    block refit, rather than instead of one, replaces the guessed directions
    with the boundaries actually found. Layer 15 went from `14/16` atoms and a
    minimum joint cosine of `0.2386` to `16/16` at `0.99996`. Layer 1, which
    already passed, went from worst error `0.0638` to `0.0049` and R2
    `0.998568` to `0.999987`, so the gain is a property of the method rather
    than a repair aimed at one block. Cost is `461398` oracle calls against
    `296770`, inside the `300000` per-case budget.

  - Current cross-family finite-precision 16-unit asymptotic-structure court:
    DONE with
    `npm run benchmark:nn-live-blind-swiglu-asymptotic-structure`.
    The court regenerates three snapshot-backed real-model cases: Mistral
    4096D layers `9` and `15` at output channel `2048`, plus LLaMA-family
    3072D layer `9`; every scalar block contains `16` SwiGLU atoms and every
    oracle response is quantized at `1e-6`. Asymptotic line boundaries recover
    gate normals, Richardson gradient/Hessian witnesses recover up directions,
    source-blind support fallback protects weak clusters, and bounded
    variable-projection trust-region refinement completes the structure.
    Artifact:
    `benchmarks/release/nn-live-blind-swiglu-asymptotic-structure/mistral4096d-llama3072d-16unit-q000001-oracle-only/report.json`.
    Fresh regeneration result: pass `true`, cases `3/3`, min R2
    `0.999647819803404`, max MAE `0.0006021497608951118`, max worst error
    `0.003997934982926221`, min subspace principal cosine
    `0.9996985570973159`, min one-to-one paired gate/up joint cosine
    `0.9989749892979034`, all `16/16` paired atoms above `0.99` in every
    case, max raw oracle calls per case `109808`, total raw oracle calls
    `319042`, and access invariant `true`. A separate `:audit` command writes
    to an audit-only variant so it cannot overwrite the regeneration witness.
  - Current fresh no-snapshot finite-precision 16-unit cold-start gate: DONE
    with `npm run benchmark:nn-live-blind-swiglu-asymptotic-cold-start`.
    This gate starts from a newly loaded real quantized Mistral layer-15 scalar
    oracle, rebuilds the 4096D rank-32 active subspace, and then reconstructs
    all 16 explicit SwiGLU gate/up atoms through asymptotic line stitching and
    finite-precision value calibration. Fit snapshots and nonlinear initial
    atom fitting are forbidden by preflight and audited after the run. Bounded
    nonlinear calibration/refinement still runs after structure recovery.
    Artifact:
    `benchmarks/release/nn-live-blind-swiglu-asymptotic-cold-start/mistral4096d-layer15-16unit-q000001-fresh-oracle-only/report.json`.
    Fresh result: pass `true`, `regenerated = true`, R2
    `0.9995787659698843`, MAE `0.00000616945597616291`, worst error
    `0.000032305729901930614`, subspace minimum principal cosine
    `0.9996985570973159`, minimum one-to-one paired gate/up joint cosine
    `0.9972788963034496`, all `16/16` pairs above `0.99`, raw oracle calls
    `1279238` of `1500000`, `nonlinearInitialFitUsed = false`,
    `fitSnapshotUsedBySolver = false`, and failures `[]`.
  - Current process-separated opaque-oracle cold-start gate: DONE with
    `npm run benchmark:nn-live-blind-swiglu-isolated-cold-start`. The teacher
    manifest is delivered only to the oracle server through a private inherited
    file descriptor. The dedicated solver argv contains only endpoint,
    dimension, atom count, budget, and output paths; its environment contains no
    model-cache or teacher-manifest variables. It sees an eval-only binary Unix
    protocol with no path fields or truth capability. A separate verifier starts
    after solver exit and checks the immutable frozen recovery. Fresh artifact:
    `benchmarks/release/nn-live-blind-swiglu-isolated-cold-start/mistral4096d-layer15-16unit-q000001-opaque-oracle/report.json`.
    Run `926693cf03df48e7a26e8466b8b83b11` passed with R2
    `0.9995787659698843`, MAE `0.00000616945597616291`, worst error
    `0.000032305729901930614`, all `16/16` pairs above `0.99`, and
    `1279238/1500000` oracle calls. Solver and oracle transcript SHA-256 values
    match; the frozen SHA-256 is unchanged across verification; the hash-stamped
    `3098`-line `strace` witness records no model/weight or orchestrator/oracle/
    verifier control-file access by the solver. This artifact was regenerated on
    2026-07-27 against the current runtime source after the earlier
    `3f82268a921048e7bc21c333d3ecb5e8` artifact went stale relative to the
    orchestrator and oracle-server hashes. The regeneration reproduced the same
    R2, MAE, worst error, and oracle-call count bit-for-bit under the fixed seed,
    so the lane is deterministic across runs as well as gated.
  - Boundary: this proves selected real Mistral-family SwiGLU units can be
    recovered as blind oracle-only explicit SwiGLU atoms under the recorded
    budgets, plus selected real LLaMA-family 16-unit parity under its recorded
    budget. The current evidence includes first selected and first randomized
    2-unit scalar
    blocks, first selected and randomized 3-unit scalar blocks, and a first
    selected and randomized 4-unit scalar block plus first selected and
    randomized 5-unit scalar blocks plus first selected and randomized 6-unit
    scalar blocks, plus first selected/randomized LLaMA-family 6-unit scalar
    blocks and forward-coordinate query-budget parity for selected/randomized
    Mistral/LLaMA 6-unit blocks plus selected/randomized Mistral 8-unit blocks
    and selected/randomized LLaMA-family 8-unit blocks, plus a first
    self-rescuing Mistral-family 16-unit layer-expansion default runner, a
    first deterministic-random 2-case Mistral-family 16-unit layer-expansion
    sweep, a first selected LLaMA-family 16-unit layer-expansion parity gate,
    and a first deterministic-random 2-case LLaMA-family 16-unit gate with
    individual atom-structure evidence.
    It does not prove full Mistral/LLaMA FFN extraction,
    all-channel 3072D/4096D block decompilation, exhaustive circuit recovery,
    arbitrary high-count mixtures, or semantic original-neuron identity from
    output-only access.
  - Next K5 hardening:
    1. widen the process-separated finite-precision cold-start court across randomized early,
       middle, and late layers in both model families. DONE AS A GATE, NEGATIVE
       AS A RESULT. The cross-family court
       (`benchmark:nn-live-blind-swiglu-isolated-cold-start-court`) covers both
       families and both oracle modes at fixed layers Mistral `15` and LLaMA
       `9`, passing `2/2`. The randomized court
       (`benchmark:nn-live-blind-swiglu-randomized-cold-start-court`, master
       seed `260729`) removes the hand-picking and passes only `1/6`:

       | case | R2 | worst error | cause |
       | --- | ---: | ---: | --- |
       | mistral layer `9`, output-channel | n/a | n/a | structure-not-recovered |
       | mistral layer `11`, unit-sum | `0.9999999194986897` | `9.755e-05` | PASS |
       | mistral layer `28`, output-channel | `0.978313682106669` | `1.4947e-04` | recovered-but-low-r2 |
       | llama layer `1`, output-channel | `0.9899565033743015` | `0.010985830786309943` | recovered-but-inaccurate |
       | llama layer `10`, unit-sum | `0.9999991253174392` | `0.005582131817034153` | recovered-but-worst-error-over-bar |
       | llama layer `23`, output-channel | `0.9948086207127055` | `0.01800143558433407` | recovered-but-inaccurate |

       Conclusion: the opaque cold-start recovery is family-independent but it is
       NOT selection-independent. The failures split into three different
       problems, and they need different work:
       - `structure-not-recovered` (1 case): the asymptotic gate stitch rejects
         the structure before any fit exists. This is the hardest case and the
         one that most limits the claim.
       - `recovered-but-inaccurate` (2 cases): a structure is recovered but
         held-out worst error is 2-4x over the bar. This looks like a fit-quality
         and probing-budget problem, not an identifiability wall.
       - `recovered-but-low-r2` (1 case): R2 `0.978` with worst error `1.49e-04`.
         An earlier revision of this document claimed the low R2 was a metric
         artifact on a low-variance block. Direct measurement disproves that. The
         block's output standard deviation over the validation distribution is
         `1.453196e-04`, against `3.434522e-02` for the passing Mistral layer-11
         block, a `237x` difference. The worst error therefore equals roughly the
         entire signal amplitude, so R2 `0.978` is an honest verdict: the recovery
         genuinely has near-100% relative error at its worst points. The metric is
         not at fault and must not be replaced.

         The measured cause is the solver's absolute input scale. Holding
         everything else fixed and raising `--input-scale` from `0.45` to `0.9`
         moves this block from R2 `0.978313682106669` / worst
         `1.4947e-04` to R2 `0.9937954126258921` / worst `6.720740704713169e-05`.
         Raising it further to `1.4` destroys recovery entirely with
         `structure-court-rejected`, so there is an optimum, not a monotone gain.
         Stronger structure refinement is not the lever: `600` refine steps with
         `5` restarts and `400` gate-norm calibration steps move R2 only from
         `0.993795` to `0.993873`.

         A single global constant cannot be the fix. The same `--input-scale`
         `0.9` that helps this block destroys the passing Mistral layer-11 case,
         taking it from R2 `0.9999999194986897` / worst `9.755e-05` to R2
         `0.9505607375632952` / worst `0.06374726936482406`. The optimal input
         scale is block-dependent.

         Implemented as `calibrate_input_scale` plus `--auto-input-scale`, enabled
         in the solver worker. Candidate scales are walked from smallest to
         largest and the first whose measured output std clears `1000x` the
         oracle's own quantization step is taken; if none clears it, the largest
         candidate is used, and that list doubles as the structure-safety cap.
         The probe uses oracle evaluations only and fits nothing, so the
         blind-access invariant holds; it costs one `512`-call batch per candidate
         tried.

         Effect, measured: mistral layer `28` selects `0.9` and moves from R2
         `0.978313682106669` to `0.9937954126258921`, worst error `1.4947e-04` to
         `6.720740704713169e-05`. Mistral layer `11` selects `0.45`, reproduces
         bit-identically, and still passes. One further distinct hand-picked
         block, mistral layer `15`, also selects `0.9` and continues to pass, so
         the rule is not tuned to the single block it was derived from.

         Correction: an earlier revision of this paragraph said "two of the three
         hand-picked cold-start blocks". There are two distinct hand-picked
         blocks, not three. The single-case isolated lane and the cross-family
         court's `mistral-4096d-layer15-output2048` case are the same block, the
         same 16 units, the same output channel `2048` and the same seed
         `260723`, and their artifacts agree to 16 significant digits. Counting
         it twice overstated the weight of evidence. The substance survives,
         because layer `15` is still a different block from the layer `28` the
         rule was derived from, but the count was wrong.

         Not yet sufficient: layer `28` remains under the `0.999` bar, so this
         repairs the amplitude class without closing it. The three LLaMA cases
         select `0.45`, which means their failures are not amplitude-driven and
         need a separate lever.
       - `recovered-but-worst-error-over-bar` (1 case): worst error
         `0.00558` against a `0.005` bar with R2 `0.9999991`. A marginal miss.

       Thresholds must not be widened to convert these into passes;
    2. reduce line/Hessian query cost with adaptive support allocation before
       increasing the explicit atom count beyond `16`;
    3. connect recovered scalar block structures into output-channel and
       cross-layer composition courts instead of treating each scalar block
       as an isolated endpoint.
- Stage L: `LLMOracleAdapter` control plane.
  - Refactor the current live-reader/fact-finder implementation into a reusable
    oracle interface with typed request/response records for model identity,
    tokenizer identity, prompts, token positions, layers, components,
    activation families, logits, activations, interventions, ablations, control
    families, cache ids, runtime warnings, and budget accounting.
  - `GPT2Oracle` remains adapter implementation #1 and fast proof gate, not the
    framework contract.
  - Runtime failures, import warnings, hook mismatches, tokenizer mismatches,
    non-disjoint controls, and empty-control families must be hard failures or
    visible artifact errors.
  - The oracle must support both MLP activation ablation and attention
    head/value-path intervention without changing the circuit-finder API.

The following items are still required before any broad SOTA language:

- Wider randomized and adversarial 3D sweep, beyond the current smoke,
  non-degenerate randomized, and two-teacher cancellation hard-case gates.
  Status: started with `benchmark:nn-noisy-blackbox-relu3d-adversarial`;
  broaden further with generated/adaptive adversarial search before using broad
  SOTA language.
- True general-ND solver, not only 3D. Status: started with the 4D line-probe
  gate and broadened to deterministic clean/noisy 4D/5D seed sweeps plus a
  generated adversarial noisy 4D/5D profile sweep and an adaptive
  search-generated noisy 4D/5D adversary lane.
- Random-direction probing instead of relying on z-slices and surface
  refinement only. Status: started with `benchmark:nn-blackbox-relu-nd-line-probe`.
- Deep ReLU composition recovery, not only shallow hidden-layer recovery.
  Status: started with `benchmark:nn-deep-blackbox-relu1d-decompile` and
  broadened to `benchmark:nn-deep-blackbox-relu2d-region-graph`, which recovers
  an output-equivalent affine-region adjacency graph for a folded two-layer 2D
  ReLU black box.
- Query-budget benchmark: total oracle calls, repeated calls per point, and
  accuracy versus query cost. Status: started with
  `benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph` and
  broadened to `benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph`;
  further broadened to
  `benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph`,
  which records rejected too-coarse candidate grids before promotion, and to
  `benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph`, which
  requires zero fallback topology segments and endpoint residual below
  `1e-9`. Next broaden this boundary discipline beyond 2D before broad SOTA
  language.
- Comparison against strong methods for piecewise-linear model extraction, with
  clear baselines and failure accounting.
- Smooth activation decompilation for GELU/SwiGLU LLM-style FFN channels, with
  separate gates from the ReLU recovered-adjacency geometry lane. Status:
  started with `benchmark:nn-active-subspace-ffn-decompiler` and extended to a
  controlled 4096D stress gate with
  `benchmark:nn-active-subspace-ffn-4096d-stress`; the noisy rank-1 failure
  mode is now covered by `benchmark:nn-active-subspace-ffn-noisy`; the first
  clean rank-adaptive mixed GELU structure gate is covered by
  `benchmark:nn-gelu-ffn-mixed-structure`; near-parallel clean mixed-channel
  stability is covered by `benchmark:nn-gelu-ffn-near-parallel-family`;
  clean canonical structure invariance under equivalent hidden decompositions is
  covered by `benchmark:nn-gelu-ffn-cross-decomposition-invariance`; and the
  first formalization negative-case enforcement gate is covered by
  `benchmark:nn-gelu-ffn-negative-cases`; controlled noisy/quantized mixed
  GELU structure recovery is covered by
  `benchmark:nn-gelu-ffn-noisy-mixed-structure`, which now includes
  IID/OOD distribution checks plus scikit-learn, XGBoost, and sklearn-MLP
  package-backed prediction-sanity baselines; the first clean mixed SwiGLU
  value/gate pair structure gate is covered by
  `benchmark:nn-swiglu-ffn-mixed-structure`; controlled noisy/quantized mixed
  SwiGLU value/gate recovery is covered by
  `benchmark:nn-swiglu-ffn-noisy-mixed-structure`; controlled shared-gate/
  cancellation-heavy SwiGLU canonicalization is covered by
  `benchmark:nn-swiglu-ffn-shared-gate-canonical`; and the first local
  transformer MLP hook is covered by
  `benchmark:nn-local-transformer-mlp-hook`. The compact real GPT-2 top-5 and
  top-10 blind source-structure gates are now covered by
  `benchmark:nn-gpt2-gelu-ffn-real-structure:top5` and
  `benchmark:nn-gpt2-gelu-ffn-real-structure:top10`; the compact real GPT-2
  top-8 report is retained as the intermediate targeted scaling proof. The
  compact real GPT-2 top-20 source extractor remains a separate instrumented
  source-graph gate. The first live GPT-2 MLP fact-circuit graph finder is now
  covered by `benchmark:nn-live-gpt2-fact-circuit-finder`, now with Stage-I
  paraphrase, wrong-answer, negative-fact, and unrelated-logit hardening. Next
  evidence must add attention/path edges before any broad GPT-2/LLM
  decompilation claim.

## Current Frontier Target

The current frontier target is Active-Subspace Neural Decompilation plus a
model-agnostic LLM oracle adapter framework. The release-gated ReLU
recovered-geometry lanes remain the oracle/probing discipline floor, the
GELU/SwiGLU active-subspace lanes are the bridge toward LLM-style FFN channels,
and the live GPT-2 fact-circuit graph finder is the first fast real-model
adapter gate. Pythia/GPT-Neo are the next GELU-family portability gates;
LLaMA/Mistral are the SwiGLU-family portability gates.

The adaptive adversarial noisy 4D/5D ND shallow-ReLU lane and the adaptive
adversarial noisy folded-2D deep-ReLU lane are now release-gated robustness
floors. GCL now has release-gated lower-budget deep-2D selection, boundary-
refined local switching, subcell-boundary adaptive validation with an explicit
rejected-candidate ledger, topology-preserving recovered-adjacency boundary
geometry, noisy/adaptive adversarial deep-ND folded recovery, deep-ND query-
budget reduction, and recovered-adjacency geometry beyond 2D. The next
capability jump is no longer simply 7D/8D ReLU scaling. The next move is active-
subspace smooth FFN recovery: 256D/1024D controlled GELU/SwiGLU channels, 4096D
stress, noisy 128D/256D rank-1 GELU recovery, explicit GELU atom recovery, the
first clean rank-adaptive mixed GELU block, the near-parallel epsilon family,
cross-decomposition canonical invariance, and the first formalization
negative-case gate including both weak-under-noise pruning and weak-above-noise
exact recovery, plus controlled noisy/quantized mixed GELU structure recovery,
are now release-gated, including IID/OOD distribution checks plus scikit-learn,
XGBoost, and sklearn-MLP package-backed prediction-sanity court for the
  controlled noisy mixed GELU lane. The first controlled clean/noisy mixed
  SwiGLU value/gate structure gates, the shared-gate/cancellation-heavy SwiGLU
  canonical gate, source-neuron provenance verifier, and local transformer MLP
  hook gate, compact real GPT-2 top-3/top-5/top-10 blind source-structure gates,
  and compact real GPT-2 top-20 instrumented source-extractor gate are now
  release-gated. Next is scaling and hardening blind smooth-FFN structure, not
  another isolated surrogate or an instrumented-only claim.
The 6D ReLU lane remains a foundation and is not a broad SOTA claim by itself.

Target commands:

- `npm run benchmark:nn-deep-blackbox-relu-nd-region-sampler`
- `npm run benchmark:nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler`
- `npm run benchmark:nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler`
- `npm run benchmark:nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler`
- `npm run benchmark:nn-nd-adjacency-geometry-5d-sweep`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-sweep`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-adaptive-probe`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-baselines`
- `npm run benchmark:nn-nd-adjacency-geometry-6d-query-budget`
- `npm run benchmark:nn-active-subspace-ffn-decompiler`
- `npm run benchmark:nn-active-subspace-ffn-4096d-stress`
- `npm run benchmark:nn-active-subspace-ffn-noisy`
- `npm run benchmark:nn-gelu-ffn-structure-finder`
- `npm run benchmark:nn-gelu-ffn-mixed-structure`
- `npm run benchmark:nn-gelu-ffn-near-parallel-family`
- `npm run benchmark:nn-gelu-ffn-cross-decomposition-invariance`
- `npm run benchmark:nn-gelu-ffn-negative-cases`
- `npm run benchmark:nn-gelu-ffn-noisy-mixed-structure`
- `npm run benchmark:nn-swiglu-ffn-mixed-structure`
- `npm run benchmark:nn-swiglu-ffn-noisy-mixed-structure`
- `npm run benchmark:nn-swiglu-ffn-shared-gate-canonical`
- `npm run benchmark:nn-gelu-ffn-rank-cap-negative`
- `npm run benchmark:nn-gelu-ffn-provenance-verifier`
- `npm run benchmark:nn-local-transformer-mlp-hook`
- `npm run benchmark:nn-gpt2-gelu-ffn-real-structure`
- `npm run benchmark:nn-gpt2-gelu-ffn-real-structure:top5`
- `npm run benchmark:nn-gpt2-gelu-ffn-source-extractor`
- `npm run benchmark:nn-live-gpt2-fact-circuit-finder`
- `npm run research:nn-gpt2-gelu-ffn-structure-finder`
- `npm run experiment:nn-nd-adjacency-geometry-5d`
- `npm run experiment:nn-nd-adjacency-geometry-6d`
- `npm run benchmark:nn-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-deep-blackbox-relu2d-region-graph:randomized`
- `npm run benchmark:nn-noisy-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-compact-deep-blackbox-relu2d-boundary`
- `npm run benchmark:nn-adaptive-budget-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-boundary-refined-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph`
- `npm run benchmark:nn-topology-boundary-deep-blackbox-relu2d-region-graph`

Current status:

- Direct benchmark PASS on 2026-05-25.
- Teacher: 2-input / 4-hidden / 4-hidden / 1-output ReLU MLP.
- GCL access: oracle calls only, `f(x,y)`.
- Recovered affine regions: 21.
- Region adjacency edges: 46.
- Kink-cell boundaries: 964.
- GCL MAE: `1.485e-4`.
- GCL R2: `0.999999399`.
- Plane baseline MAE: `0.830813`.
- First-layer-only baseline MAE: `0.230876`.
- Raw oracle calls: `51005`.
- Artifact:
  `benchmarks/release/nn-deep-blackbox-relu2d-region-graph/folded-two-layer/report.json`
- The lane passes inside `npm run release:gate` on 2026-05-25.
- Randomized seed-sweep benchmark PASS on 2026-05-25.
- Randomized teachers: 4 deterministic 2-input / 4-hidden / 4-hidden /
  1-output ReLU MLPs.
- Randomized passed teachers: 4/4.
- Randomized minimum recovered regions: 20.
- Randomized minimum deep-only region estimate: 15.
- Randomized minimum adjacency edges: 40.
- Randomized minimum kink-cell boundaries: 839.
- Randomized max MAE: `1.890e-4`.
- Randomized min R2: `0.999996583`.
- Randomized total raw oracle calls: `204020`.
- Randomized artifact:
  `benchmarks/release/nn-deep-blackbox-relu2d-region-graph-randomized/seed-sweep/report.json`
- Noisy/quantized seed-sweep benchmark PASS on 2026-05-25.
- Noisy teachers: 2 deterministic 2-input / 4-hidden / 4-hidden / 1-output
  ReLU MLPs.
- Noisy oracle: Gaussian output noise `0.003`, quantization `0.001`,
  `21` repeated calls per probe point, median aggregation.
- Noisy passed teachers: 2/2.
- Noisy minimum recovered regions: 125.
- Noisy minimum deep-only region estimate: 120.
- Noisy minimum adjacency edges: 304.
- Noisy minimum kink-cell boundaries: 2061.
- Noisy max MAE: `0.002395`.
- Noisy min R2: `0.99994584`.
- Noisy total raw oracle calls: `1377810`.
- Noisy artifact:
  `benchmarks/release/nn-noisy-deep-blackbox-relu2d-region-graph/seed-sweep/report.json`
- Adaptive-adversarial benchmark PASS on 2026-05-25 after a failing first
  attempt exposed near-parallel and weak-signal hard cases.
- Adaptive-adversarial teachers: 12 generated 2-input / 4-hidden / 4-hidden /
  1-output folded ReLU MLPs across near-parallel, offset-collision,
  cancellation, low-amplitude, boundary-grazing, and mixed-oblique profiles.
- Adaptive-adversarial passed final candidates: 12/12.
- Adaptive-adversarial selected hard cases passed: 4/4.
- Adaptive-adversarial minimum selected regions: 75.
- Adaptive-adversarial minimum selected deep-only region estimate: 70.
- Adaptive-adversarial minimum selected adjacency edges: 246.
- Adaptive-adversarial max selected MAE: `0.006062`.
- Adaptive-adversarial min selected R2: `0.99895766`.
- Adaptive-adversarial max selected first-layer MAE ratio: `0.126213`.
- Adaptive-adversarial total raw oracle calls: `9313740`.
- Adaptive-adversarial artifact:
  `benchmarks/release/nn-adaptive-adversarial-deep-blackbox-relu2d-region-graph/adaptive-search/report.json`
- Compact boundary benchmark PASS on 2026-05-25.
- Compact boundary teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Compact boundary passed teachers: 3/3.
- Compact boundary minimum compression ratio: `19.068182x` versus raw
  cell-edge boundary contacts.
- Compact boundary max fit RMSE: `0.075528`.
- Compact boundary max MAE: `1.775e-4`.
- Compact boundary min R2: `0.999996769`.
- Compact boundary total raw oracle calls: `153015`.
- Compact boundary artifact:
  `benchmarks/release/nn-compact-deep-blackbox-relu2d-boundary/fixed-plus-randomized/report.json`
- Adaptive-budget benchmark PASS on 2026-05-25.
- Adaptive-budget teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Adaptive-budget passed teachers: 3/3.
- Adaptive-budget selected grid widths: `89, 89, 89`.
- Adaptive-budget minimum oracle reduction versus fixed `101x101`: `20.102%`.
- Adaptive-budget total oracle calls: `122256` versus fixed `153015`.
- Adaptive-budget max MAE: `2.407e-4`.
- Adaptive-budget min R2: `0.999992859`.
- Adaptive-budget artifact:
  `benchmarks/release/nn-adaptive-budget-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Adaptive-budget boundary: this is a validated whole-grid budget selector
  with fallback, not the final form of local spatial refinement.
- Boundary-refined benchmark PASS on 2026-05-25.
- Boundary-refined teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Boundary-refined grid: `57x57` plus held-out validation.
- Boundary-refined passed teachers: 3/3.
- Boundary-refined minimum oracle reduction versus fixed `101x101`: `65.901%`.
- Boundary-refined total oracle calls: `52176` versus fixed `153015`.
- Boundary-refined max MAE: `7.994e-5`.
- Boundary-refined min R2: `0.999999013`.
- Boundary-refined minimum held-out coarse-to-refined MAE improvement:
  `7.996471x`.
- Boundary-refined artifact:
  `benchmarks/release/nn-boundary-refined-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Boundary-refined boundary: compact boundary segments are now an active local
  switching mechanism, but this still does not claim topology-perfect true-region
  merging or original hidden-layer identity.
- Subcell-boundary adaptive benchmark PASS on 2026-05-25.
- Subcell-boundary adaptive teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Subcell-boundary adaptive selected grid widths: `37, 37, 41`.
- Subcell-boundary adaptive rejected too-coarse attempts: 1.
- Subcell-boundary adaptive total oracle calls: `33528` versus fixed `153015`
  and boundary-refined 57x57 `52176`.
- Subcell-boundary adaptive total oracle reduction: `78.088%` versus fixed
  `101x101`, and `35.741%` versus boundary-refined 57x57.
- Subcell-boundary adaptive max MAE: `2.837e-4`.
- Subcell-boundary adaptive min R2: `0.999990210`.
- Subcell-boundary adaptive minimum held-out coarse-to-subcell MAE improvement:
  `3.550431x`.
- Subcell-boundary adaptive artifact:
  `benchmarks/release/nn-subcell-boundary-adaptive-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Topology-boundary benchmark PASS on 2026-05-26 after a first randomized
  attempt exposed fallback contact-corridor segments on seed `83`.
- Topology-boundary teachers: 3 fixed/randomized 2-input / 4-hidden /
  4-hidden / 1-output folded ReLU MLPs.
- Topology-boundary passed teachers: 3/3.
- Topology-boundary fallback segments: 0.
- Deep ND region-sampler benchmark PASS on 2026-05-26.
- Deep ND teachers: 4 deterministic 3D/4D folded two-layer ReLU MLPs.
- Deep ND visibility: oracle calls only, `f(vector)`, no hidden activations,
  layer widths, or teacher weights.
- Deep ND passed teachers: 4/4.
- Deep ND minimum recovered regions: `40`.
- Deep ND minimum deep-only region estimate: `35`.
- Deep ND max MAE: `0.011095250643373489`.
- Deep ND min R2: `0.998534063321796`.
- Deep ND min global-linear MAE improvement ratio: `42.81846369086524`.
- Deep ND total raw oracle calls: `315800`.
- Deep ND artifact:
  `benchmarks/release/nn-deep-blackbox-relu-nd-region-sampler/3d-4d-seed-sweep/report.json`
- Noisy adaptive-adversarial deep ND benchmark PASS on 2026-05-27.
- Noisy adaptive-adversarial deep ND teachers: 7 generated 3D/4D folded
  two-layer ReLU MLPs across near-parallel, cancellation, low-amplitude, and
  mixed-oblique hard profiles.
- Noisy adaptive-adversarial deep ND visibility: noisy/quantized oracle calls
  only, `f(vector)`, no hidden activations, layer widths, or teacher weights.
- Noisy adaptive-adversarial deep ND passed teachers: 7/7.
- Noisy adaptive-adversarial selected local-affine KNN attempts: `2`.
- Noisy adaptive-adversarial minimum recovered regions: `6`.
- Noisy adaptive-adversarial median deep-only region estimate: `25`.
- Noisy adaptive-adversarial max MAE: `0.026450213096249696`.
- Noisy adaptive-adversarial min R2: `0.9830270507435355`.
- Noisy adaptive-adversarial min global-linear MAE improvement ratio:
  `7.392219720751436`.
- Noisy adaptive-adversarial total selected raw oracle calls: `4274160`.
- Noisy adaptive-adversarial total adaptive-search raw oracle calls:
  `14132880`.
- Noisy adaptive-adversarial artifact:
  `benchmarks/release/nn-noisy-adaptive-adversarial-deep-blackbox-relu-nd-region-sampler/3d-4d-adaptive-search/report.json`
- Adaptive-budget noisy deep ND benchmark PASS on 2026-05-27.
- Adaptive-budget noisy deep ND teachers: same 7 generated 3D/4D folded
  two-layer ReLU MLP hard profiles as the noisy adaptive-adversarial gate.
- Adaptive-budget selection: sequential noisy holdout promotion over low/mid/high
  region-vote/local-affine candidates, with rejected candidates written into the
  report.
- Adaptive-budget passed teachers: 7/7.
- Adaptive-budget selected low/mid/high budgets: `6/0/1`.
- Adaptive-budget selected local-affine KNN attempts: `3`.
- Adaptive-budget rejected candidates recorded: `7`.
- Adaptive-budget max attempted candidates per teacher: `5`.
- Adaptive-budget max MAE: `0.03291800518211379`.
- Adaptive-budget min R2: `0.9865538047691716`.
- Adaptive-budget selected raw oracle calls: `1862760`.
- Adaptive-budget adaptive-search raw oracle calls: `3404120`.
- Adaptive-budget selected-call reduction vs previous noisy deep-ND release:
  `56.418%`.
- Adaptive-budget search-call reduction vs previous noisy adaptive-search cost:
  `75.913%`.
- Adaptive-budget artifact:
  `benchmarks/release/nn-adaptive-budget-noisy-deep-blackbox-relu-nd-region-sampler/3d-4d-budget-search/report.json`
- ND adjacency-geometry benchmark PASS on 2026-05-27.
- ND adjacency-geometry teachers: clean/noisy 3D/4D folded two-layer ReLU MLP
  profiles, with no hidden teacher state exposed to GCL.
- ND adjacency-geometry mechanism: recovered affine-region assignments are
  scanned for cross-region nearest-neighbor contact witnesses; every passing
  edge stores an affine-equality hyperplane and bounded projection residual.
- ND adjacency-geometry passed teachers: 4/4.
- ND adjacency-geometry min passing edges: `62`.
- ND adjacency-geometry min largest-component ratio: `0.9622641509433962`.
- ND adjacency-geometry max normalized projection RMSE:
  `0.9926122288520228`.
- ND adjacency-geometry max MAE: `0.035446586421537286`.
- ND adjacency-geometry min R2: `0.9973558277920783`.
- ND adjacency-geometry total raw oracle calls: `805100`.
- ND adjacency-geometry artifact:
  `benchmarks/release/nn-nd-adjacency-geometry-deep-blackbox-relu-region-sampler/3d-4d-clean-noisy/report.json`
- ND adjacency-geometry stabilization check on 2026-05-28: the existing 3D/4D
  clean/noisy release benchmark was run three times consecutively and passed
  all three cycles with identical gate metrics: `4/4` teachers, min passing
  edges `62`, min largest-component ratio `0.9622641509433962`, max normalized
  projection RMSE `0.9926122288520228`, max MAE `0.035446586421537286`, min
  R2 `0.9973558277920783`.
- 5D single-probe experiment PASS on 2026-05-28.
- 5D probe criteria were deliberately narrow: R2 > `0.99`, regionCount > `50`,
  and raw oracle calls < `2000000`; it was not promoted to release capability.
- 5D probe result: R2 `0.9987241546608961`, MAE
  `0.01894093946992718`, regionCount `57`, raw oracle calls `198000`,
  geometry pass `true`, passing edges `432`, largest-component ratio
  `0.9649122807017544`, max normalized projection RMSE
  `0.5429795030091397`.
- 5D probe artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-5d/single-probe/report.json`
- 5D clean/noisy sweep PASS on 2026-05-28.
- First 5D sweep attempt failed correctly instead of being hidden: clean/noisy
  accuracy passed, but low-amplitude and near-parallel profiles collapsed below
  the `regionCount > 50` gate. The repaired benchmark tightened per-profile
  noisy clustering and increased sample budgets while keeping each teacher
  below `2000000` raw oracle calls.
- 5D sweep result: `6/6` teachers passed across clean, low-amplitude,
  cancellation, near-parallel, and weak-support profiles; min region count
  `57`, min passing edges `462`, min largest-component ratio
  `0.9672131147540983`, max normalized projection RMSE `1.2786982674727818`,
  max MAE `0.07984871493240715`, min R2 `0.985430220526229`, max raw oracle
  calls per teacher `1188000`.
- 5D sweep artifact:
  `benchmarks/release/nn-nd-adjacency-geometry-5d-sweep/clean-noisy-hard-profiles/report.json`
- 6D guarded research probe PASS on 2026-05-28.
- 6D probe result: `4/4` teachers passed across clean, noisy cancellation, and
  noisy near-parallel profiles; min region count `63`, min passing edges `374`,
  min largest-component ratio `0.9850746268656716`, max normalized projection
  RMSE `0.939255366086159`, max MAE `0.09099788983192539`, min R2
  `0.9674291733376437`, max raw oracle calls per teacher `1989000`.
- 6D probe artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d/guarded-probe/report.json`
- 6D adaptive probe-distribution search PASS on 2026-05-28.
- 6D adaptive search result: `4/4` teachers passed across clean, noisy
  cancellation, and noisy near-parallel profiles; `9` probe candidates
  evaluated; `5` rejected candidates were genuine gate failures; min region
  count `63`, min passing edges `374`, max MAE `0.09099788983192539`, min R2
  `0.9674291733376437`, max selected raw oracle calls per teacher `1989000`,
  total selected raw oracle calls `4537000`, and total adaptive search raw
  oracle calls `14482000`.
- 6D adaptive artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d-adaptive-probe/adaptive-probe-search/report.json`
- 6D release-candidate hard-profile sweep PASS on 2026-05-28.
- First 6D sweep attempt failed correctly instead of being hidden: weak-support
  accuracy passed, but region count was only `49` against the `56` floor, and
  the total search budget was `24836500` raw oracle calls against the
  `24000000` gate. The repaired sweep tightened weak-support clustering and
  shortened the near-parallel candidate list while keeping the failure ledger.
- 6D sweep result: `6/6` teachers passed across clean, cancellation,
  near-parallel, low-amplitude, and weak-support profiles; `12` candidates
  evaluated; `4` rejected candidates were genuine gate failures; min region
  count `57`, min passing edges `374`, max MAE `0.09099788983192539`, min R2
  `0.9674291733376437`, max selected raw oracle calls per teacher `1989000`,
  total selected raw oracle calls `8105500`, and total adaptive search raw
  oracle calls `19220500`.
- 6D sweep artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d-sweep/release-candidate-hard-profiles/report.json`
- 6D dependency-free external-method baseline court PASS on 2026-05-31.
- 6D baseline court covered the exact 12-teacher robustness manifest against
  global linear ridge, quadratic ridge, random-ReLU ridge, local-affine kNN, and
  ExtraTrees-style forest baselines. GCL won MAE on `12/12` teachers; max GCL
  MAE was `0.09301806108793807`, min R2 was `0.9601449089658317`, max GCL MAE
  ratio versus the best baseline was `0.8778374199232243`, and median ratio was
  `0.7684385065146409`.
- 6D baseline court artifact:
  `benchmarks/release/nn-nd-adjacency-geometry-6d-baselines/robustness-manifest-baseline-court/report.json`
- 6D is stronger now, but still not a broad SOTA claim. The baseline comparison
  shows an accuracy win against the dependency-free methods, while also showing
  the query-cost boundary: GCL used up to `1501500` raw oracle calls per
  teacher, while these direct-sample baselines used at most `18000`. This
  baseline court is now in `release:gate`; next target: query-efficiency
  hardening plus stronger external package baselines.
- 6D query-budget ladder PASS on 2026-05-31 after adding optional
  Latin-hypercube sampling and explicit weak-region support floors to the ND
  region sampler. The old random sampling mode and default support floor remain
  the default path, and the high-budget candidates remain as fallback proof.
- 6D query-budget result: `6/6` teachers passed the same hard-profile
  accuracy/geometry gates; selected tiers improved from low/low/mid/high/high/low
  to low/low/mid/mid/mid/low after the near-parallel Latin-hypercube challenger
  and low-amplitude weak-region coverage challenger both passed before high
  tier. `5` rejected candidates were genuine gate failures; selected raw oracle
  calls dropped from `8105500` to `5050500`, a `37.69045709703288%` reduction;
  total search raw oracle calls dropped from `19220500` to `10692500`, a
  `44.36929320257017%` reduction. Max selected calls per teacher dropped from
  `1989000` to `1501500`, while min region count increased to `58`, min
  adjacency edges increased to `468`, max MAE was `0.09301806108793807`, and
  min R2 was `0.9601449089658317`. The gate now requires
  `noHighTierSelections`.
- 6D query-budget artifact:
  `benchmarks/research/nn-nd-adjacency-geometry-6d-query-budget/budget-ladder-hard-profiles/report.json`
- 6D still needs more evidence before broad SOTA language: the current ladder
  now has zero high-tier selections, but still needs wider randomized 6D
  coverage and stronger external package baselines.
- Topology-boundary max endpoint line residual: `4.441e-16`.
- Topology-boundary minimum topology segment length: `0.070194`.
- Topology-boundary minimum boundary compression ratio: `10.022222x`.
- Topology-boundary max MAE: `6.393e-4`.
- Topology-boundary min R2: `0.999979703`.
- Topology-boundary total raw oracle calls: `48735`.
- Topology-boundary artifact:
  `benchmarks/release/nn-topology-boundary-deep-blackbox-relu2d-region-graph/fixed-plus-randomized/report.json`
- Topology-boundary passes inside `npm run release:gate` on 2026-05-26.
- Fixed failure: the first topology-boundary corridor used only raw grid
  contact midpoints and missed the affine equality line for three seed-83
  segments. The solver now expands the contact corridor with projected contact
  points, so valid recovered equality lines clip without fallback while the
  prediction segments remain unchanged.
- Direct boundary-offset/root-localization prototypes were not promoted because
  they regressed held-out accuracy on randomized teachers. The release-gated
  fix is lower-risk oracle-guided subcell grid validation with explicit failed
  candidate accounting.
- The first adaptive run was not silenced: it failed on near-parallel /
  low-amplitude cases, then the solver added per-region affine refitting and
  the generator was adjusted to keep weak profiles non-degenerate under the
  benchmark noise model.
- Fixed failure: the first noisy attempt failed because the solver capped
  explicit `gradientStep` to 10% of a grid cell, shrinking the requested
  noise-tolerant step `0.075` to `0.00494` and amplifying noise into false
  gradients. The solver now preserves explicit gradient steps up to a domain
  safety cap, and the regression test checks repeated-query median probing.

Required behavior:

- Treat the deep black box as an output function first: recover the observable
  piecewise-linear region graph accurately enough to predict held-out points.
- Prove that the recovered surface is not just a plane or first-layer-only
  ReLU basis by gating on region count, adjacency, kink-cell boundaries, and
  strong baseline improvements.
- Keep the boundary explicit: output-equivalent region-graph recovery is a
  valid black-box claim; original hidden-weight recovery and exact layer
  attribution are not identifiable from output-only access without more
  assumptions.

Next exits:

- Move from 2D folded region graphs to ND deep folded region recovery.
- Keep topology-boundary geometry as a non-negotiable recovered-adjacency gate
  for future deep-ND folded recovery.
- Add an activation-access control lane that can validate layer attribution
  when hidden activations are explicitly available.

## Previous Frontier: Adaptive Adversarial ND Teacher Generation

The previous frontier target was adaptive adversarial ND teacher generation.

The generated adversarial noisy 4D/5D line-probe lane proved that GCL can
recover shallow ReLU hyperplanes under noisy and quantized oracle-only access
for fixed hard profiles: near-parallel, offset-collision, low-amplitude, and
oblique weak-support. The adaptive lane makes the hard cases search-generated
instead of hand-selected.

Target command:

- `npm run benchmark:nn-noisy-blackbox-relu-nd-adaptive-adversary`

Current status:

- Direct benchmark PASS on 2026-05-24.
- Candidate pool: 24 generated 4D/5D teachers.
- Selected hard teachers: 4.
- Passed teachers: 4/4.
- Max MAE: `0.001696`.
- Min R2: `0.99982863`.
- Max plane error: `0.008613`.
- Total raw oracle calls: `3708980`.
- Artifact:
  `benchmarks/release/nn-noisy-blackbox-relu-nd-adaptive-adversary/adaptive-search/report.json`
- The lane passes inside `npm run release:gate` on 2026-05-24.

Required behavior:

- Generate a large candidate pool of 4D/5D shallow ReLU teachers with
  adversarial knobs: near-parallel normals, offset collisions, weak readout
  amplitudes, weak first-axis line intersections, low support near domain
  boundaries, coarse quantization, and stochastic oracle noise.
- Score candidates before the expensive solve by estimated difficulty: minimum
  pairwise normal distance, offset distance, expected line delta slope, support
  fraction, boundary proximity, and cancellation risk.
- Select the hardest candidates and run the real noisy ND line-probe solver
  against them with repeated oracle calls only.
- Save the selected teachers, difficulty scores, solver metrics, query counts,
  and failures under `benchmarks/release/`.
- Gate on output-equivalent recovery: low MAE, high R2, bounded plane error,
  support thresholds, and explicit raw oracle-call budget.
- If an adaptive case fails, do not lower the gate and do not hide the failure.
  Either repair the solver and keep the case as a regression fixture, or leave
  the lane non-promoted with the failing artifact visible.

Likely solver upgrades if adaptive search exposes a failure:

- Multi-threshold kink extraction so weak line kinks are collected without
  letting noise dominate.
- RANSAC plane refit from all inliers after a candidate plane is selected.
- Weighted inlier scoring by kink confidence and absolute line delta slope.
- Residual-driven second pass that looks for missed weak hyperplanes after the
  first symbolic circuit is fitted.
- Query-budget controller that raises repeats only where line probes are
  ambiguous instead of uniformly spending calls everywhere.

Fixed failure case:

- The first adaptive run solved all selected teachers accurately, but one 5D
  mixed case exceeded the raw oracle-call budget because the generator increased
  `lineCount` once per hidden term instead of once per teacher.
- The generator now applies that budget increase once per teacher. The same
  adaptive search passes with the 5D mixed case reduced from 560 to 410 line
  probes while preserving accuracy.

Promotion boundary:

- This target can promote a stronger shallow noisy ND black-box ReLU recovery
  claim for the gated shallow 4D/5D setting.
- It still must not claim deep ReLU composition recovery, arbitrary production
  model extraction, or exact hidden-unit identity for functionally
  indistinguishable cancellation pairs.

## Direction 1: Noise-Tolerant Oracle Decompilation

Status: started and release-lane smoke gated for 1D, 2D, and 3D.

Why first:

- Real black-box APIs are not clean mathematical oracles.
- Production outputs may be stochastic, rounded, quantized, batched, or served
  through low precision inference.
- ND and deep recovery are not useful if the base oracle layer breaks under
  small output noise.

Current implemented gate:

- Command: `npm run benchmark:nn-noisy-blackbox-relu-decompile`
- Teacher: fixed 1-input/5-hidden ReLU MLP
- GCL access: repeated noisy `f(x)` only
- Noise: Gaussian output noise plus quantization
- Required: recover all kinks within tolerance, beat linear strongly, maintain
  high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-decompile`
- Teacher: fixed 2-input/3-hidden non-crossing non-vertical ReLU MLP
- GCL access: repeated noisy `f(x,y)` only
- Noise: Gaussian output noise plus quantization
- Required: recover kink-line geometry within tolerance, beat a plane baseline,
  maintain high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-decompile`
- Teacher: fixed 3-input/3-hidden non-vertical ReLU MLP
- GCL access: repeated noisy `f(x,y,z)` only
- Noise: Gaussian output noise plus quantization
- Required: recover hyperplane geometry within tolerance, beat a linear
  hyperplane baseline, maintain high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-randomized`
- Teacher: deterministic non-degenerate randomized 3-input/3-hidden ReLU MLP
  seed sweep.
- GCL access: repeated noisy `f(x,y,z)` only.
- Noise: Gaussian output noise plus quantization.
- Required: every seed recovers 3/3 hyperplanes within tolerance, beats a
  linear hyperplane baseline, and maintains high R2 against the clean teacher.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-cancellation`
- Teacher: deterministic 3-input/3-hidden ReLU MLP hard cases with close
  opposite-sign kink pairs.
- GCL access: repeated noisy `f(x,y,z)` only.
- Method: noisy slice stitching seeds the model, then a global parametric
  ReLU-surface refiner fits the full 3D piecewise-linear surface.
- Required: recover an output-equivalent symbolic surface on every hard
  teacher, beat slice-only stitching by at least 70%, beat a linear hyperplane
  baseline by at least 85%, and maintain high R2 against the clean teacher.
- Boundary: this gate proves functional transparent surface recovery, not
  unique hidden-unit identity for cancelling near-collision pairs that can be
  functionally indistinguishable under noisy black-box output.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-adversarial`
- Teacher: six adversarial shallow-3D ReLU MLP profiles: near-parallel,
  beta-collision, cancellation, oblique-wide, and low-amplitude mixed.
- GCL access: repeated noisy `f(x,y,z)` only.
- Method: noisy slice stitching plus global parametric ReLU-surface refinement.
- Required: every profile reconstructs an output-equivalent symbolic surface,
  beats a linear hyperplane baseline, avoids regression against slice-only
  stitching, and reports actual oracle query counts.

Next exits:

- Expand query-budget reporting beyond the current adversarial 3D sweep into
  the older 1D/2D/3D smoke and randomized lanes.
- Add adversarial precision variants: low query repeat, coarse quantization,
  and biased API rounding.
- Add harder noisy 2D cases: low-amplitude hidden units.

Fixed failure case:

- The original noisy 2D crossing-line teacher produced good MAE but wrong
  geometry for one weak kink line because bad per-slice detections polluted the
  line fit.
- The stitching layer now uses robust kink-line fitting with slice outlier
  rejection.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-crossing`
- Required: recover crossing-line geometry and maintain high held-out R2.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-near-parallel`
- Required: separate near-parallel kink lines without merging and maintain high
  held-out R2.
- Command: `npm run benchmark:nn-noisy-blackbox-relu2d-near-collision`
- Required: separate kink lines when two collide on an observed slice and
  maintain high held-out R2.
- The unrestricted noisy 3D randomized attempt exposed near-collision and
  cancellation cases where close opposite-sign hyperplanes were merged or missed
  by slice-only stitching.
- The 3D noisy hard case is now covered by the global refiner gate.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-cancellation`
- Required: output-equivalent symbolic surface recovery on the hard cases and
  strong improvement over the slice-only baseline.
- The wider adversarial 3D sweep now covers near-parallel, beta-collision,
  cancellation, oblique-wide, and low-amplitude mixed profiles.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-adversarial`
- Required: 6/6 profiles pass with explicit oracle query-budget accounting.

## Direction 1b: Whole-Layer FFN Recovery

Status: built and release-gated from 16 to 512 units. The climb to a whole layer has now run
and it stops. This section records the route, what it replaced, and which of its own earlier
predictions measurement has since destroyed - including the one at the top of this paragraph
until the run came back.

The ceiling, measured rather than predicted:

| units | recovered | held-out R2 | outcome |
| --- | --- | --- | --- |
| 512 | 512 / 512 | 0.99936 | release-gated |
| 1024 | 1022 / 1024 | 0.99788 | measured |
| 2048 | 1565 / 2048 | 0.85138 | degraded |
| 4096 | 0 / 4096 | 0.36029 | failed |
| 8192 | — | — | did not run |

The route breaks between 1024 and 2048 at this probing density, and 4096 cannot work at all:
the block is wider than the oracle's 3072-wide answer, which makes the unmixing `R = A D`
underdetermined, so `D+` returns the minimum-norm member of an infinite family and the
projection fallback leaks as `sqrt(k/width)` - above one at 4096 in 3072, larger than the
signal. That is arithmetic, so the run now refuses it up front instead of spending forty-five
million oracle calls to arrive at it.

Two things this is *not*. It is not the directions becoming parallel: 8192 random directions
in 3072 dimensions still have typical pairwise cosine near 0.018, and both failing runs
over-segmented rather than merged, finding 14503 and 16008 clusters. And it is not a limit on
reading a whole layer in principle - the up solve recovers each rank-1 `M_i` from known gates
without inverting `D` at all, so the unmixing is an optimisation that has to be dropped past
the response width rather than a step the method depends on.

All four runs above reported `verdict: recovered` with an empty findings list, including the
one that recovered nothing. Every family in the identifiability check examined the solutions
that came back and none examined what failed to; two counting families now close that, and no
released rung is affected because each returns exactly its requested count.

Why the scalar path stops. Boundaries are read as kinks in a scalar oracle. A SiLU
transition along a probe line has width `1/(S*|g.d|)` for probe scale `S`, while the number
of boundaries crossed grows with the unit count, so kinks merge. Measured on a synthetic
teacher at fixed scale, the share of boundaries read at cosine above `0.99` falls `94%` at 32
units, `62%` at 128, `53%` at 512, `17%` at 2048.

Those numbers are real and the conclusion once drawn from them was not. This section used to
end by extrapolating a required probe scale of `5e6` for a full layer; see the retraction
below, where measurement at 1024 units contradicted it in direction as well as magnitude. The
scalar path does stop - the vector oracle replaced it for reasons that survive - but not for
the reason stated here.

Why per-channel decomposition does not rescue it. Measured from the LLaMA-3.2-3B layer-15
`down_proj`: the participation ratio of an output channel over the 8192 units has median
`2587`, and `3601` units are needed for 90% of a channel's energy. No channel is sparse. What
is true is that `8190` of `8192` units are in the top 256 of some channel, so channels select
different units even though none of them selects few.

What replaces it. The full FFN output is a vector, `y = D a(x)`, so the Jacobian is
`J(x) = sum_i d_i (x) da_i/dx`. Crossing unit `i`'s boundary switches its activation
indicator while `silu(g_i.x)` is zero there, so `J` jumps by `(u_i.x) * d_i (x) g_i`, a rank-1
matrix. One SVD of that jump returns the unit's gate direction and its output direction at
once, with no fitting. Boundaries that overlap along the line stop colliding, because their
jumps occupy different rank-1 subspaces rather than competing for the same scalar.

Measured on a 512-unit synthetic teacher in 512 dimensions,
`tools/llm_reader/jacobian_jump_probe.py`:

- Differencing the Jacobian either side of a peak gives a rank-1 share of `0.14`.
  Extrapolating each side linearly from two points first removes the other units' smooth
  drift and takes it to `0.54`, with median gate cosine `0.99991`.
- The rank-1 share is itself the quality filter and needs no access to truth. Unfiltered,
  `68%` of crossings clear cosine `0.99`; at threshold `0.5`, `81%`; at `0.7`, `100%` at a
  median of `0.99999` for both the gate and the output direction.
- One line of 8001 samples touches every one of 512 units at least once across 60 lines, and
  costs `179842` oracle calls for 40 examined crossings.

Closing the layer once the gates are known. With `g_i` fixed, `silu(g_i.x)` is a known
function of the probe point and

    y(x) = sum_i silu(g_i.x) * (d_i (x) u_i) x

is linear in the matrices `M_i = d_i (x) u_i`. One least-squares solve returns all of them,
and each is rank-1, so its SVD splits it back into the output direction and the up direction.
Measured on a 64-unit teacher in 64 dimensions with 4000 points: every unit returned above
cosine `0.99` on both, median `0.999834` for down and `0.999575` for up, with no nonlinear
optimisation anywhere. A wrong gate set does not survive this step, which is the property that
makes it a check as well as a solve.

Run against the real model. `tools/llm_reader/jacobian_boundary_recovery.py` drives this
route end to end through `OracleClient`. On LLaMA-3.2-3B layer 15, measured per source unit
and not per solution:

`unitsFullyRecovered` is the bar: gate direction, up direction and gate norm together,
matched to one source unit. It replaced a joint direction bar that read `64/64` on a run whose
gate norms were wrong by four orders of magnitude - a measure a bad solution also satisfies,
the same shape of mistake as `distinctUnitsMatched`.

`heldOutR2` is the only column a caller without the model's weights can produce. The recovered
block is a complete function, so it is run on points the solve never saw and compared against
the oracle.

`realTextDeltaCosine` is the second half of the blind check and the reason it is not one
number. `heldOutR2` is measured on the probe cloud, where `||x|| ~ 111`; the model runs on
hidden states that put `g.x` at a standard deviation of 0.265. So each recovered unit is also
silenced against real sentences and compared with a hooked forward pass.

| block | fully recovered | spurious | held-out R2 | real-text delta cosine | calls |
| --- | --- | --- | --- | --- | --- |
| 16 unit | `16/16` | 0 | `0.9986804` | `0.9995280` | `100624` |
| 32 unit | `32/32` | 0 | `0.9990760` | `0.9995760` | `167560` |
| 64 unit | `64/64` | 0 | `0.9994036` | `0.9996909` | `338288` |
| 128 unit | `128/128` | 0 | `0.9994209` | `0.9997182` | `787264` |
| 256 unit | `256/256` | 0 | `0.9994170` | - | `2074336` |
| 512 unit | `512/512` | 0 | `0.9992410` | - | `6155552` |

The held-out bar is per block and taken from measurement, not from a round number. A 16-unit
block reconstructs to `0.99868` and stays there from 8000 probe samples to 80000, so that is
its floor rather than a sampling artefact: with fewer units, each unit's parameter error is a
larger share of the total signal. The real-text figure does not show the same effect, which is
consistent with that reading.

| 1024 unit | `1022/1024` | 0 | `0.9978800` | - | `21134848` |

The resolvability rule that used to sit here was wrong, and 1024 units is what disproved it.
It compared a `silu` knee's width along a line, `1/(S ||g|| |g.d|)`, against the spacing
between boundaries, `2/k`, and concluded that a probe scale of `2e4` carries about 770 units.
Measured at 1024 on that exact scale: 1022 recovered, no spurious solutions, held-out R2
`0.99788`. Doubling the scale to `4e4` made it slightly *worse* - 1015 and `0.99032` - which
also rules out the rule's direction, not only its constant.

The error was in comparing the wrong quantities. Recovery does not need every boundary
resolved on every line; it needs enough crossings of each unit across all lines. At 1024 units
the scan reads 34.5 crossings per line, a detection rate of 3.4%, and with 1280 lines that is
still about 44 independent reads per unit. Per-line resolution was never the binding
constraint.

A 1024-unit block used to die inside the CUDA caching allocator on `nvmlInit_v2`, because the
host's kernel module and userspace library were on different releases: the allocator calls
NVML itself once the design is large enough, which 1024 reached and 512 did not. A reboot
loaded the matching module and the block ran. It is recorded because the failure was a host
condition wearing the costume of an algorithmic ceiling, and because the repair that made it
visible - letting the card's failures leave by the front door instead of falling back to the
host in silence - is what turned it from a mystery into a one-line diagnosis.

The null-space gate is measured and not shipped. On the same 32 units it reaches 31 of 32 at
600 lines and `1047784` calls, against 32 of 32 at `167560` for the amplitude route - six times
the cost for a worse result. Recording that is useful; asserting it as a passing gate, or
lowering a bar to let it through, would not be.

`fullyRecovered` requires the gate direction, the up direction and the gate norm together,
matched to one source unit, and the gate cosine is **signed** - a unit whose gate points the
wrong way fires exactly when the real one is silent, and an absolute cosine counted it as
recovered.

Scope, because the numbers mean nothing without it. The blocks are uniform random draws over
the layer's 8192 units, and measured against the whole layer they sit at the 47th to 51st
percentile of both `||g||` and `||d||`, where a uniform sample would average the 50th - so
these are ordinary blocks rather than an easy tail. The 64- and 128-unit blocks are
independent draws sharing no unit; 256 extends 128 and 512 extends 256, so the upper rungs
measure what **density** does to the same units rather than sampling fresh ones.
`benchmarks/fixtures/jacobian-boundary-units/README.md` carries the detail.

The held-out number is what makes the claim falsifiable rather than self-reported: the same
64-unit block scanned with 14 lines instead of 80 recovers 37 of 64 and scores `0.61923`,
without being told the unit count and without touching a weight. Four units missing out of 128
takes it from `0.999421` to `0.962627`, so it notices a partial block and not only a broken
one.

Cost, measured. `787264` calls for 128 units, `2074336` for 256, `6147360` for 512,
`21134848` for 1024 - a growth exponent that rises from 1.40 to 1.78 across those steps. The
amplitude sweep is 83% of the 1024 figure and is the term that decides a layer: `8 * rank`
calls per sweep, with the rank capped at the input dimension, and roughly one sweep per unit
once the sweep stops when the direction already explains the crossings it was not built from.
That projects to about 228M calls for 8192 units. The oracle answers at 3942 calls a second
over the whole layer, so the projection is hours rather than the days an earlier estimate of
400M and two sweeps per unit produced.

The joint gate/up cosine is the bar the fitted lanes report, so that row is a like-for-like
comparison with the fitted path's `230633` calls - and it also returns the output direction,
which the fitted path does not produce.

Three things had to be built for the joint number to exist at all, and each was found by a
measurement that contradicted the obvious reading:

- Spurious solutions are rejected, not reported. A gate is only right if the crossings it was
  built from lie on its own hyperplane, so the residual over the cluster is a check that costs
  no calls and needs no truth.
- A gate read from one crossing reaches cosine `0.99866`; averaged over several crossings of
  the same unit it reaches `0.99962`. Each crossing is an independent read of one direction.
- **The gate norm has to be measured, not fitted.** A recovered gate is a direction with norm
  1 while the real one on this layer is `1.07` median, and `silu` is not scale free, so the
  norm decides which part of the curve the unit lives on. Searching it against the unit's own
  projection in the isotropic probe cloud is degenerate: as the candidate goes to zero
  `silu(cz) -> cz/2`, the design collapses onto `z x` and the free up direction absorbs `c`,
  so every small candidate scores the same residual and noise picks the grid floor. That put
  25 of 64 norms outside 10% of truth, several at `2e-4`, which describes a unit that never
  fires. Probing along the unit's own gate direction removes the degeneracy - on
  `x(s) = s g_i + p` the pre-activation is exactly `||g_i|| s`, so the norm sets where the
  curve turns on, which no fitted coefficient can imitate. 64 of 64 within 10%. There is no nonlinear optimisation in it: a basis from stacked Jacobians,
an identity from output directions, a gate from a null space. Artifacts:
the six `llama-3072d-layer15-*unit-amplitude-gate` variants under
`benchmarks/release/nn-live-jacobian-boundary-recovery/`, each carrying the sha256 of the tool
that produced it so a stale one fails the gate rather than reading `pass`.

Three settings decide whether it works, and each was measured wrong first:

- One Jacobian does not span the active subspace. At a single point each unit contributes one
  direction, its own mixture of `g_i` and `u_i`, so a `k`-unit block gives rank `k` and taking
  `2k` singular vectors returns `k` real directions and `k` of noise. Measured: basis energy
  `0.4649`, and every gate solved out of it at cosine `0.46`. Eight stacked Jacobians give
  `1.000000`.
- The basis probe and the boundary scan need opposite scales. At scale `2e4` the block's
  outputs are of order `4e8` in float32 and a step of `1e-4` sits below the noise: basis energy
  on the live model fell from `0.596` to `0.445` as points were added, every added Jacobian
  being mostly noise. At basis scale `1` it is `0.999999849225`.
- The scan's finite-difference step must scale with the point. At `1e-4` on scale `2e4` every
  crossing carried its own noise direction and clustering produced 233 clusters from 233
  crossings. At step `1` the same scan gives 110 clusters from 787 crossings, and the largest
  hold 58 points.

What was open here, and what closed it:

1. The live oracle was scalar. `--oracle-output-mode selected-unit-ffn-vector` returns the
   block's whole residual-stream contribution, and `oracleResponseWidth` records that a call
   carries `hiddenSize` numbers rather than 1 so budgets stay comparable. Closed: every row of
   the table above is a run through it.
2. The closure solve has `units * input_dim` columns - 25 million for a layer - and was written
   off as infeasible. Closed by not forming the matrix: conjugate gradient needs only its
   action, and the action is two matrix products. Given exact gates it returns every up
   direction at cosine `1.000000` where the per-unit projection returns none above `0.99`, and
   through the whole route on synthetic blocks above the channel boundary it takes 86 units
   from `0/86` at R2 `0.8456` to `82/86` at `0.9795`.
3. Cost at layer scale was estimated. Now measured to 1024 units and projected from four
   measured points; see the cost paragraph above.
4. Float32 behaviour was unmeasured. Now measured, and the answer is a rule rather than a
   number: single precision is safe where quantities are compared and fatal where they are
   subtracted. It destroyed the recovery three separate times - the basis Gram from `0.998` to
   `0.888`, the scan difference to `0.56` with half the gates reversed, and the probe-line
   construction to `0.38` - and each time the run completed and reported.

Boundary, and it is the one that matters. Every number above comes from an oracle configured
with `down_weight[:, units]`: the harness selects the block and returns only its contribution.
The solver never reads a weight and the post-hoc truth stays locked until it has finished -
both are checked by the gate, which requires the unlock query count and the solver's own call
count to agree. But the setup is not blind, so these are claims about a block the harness
isolates, not about separating units out of a layer's full summed output. The unselected
whole-layer run is what would change that sentence, and it has not finished.

### The third-derivative route, and the wall it also has

The paragraph above says the response width is an optimisation to be dropped rather than a step
the method depends on, which invites the reading that some other identity reaches a whole layer.
One was built and measured: `tools/llm_reader/cubic_identification_probe.py`. It does identify
units, and it does not reach 8192 either.

Along `t -> y(t d)` the third derivative is `1.5 sum_i (g_i.d)^2 (u_i.d) d_i`, so through a fixed
reader it is a cubic form whose symmetric tensor is `sum_i lambda_i sym(g_i g_i u_i)`. The pencil
`S(.,.,x) S(.,.,y)^+` returns the gate directions at cosine `1.000000`, the unit count comes out
as `rank/2` rather than being supplied, and with the gates known the ups are one linear least
squares - nothing iterative anywhere. The second derivative cannot do this: `sym(g u^T)` is
symmetric in the two, so it cannot say which is the gate.

What it costs and where it stops, measured on a synthetic teacher at three input dimensions:

| n | ceiling `n/2` | at the ceiling | one past it | full-tensor calls | two-slice calls |
| --- | --- | --- | --- | --- | --- |
| 16 | 8 | gate 8/8, up 7/8 | `k=9`: 0/9 | 19584 | 3664 |
| 24 | 12 | gate 12/12, up 12/12 | `k=13`: 0/13 | 62400 | 7792 |
| 28 | 14 | gate 14/14, up 14/14 | `k=15`: 1/15 | 97440 | 10432 |

The wall is `2k <= n` on the *input* dimension, because each unit spans the plane `span{g_i,u_i}`
and the unfolding rank saturates at `n`. For a 3072-wide input that is 1536 units. So this route
moves the wall off the response width and onto the input dimension; it does not remove it, and
1536 must not be cited as a path to a full layer.

The cost is the better half. Fitting every degree-3 coefficient is `4n(n+1)(n+2)` oracle points,
but the pencil reads two slices, and measuring only those by polarisation is
`8(1.5n(n+1) + 3n + 2)` - the same gates to the same cosine for `n/3` less. At `n=3072` that is
`113356816` against `116077387776`, which puts the affordable version in the range the boundary
route already spent (45M on the 4096 rung). Both figures are arithmetic on laws checked against
the call counter at the three dimensions above, not measurements of a live layer, and the tool
labels them `extrapolatedFromMeasuredLaw`.

Reaching 8192 units in 3072 dimensions needs `k > n`, which is an overcomplete decomposition and
a different algorithm rather than a setting on this one. That is the next gate for the whole
layer, and nothing here is wired into a lane: the numbers are from a synthetic teacher where the
truth is known, which is the whole scope of the claim.

#### Retraction: the response width is a bound on probing at one point, not on the layer

This section says above that `4096 cannot work at all` because the block is wider than the
answer, and calls it arithmetic. The arithmetic is right and its scope was not. It is a statement
about derivatives taken at the origin.

At the origin the third derivative weights unit `i` by `1.5 (w.d_i)`, so every reader sees the
same `k` unit tensors through the down projection and the accessible dimension is `rank(D) <= m`.
Measured directly: `rank span{S_w} = min(k, m)` at six settings, including `k=40` behind `m=8`
returning exactly `8`. That is the ceiling this section described.

Off the origin the weight is `beta_i = 3 (w.d_i) silu''(g_i.x0)`. The second factor is a
nonlinear function of the unit, not a linear read through `D`, so base points buy rank that
readers cannot. Measured, `k=40` units behind an `m=8` answer:

| base points `P` | `m*P` | profile rank | smallest/largest singular value |
| --- | --- | --- | --- |
| 1 | 8 | 8 | 4.45e-01 |
| 2 | 16 | 16 | 4.33e-02 |
| 4 | 32 | 32 | 1.18e-02 |
| 8 | 64 | **40** | 1.98e-02 |
| 16 | 128 | 40 | 3.12e-02 |

The rank is `min(k, m*P)` at every point, and the conditioning stays between `1.2e-02` and
`4.5e-01` across two seeds, so the separation is real rather than a rank counted at a threshold.
The same law asks for three base points to reach 8192 units behind a 3072-wide answer.

And the input dimension is not the last wall either. The pencil stops at `2k <= n` because it
diagonalises in `R^n`, where each unit costs two of `n` directions. The `(ab)` unfolding of the
same tensor puts each unit in `Sym^2(R^n)` - `g (x) g` and `sym(g (x) u)` - where the room is
`n(n+1)/2`, quadratic where it was linear. Measured at `n=10`, whose pencil ceiling is 5:

| units `k` | `2k` | span rank | smallest/largest singular value |
| --- | --- | --- | --- |
| 5 | 10 | 10 | 1.82e-01 |
| 12 | 24 | 24 | 7.93e-02 |
| 25 | 50 | 50 | 1.59e-02 |
| 27 | 54 | 54 | 1.13e-03 |
| 28 | 56 | 55 | 8.86e-17 |

The rank is `min(2k, n(n+1)/2)` at every setting and the collapse lands exactly where `2k`
passes the room, so the ceiling is `k <= n(n+1)/4`: 27 units at `n=10` against the pencil's 5,
and **2360064** at `n=3072` against 1536.

Which puts 8192 nowhere near any of the three walls this measured - response width, input
dimension, or lifted room. Identifiability is not what is stopping a whole layer.

#### The overcomplete recovery, and the ceiling it runs into

The extraction was the gate, and it is now written. Searching for `v` directly makes rank-1-ness
a quartic with a flat landscape and nothing to steer by at the solution - three routes tried that
way and returned 2 to 5 of `k`. Rank is a statement about minors, and minors are bilinear:
`Phi(M,N)_ijkl = 1/2 (M_ij N_kl + N_ij M_kl - M_il N_kj - N_il M_kj)` vanishes identically at
`M = N = v v^T`, so `Phi(M(c), M(c)) = 0` is linear in `c (x) c`. The search becomes a null space,
whose spanning elements `c^(i) (x) c^(i)` are rank-1 in the `2k`-dimensional coefficient space -
where a pencil separates them, the same pencil that could not work in `R^n`. Nothing iterates.

| n | pencil ceiling | k | k/n | recovered | median cosine |
| --- | --- | --- | --- | --- | --- |
| 8 | 4 | 10 | 1.25 | 10/10 | 0.999999999999999 |
| 8 | 4 | 12 | 1.50 | 12/12 | 1.000000000000000 |
| 8 | 4 | 13 | 1.63 | 13/13 | 0.999999999999999 |
| 8 | 4 | 14 | 1.75 | 0/14 | - |
| 10 | 5 | 15 | 1.50 | 15/15 | 0.999999999999999 |
| 10 | 5 | 18 | 1.80 | 18/18 | 1.000000000000000 |

Three seeds at n=8, agreeing at every row including the failure. The break at 14 is where the
isolation bound `(n^2-n+2)/4 = 14.5` says solutions stop being isolated, so the solver stops
where the geometry does rather than where its own conditioning gives out - which also makes that
bound a prediction that held rather than a correction after the fact.

Scope, and it is narrower than the table looks. The span `W` here is built analytically from the
true gates and ups, not measured from an oracle. What is verified is the extraction step, not the
whole chain. Joining it to a measured span - `W` from third derivatives at several base points,
which is where the rank was shown to be available - is the remaining work, and until that runs
this is recovery from a span rather than recovery from an oracle. Reproduce with
`--recover-sweep`.

What this changes and what it does not. The units behind a too-narrow answer are separable in
what the oracle can be made to show - that is the necessary condition, and it is now measured
rather than assumed against. It is not a recovery: the pencil still needs `2k <= n`, and nothing
here has recovered `k > n` units. The refusal in the solver stays correct for what the solver
does. What is retracted is only the wider reading that a layer wider than its answer cannot be
read at all; the wall that a whole-layer method has to beat is the input dimension, not the
response width. Reproduce with `--base-point-sweep`.

## Direction 2: 3D / General-ND Hyperplane Stitching

Status: started and release-lane gated for clean/noisy 3D teachers, the first
clean 4D random-direction line-probe teacher, and a deterministic randomized
4D/5D seed sweep.

Why second:

- The current 2D black-box solver already recovers non-vertical ReLU kink
  lines from horizontal slices.
- The natural generalization is to probe many 1D lines through the domain,
  recover kink locations on those lines, then stitch those points into
  hyperplanes.

Proposed architecture:

- Active line sampler: choose lines across coordinate axes and random
  directions.
- Kink point extraction: reuse robust 1D/noisy 1D decompilers per line.
- Hyperplane clustering: group recovered kink points by compatible normals.
- Region adjacency graph: maintain which linear region lies on each side of
  each recovered hyperplane.
- Verification pass: hold out random points and compare reconstructed symbolic
  circuit against the oracle.

Current implemented gate:

- Command: `npm run benchmark:nn-blackbox-relu3d-decompile`
- Teacher: fixed 3-input/3-hidden ReLU MLP with non-degenerate non-vertical
  hyperplanes.
- GCL access: `f(x,y,z)` oracle only.
- Method: recover 2D kink lines on z-slices, then stitch matched lines through
  z into hyperplanes.
- Required: recover 3/3 hyperplanes with geometry error below `1e-8`, beat a
  linear hyperplane baseline by at least 1000x, and keep held-out MAE below
  `1e-9`.
- Command: `npm run benchmark:nn-blackbox-relu3d-randomized`
- Teacher: deterministic randomized 3-input/4-hidden ReLU MLP seed sweep.
- GCL access: `f(x,y,z)` oracle only.
- Method: clean 2D slice recovery with denser `y` slicing per z-slice, then
  hyperplane stitching.
- Required: every seed recovers all hyperplanes with geometry error below
  `1e-8`, and every seed keeps held-out MAE below `1e-9`.
- Command: `npm run benchmark:nn-blackbox-relu-nd-line-probe`
- Teacher: fixed 4-input/4-hidden shallow ReLU MLP.
- GCL access: `f(vector)` oracle only.
- Method: random-direction line probes recover 1D kink points; RANSAC stitches
  kink points into N-D hyperplanes; a final linear readout fit reconstructs an
  equivalent symbolic ReLU circuit.
- Required: recover 4/4 hyperplanes, keep max plane error below `1e-3`, beat a
  linear hyperplane baseline by at least `1000x`, keep held-out MAE below
  `1e-8`, and report oracle query count.
- Command: `npm run benchmark:nn-blackbox-relu-nd-line-probe-randomized`
- Teacher: deterministic randomized 4D/5D shallow ReLU MLP seed sweep.
- GCL access: `f(vector)` oracle only.
- Method: same random-direction line probing, RANSAC hyperplane stitching, and
  linear readout fitting as the fixed 4D lane.
- Required: all randomized 4D/5D teachers recover every hyperplane, keep max
  plane error below `1e-3`, beat linear by at least `1000x`, keep MAE below
  `1e-8`, satisfy support thresholds, and stay under the query-budget cap.
- Command: `npm run benchmark:nn-noisy-blackbox-relu-nd-line-probe`
- Teacher: deterministic randomized 4D/5D shallow ReLU MLP seed sweep with
  stochastic output noise and quantization.
- GCL access: repeated noisy `f(vector)` oracle only.
- Method: random-direction line probing uses median-aggregated 1D noisy ReLU
  segmentation; weak noisy line kinks are filtered before RANSAC stitching; a
  median-aggregated final readout fit reconstructs an equivalent symbolic ReLU
  circuit.
- Required: all noisy 4D/5D teachers recover every hyperplane, keep max plane
  error below `0.06`, beat a linear hyperplane baseline by at least 75%, keep
  MAE below `0.025`, keep R2 above `0.998`, satisfy support thresholds, and
  stay under the query-budget cap.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-decompile`
- Teacher: fixed 3-input/3-hidden ReLU MLP with noisy and quantized oracle
  output.
- GCL access: repeated noisy `f(x,y,z)` oracle only.
- Method: noisy 2D slice recovery per z, followed by robust `beta(z)`
  hyperplane stitching.
- Required: recover 3/3 hyperplanes within `0.12` geometry tolerance, beat a
  linear hyperplane baseline by at least 65%, and keep held-out MAE below
  `0.08`.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-cancellation`
- Teacher: noisy 3D near-collision/cancellation hard cases with close
  opposite-sign hidden units.
- GCL access: repeated noisy `f(x,y,z)` oracle only.
- Method: noisy slice recovery seeds a global parametric refiner that optimizes
  the full 3D ReLU surface and readout coefficients.
- Required: output-equivalent surface MAE below `0.012`, R2 above `0.999`, and
  strong improvement over both linear and slice-only baselines.
- Command: `npm run benchmark:nn-noisy-blackbox-relu3d-adversarial`
- Teacher: wider noisy adversarial 3D profile sweep over shallow ReLU MLPs.
- GCL access: repeated noisy `f(x,y,z)` oracle only.
- Method: slice stitching seeds a global refiner; benchmark counts actual oracle
  calls.
- Required: 6/6 profiles pass, max MAE below `0.020`, R2 above `0.9985`, and
  no bad regression against slice-only stitching.

Boundary:

- This should not claim arbitrary high-dimensional network recovery until it
  survives adaptive search-generated ND profiles and deep composition recovery.
  The clean, noisy, generated adversarial, and adaptive-adversary 4D/5D shallow
  lanes now pass inside the release gate.

Next exits:

- Broaden adaptive adversarial search to larger candidate pools and stricter
  budget tiers.
- Add residual-driven second-pass recovery for missed weak hyperplanes.
- Start the deep/staked ReLU composition recovery lane.

## Direction 3: Deep ReLU Composition Recovery

Status: started with a release-lane folded 1D output-equivalent recovery gate.

Why third:

- Deep ReLU networks fold space. The visible function is still piecewise
  linear, but internal layers are not identifiable from output alone without
  extra assumptions.
- A correct deep lane needs to distinguish equivalent surface recovery from
  exact hidden-layer recovery.

Proposed architecture:

- Start with equivalent-function recovery, not exact original weights.
- Recover piecewise-linear regions and adjacency graph.
- Search for compact factorizations of the recovered surface into layered ReLU
  circuits.
- Add an activation-access control lane for validating factorization when
  hidden layers are available.

First gates:

- Command: `npm run benchmark:nn-deep-blackbox-relu1d-decompile`
- Teacher: fixed two-layer ReLU network over 1D.
- GCL access: output-only equivalent surface recovery.
- Required: held-out reconstruction accuracy, recovered canonical kink count
  greater than the first-layer kink count, at least 5 deep-only kinks, and
  a GCL MAE advantage over both a linear baseline and a first-layer-kink-only
  baseline.
- Command: `npm run benchmark:nn-deep-blackbox-relu2d-region-graph`
- Teacher: fixed two-layer ReLU network over 2D.
- GCL access: output-only equivalent surface recovery.
- Required: held-out reconstruction accuracy, recovered affine-region
  adjacency graph, region count greater than the first-layer structure, and
  a GCL MAE advantage over both plane and first-layer-only ReLU baselines.

Boundary:

- Do not call this "exact deep reverse engineering" unless hidden activations
  or strict identifiability assumptions are part of the gate.
