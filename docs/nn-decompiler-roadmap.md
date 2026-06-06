# NN Decompiler Roadmap

This roadmap keeps the neural-network decompiler work honest. A lane becomes a
release claim only when it has a command, artifact, gate, and clear boundary.

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
4. Scale the fact set
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
    18. NEXT: Add scaling stress as a separate ladder: `768D -> 4096D -> 16384D`,
       then multi-layer smooth composition, always with query budget and
       negative-case behavior. A dimension jump is not a structure claim by
       itself.
    19. DONE for full-slice/rank-cap negative guarding: Add
       `benchmark:nn-gelu-ffn-rank-cap-negative`, where a self-contained
       synthetic 768D/3072-channel full-slice teacher must report
       `insufficient-rank` under a too-small rank cap instead of promoting a
       weak structure claim. Add
       `research:nn-gpt2-gelu-ffn-full-rank-cap-negative` as a local audit over
       the uncommitted full GPT-2 fixture.
    20. NEXT: Only after blind top-K is efficient should full FFN slices be used
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
  - Current status: first CUDA gate implemented.
  - Current command: `npm run benchmark:nn-live-gpt2-ordered-residual-path-circuit-finder:gpu`.
  - Current artifact:
    `benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json`.
  - Current result: pass `true`, 50 valid facts, 50 passed cases, pass rate
    `1.0`, combined graph pass rate `1.0`, median selected path activation
    delta `0.16316372156143188`, p10 selected path activation delta
    `0.07098615169525146`, median full-control path p95 activation delta
    `0.02243638038635254`, median ordered path lift over full-control p95
    `6.071965452028251`, median combined target-logit drop
    `1.9691314697265625`, skipped rows `33`, `model.device = cuda`.
  - Output: attention->residual->MLP->target path records with selected
    attention heads, selected MLP nodes, path activation deltas, full-control
    path deltas, combined target-logit evidence, and order-aware rescue
    attempts when a causal union graph is not a valid attention-before-MLP
    path.
  - Claim boundary: this proves benchmark-scoped ordered residual path evidence
    for the configured GPT-2 factual prompt set. It is not exhaustive residual
    patching for every edge, not full GPT-2 circuit recovery, not full FFN
    extraction, not arbitrary LLM understanding, and not output-only
    original-neuron identity.
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
