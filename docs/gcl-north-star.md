# GCL North Star

This document is the focus reset for Genetic Code Lab. If the project starts to
drift into scattered benchmark work, read this first.

## Mission

GCL should become an Active-Subspace Neural Decompiler.

That means GCL should take a black-box neuron, channel, or small block:

```text
query x -> f(x)
```

and recover the smallest useful explanation of how the function behaves:

1. the active subspace where the output actually changes;
2. the recovered functional atoms in that subspace;
3. the canonical structure under equivalent decompositions;
4. source-neuron provenance when hooks, manifests, or signed source metadata
   are available;
5. why each recovered atom reacts, measured by trigger probes, preactivation,
   activation, contribution, and held-out output fidelity.

The product is not "a better random symbolic regression toy". The product is a
structure-aware neural decomposition system with explicit evidence, query
budgets, and failure accounting.

## Access Modes Are Not Interchangeable

GCL must not blur these modes:

1. **Black-box decompiler mode**: the solver receives only oracle calls such as
   `f(x) -> y`, logits, or selected output channels. It does not read model
   weights, hidden activations, source manifests, layer widths, or original
   neuron ids. This is the mode required for a real 4096D black-box claim.
2. **Instrumented witness mode**: the system may use local model weights,
   hooks, hidden activations, or source manifests only to create a real oracle,
   validate a recovered structure after the blind recovery step, or explain
   provenance when observability is explicitly available.
3. **White-box/source-mapping mode**: the system uses weights/hooks/manifests to
   map recovered atoms to known source components. This is useful engineering,
   but it is not proof that GCL solved a black-box decompilation problem.

Any claim that says "GCL reads a black-box LLM/4096D block" must be backed by a
blind/oracle-only recovery artifact. A local model can provide the oracle and
the post-hoc witness, but the solver path being evaluated must not consume the
weights or hooks as its solution.

## LLM Reader Target

The user-facing product target is a real LLM reader, not a benchmark
collection, not a generic surrogate fitter, and not a weight-inspection trick
being mislabeled as black-box reading.

GCL should make selected LLM/NN behavior less black-box through two explicitly
separated rails:

- **Blind recovery rail**: recover active subspaces, functional atoms, and
  circuit/path structure from oracle access only.
- **Witness/provenance rail**: use local weights, hooks, activation traces, and
  source manifests after recovery to verify and label what the blind solver
  found.

For real local LLMs, the reader should do all of the following through model
adapters and measured interventions:

1. load a real local model runtime, tokenizer, and component map;
2. select a prompt/target, neuron/channel, or local block;
3. scan layers, attention heads, MLP units, residual positions, and token
   positions for causal components;
4. build a graph-level circuit family, not only a single lucky path;
5. validate selected edges with causal ablation, path replay, negative
   controls, neighboring-layer controls, paraphrases, OOD paraphrases, and
   budget accounting;
6. when a paraphrase or prompt shift breaks the current path, search for the
   replacement path and record the circuit change instead of promoting the old
   path as universal;
7. explain why selected atoms react using trigger probes, preactivation,
   activation, contribution, residual-path deltas, and target-logit effects;
8. report failures, skipped cases, runtime warnings, query/runtime budgets, and
   claim boundaries in the artifact.

Current verified status: benchmark-scoped real-model graph/path readers exist
for GPT-2, Pythia/GPT-NeoX, LLaMA-family SwiGLU, and Mistral-family SwiGLU
lanes. These gates prove configured graph/path evidence, not full arbitrary LLM
understanding and not pure black-box 4096D LLM decompilation unless the
specific artifact states that the solver consumed oracle calls only. They do
not yet prove full GPT-2 FFN extraction, arbitrary 3072-channel or 4096D LLM
block decompilation, semantic original-neuron identity, exhaustive circuit
completeness, or universal removal of black-box behavior.

## Current Capability Surface

GCL now has four working rails.

### 1. ReLU Geometry Rail

Purpose: prove oracle probing, boundary recovery, region sampling, adjacency
geometry, noise handling, and query-budget discipline.

Current status: release-gated through 1D/2D/3D/4D/5D/6D controlled ReLU
geometry lanes, including noisy/adversarial and baseline-court variants.

Boundary: this rail is a foundation. It does not by itself prove GELU, SwiGLU,
real LLM FFN extraction, or original hidden-neuron identity.

### 2. Smooth FFN Structure Rail

Purpose: recover explicit GELU/SwiGLU-style atoms from high-dimensional
black-box FFN channels without brute-forcing the full dimension.

Current status: release-gated for:

- active-subspace GELU/SwiGLU recovery in 256D/1024D;
- controlled 4096D active-subspace stress;
- noisy rank-1 GELU active-subspace recovery;
- explicit GELU atom recovery;
- rank-adaptive mixed GELU block recovery;
- near-parallel mixed-channel stability;
- cross-decomposition canonical invariance;
- formal negative cases;
- noisy/quantized mixed GELU block recovery for controlled balanced,
  near-parallel, and weak-but-resolvable teachers;
- IID, wide-OOD, and shifted-OOD distribution court for the controlled
  noisy/quantized mixed GELU block;
- scikit-learn, XGBoost, and sklearn-MLP package-backed prediction-sanity court
  for the controlled noisy/quantized mixed GELU block. These fit baselines are
  used to check output value of the recovered structure; they are not treated as
  structure-decompiler competitors;
- rank-adaptive mixed SwiGLU value/gate pair recovery for a controlled 24D
  two-block teacher where GCL is given only a maximum pair budget;
- noisy/quantized mixed SwiGLU value/gate pair recovery for controlled
  balanced, near-parallel-value, and weak-but-resolvable teachers;
- shared-gate/cancellation-heavy SwiGLU canonicalization for controlled clean
  and noisy/quantized teachers, where multiple source value channels collapse
  into one minimal functionally equivalent value/gate pair;
- source-neuron provenance verification when source manifests or equivalent
  instrumentation are provided;
- local transformer MLP hook recovery for a deterministic local
  transformer-style block, with captured residual-stream, MLP-input,
  preactivation, activation, per-source contribution, and target-output rows;
- compact real GPT-2 layer-0 top-3, top-5, and top-10 output-only
  source-structure recovery, with top-8 retained as the intermediate targeted
  scaling check that exposed and then verified gradient-consistency recovery;
- a full-slice/rank-cap negative guard that refuses unsupported 768D/3072-
  channel structure promotion when rank evidence is insufficient;
- a local real GPT-2 full-FFN rank-cap audit that uses the uncommitted 49MB
  fixture to verify that top-20 coverage is not a full-3072 extraction claim;
- compact real GPT-2 layer-0 top-20 instrumented source-graph extraction with
  strict provenance and exact IID/OOD held-out agreement.
- blind oracle-only recovery of selected real local Mistral-family 4096D SwiGLU
  units as explicit rank-2 SwiGLU atoms. The solver receives only batch `f(x)`
  oracle outputs from a separate oracle process; source weights are used only
  after recovery for witness alignment. Current aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-4unit-layer-sweep-2probe-oracle-only/report.json`,
  pass `true`, cases `4/4`, dimension `4096`, total raw oracle calls `73984`,
  max raw oracle calls per case `18496`, min validation R2
  `0.9999845855712838`, max validation MAE `1.5160815735029332e-05`, and min
  subspace principal cosine `0.9999997892743723`.
- deterministic-random blind oracle-only recovery across selected real local
  Mistral-family 4096D SwiGLU units. Current randomized aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-8unit-layer-sweep-2probe-oracle-only/report.json`,
  pass `true`, cases `8/8`, dimension `4096`, total raw oracle calls `147968`,
  max raw oracle calls per case `18496`, min validation R2
  `0.9999570341720895`, max validation MAE `2.4326629529818456e-05`, and min
  subspace principal cosine `0.9999999957944806`.
- blind oracle-only parity recovery across selected real local LLaMA-family
  3072D SwiGLU units using the same solver boundary. Current parity aggregate
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-4unit-layer-sweep-2probe-oracle-only/report.json`,
  pass `true`, cases `4/4`, dimension `3072`, total raw oracle calls `57600`,
  max raw oracle calls per case `14400`, min validation R2
  `0.9999999964505302`, max validation MAE `1.0737975340964412e-05`, and min
  subspace principal cosine `0.9999999979627894`.
- deterministic-random blind oracle-only parity recovery across selected real
  local LLaMA-family 3072D SwiGLU units. Current randomized parity aggregate
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-8unit-layer-sweep-2probe-oracle-only/report.json`,
  pass `true`, cases `8/8`, dimension `3072`, total raw oracle calls `115200`,
  max raw oracle calls per case `14400`, min validation R2
  `0.9999981324158254`, max validation MAE `0.00016030715095856503`, and min
  subspace principal cosine `0.9999994111232651`.
- first blind oracle-only 6-unit SwiGLU block recovery on real local
  LLaMA-family 3072D units. The scalar oracle returns the sum of layer-20
  units `[0,17,123,456,789,1024]`; the solver recovers a rank-12 active
  subspace and an explicit 6-atom SwiGLU sum. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-6atom-oracle-only/report.json`,
  pass `true`, raw oracle calls `88864`, R2 `0.9999999971996777`, MAE
  `2.8572995724253928e-05`, worst error `0.00021262079246214816`, and
  subspace min principal cosine `0.9999999938218761`.
- query-budget hardened 6-unit SwiGLU block recovery on the same real local
  LLaMA-family 3072D oracle using forward-coordinate witnesses. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-6atom-forward-coordinate-oracle-only/report.json`,
  pass `true`, raw oracle calls `48941`, R2 `0.9999975420086665`, MAE
  `0.0008233366167048662`, worst error `0.006377951229655121`, and subspace
  min principal cosine `0.9999960757027645`.
- first randomized blind oracle-only 6-unit SwiGLU block sweep on real local
  LLaMA-family 3072D units. Two deterministic-random same-layer 6-unit scalar
  oracles passed with active rank `12`. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-6unit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `177728`, min R2
  `0.9999999419380987`, max MAE `0.00008572753027982732`, max worst error
  `0.0008619591998977327`, and min subspace principal cosine
  `0.9999998049413197`.
- query-budget hardened randomized 6-unit SwiGLU block sweep on real local
  LLaMA-family 3072D units using forward-coordinate witnesses. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-6unit-forward-coordinate-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `97882`, max raw oracle
  calls per case `48941`, min R2 `0.9999880522847288`, max MAE
  `0.0012813828353731324`, max worst error `0.012340230224598514`, and min
  subspace principal cosine `0.9999574008675185`.
- first blind oracle-only small multi-unit SwiGLU block recovery on real local
  Mistral-family 4096D units. The scalar oracle returns the sum of layer-20
  units `[0,17]`; the solver recovers a rank-4 active subspace and an explicit
  2-atom SwiGLU sum. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-2atom-oracle-only/report.json`,
  pass `true`, raw oracle calls `44128`, R2 `0.9998005671654032`, MAE
  `9.702990494674492e-05`, and subspace min principal cosine
  `0.9999999793123026`.
- first randomized blind oracle-only small-block SwiGLU sweep on real local
  Mistral-family 4096D units. Two deterministic-random same-layer 2-unit
  scalar oracles passed: layer `1` units `[7309,8811]` and layer `27` units
  `[4733,5191]`. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-2unit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `88256`, min R2
  `0.9994744138350042`, max MAE `0.00014261562921587213`, and min subspace
  principal cosine `0.9999999823144813`.
- first blind oracle-only 3-unit SwiGLU block recovery on real local
  Mistral-family 4096D units. The scalar oracle returns the sum of layer-20
  units `[0,17,123]`; the solver recovers a rank-6 active subspace and an
  explicit 3-atom SwiGLU sum. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-3atom-oracle-only/report.json`,
  pass `true`, raw oracle calls `61968`, R2 `0.9992245717327005`, MAE
  `0.0002548553473914666`, and subspace min principal cosine
  `0.9999999974474356`.
- first randomized blind oracle-only 3-unit SwiGLU block sweep on real local
  Mistral-family 4096D units. Two deterministic-random same-layer 3-unit
  scalar oracles passed with an explicit `maxWorstError <= 0.02` gate. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-3unit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `123936`, min R2
  `0.9962650545162728`, max MAE `0.0005119471166077025`, max worst error
  `0.010120884383530829`, and min subspace principal cosine
  `0.9999999878109772`.
- first blind oracle-only 4-unit SwiGLU block recovery on real local
  Mistral-family 4096D units. The scalar oracle returns the sum of layer-20
  units `[0,17,123,456]`; the solver recovers a rank-8 active subspace and an
  explicit 4-atom SwiGLU sum. This case initially failed at R2 `< 0.995` and
  passed after the 4+ atom optimizer path was hardened with float64 fitting,
  residual-start search, and slower LR/fit budget. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-4atom-oracle-only/report.json`,
  pass `true`, raw oracle calls `79808`, R2 `0.9996964505713657`, MAE
  `0.0001659790529608388`, worst error `0.002872809058861975`, and subspace
  min principal cosine `0.9999999945348134`.
- first randomized blind oracle-only 4-unit SwiGLU block sweep on real local
  Mistral-family 4096D units. Two deterministic-random same-layer 4-unit
  scalar oracles passed with active rank `8`. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-4unit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `159616`, min R2
  `0.995727068595925`, max MAE `0.0006359374939938832`, max worst error
  `0.008566143276326953`, and min subspace principal cosine
  `0.9999999725417702`.
- first blind oracle-only 5-unit SwiGLU block recovery on real local
  Mistral-family 4096D units. The scalar oracle returns the sum of layer-20
  units `[0,17,123,456,789]`; the solver recovers a rank-10 active subspace
  and an explicit 5-atom SwiGLU sum. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-5atom-oracle-only/report.json`,
  pass `true`, raw oracle calls `97648`, R2 `0.9970749283083804`, MAE
  `0.0006105457047962797`, worst error `0.008950286237037718`, and subspace
  min principal cosine `0.9999999795104269`.
- first randomized blind oracle-only 5-unit SwiGLU block sweep on real local
  Mistral-family 4096D units. Two deterministic-random same-layer 5-unit
  scalar oracles passed with active rank `10`. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-5unit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `195296`, min R2
  `0.9951436177161573`, max MAE `0.0008022104114991162`, max worst error
  `0.010051115148071464`, and min subspace principal cosine
  `0.9999997906113538`.
- first blind oracle-only 6-unit SwiGLU block recovery on real local
  Mistral-family 4096D units. The scalar oracle returns the sum of layer-20
  units `[0,17,123,456,789,1024]`; the solver recovers a rank-12 active
  subspace and an explicit 6-atom SwiGLU sum. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-6atom-oracle-only/report.json`,
  pass `true`, raw oracle calls `115488`, R2 `0.9970869523285548`, MAE
  `0.0007282176910088243`, worst error `0.008779981374470364`, and subspace
  min principal cosine `0.9999999544565973`.
- query-budget hardened 6-unit SwiGLU block recovery on the same real local
  Mistral-family 4096D oracle using forward-coordinate active-subspace
  witnesses. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-6atom-forward-coordinate-oracle-only/report.json`,
  pass `true`, raw oracle calls `62253`, R2 `0.9970121491993152`, MAE
  `0.0007410395113780754`, worst error `0.00931293073078479`, and subspace
  min principal cosine `0.9999994812351087`.
- first randomized blind oracle-only 6-unit SwiGLU block sweep on real local
  Mistral-family 4096D units. Two deterministic-random same-layer 6-unit
  scalar oracles passed with active rank `12`. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-6unit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `230976`, min R2
  `0.9952423306790068`, max MAE `0.0009230608551877102`, max worst error
  `0.012063083889497478`, and min subspace principal cosine
  `0.9999999518366974`.
- query-budget hardened randomized 6-unit SwiGLU block sweep on real local
  Mistral-family 4096D units using forward-coordinate witnesses. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-6unit-forward-coordinate-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `124506`, max raw oracle
  calls per case `62253`, min R2 `0.9952220102880769`, max MAE
  `0.0009253719069818783`, max worst error `0.012323574241066268`, and min
  subspace principal cosine `0.9999991210977408`.
- first query-budget selected 8-unit SwiGLU block recovery on real local
  Mistral-family 4096D units using forward-coordinate witnesses and explicit
  fit restart budgeting. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-budgeted-oracle-only/report.json`,
  pass `true`, raw oracle calls `81297`, R2 `0.9956625486864737`, MAE
  `0.0010698511933124055`, worst error `0.010449555792974501`, subspace min
  principal cosine `0.9999966163799717`, and fit restarts `160`.
- centered-basis, active-gradient-fit replay of the selected 8-unit
  Mistral-family 4096D block. This is the current strongest selected Mistral
  8-unit clean witness: the solver receives only scalar oracle batches from a
  separate process; source weights remain post-hoc witness material. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-clean-centered-gradfit-oracle-only/report.json`,
  pass `true`, raw oracle calls `85649`, R2 `0.9953413995342039`, MAE
  `0.0011220708891648541`, worst error `0.01246827271765028`, subspace min
  principal cosine `0.9999966163799717`, basis moment `centered`, fit mode
  `batched-cuda`, active-gradient fit count `256`, and access audit
  `solverAccess = oracle-only`, `weightsUsedBySolver = false`,
  `hooksUsedBySolver = false`, `postHocWitnessOnly = true`.
- first finite-precision selected 8-unit SwiGLU block recovery on real local
  Mistral-family 4096D units. The oracle deterministically quantizes every
  scalar response to step `1e-8`; the solver still receives only quantized
  scalar `f(x)` batches and uses centered forward-coordinate subspace recovery
  plus active-gradient fit witnesses. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-q00000001-centered-gradfit-oracle-only/report.json`,
  pass `true`, raw oracle calls `85649`, R2 `0.9954504023764993`, MAE
  `0.0011104610821356987`, worst error `0.012667784692725533`, subspace min
  principal cosine `0.9999724425295571`, quantization step `1e-8`, basis
  moment `centered`, fit mode `batched-cuda`, and active-gradient fit count
  `256`. Coarser Mistral quantization is not promoted by this artifact.
- first deterministic-random finite-precision 8-unit SwiGLU block sweep on real
  local Mistral-family 4096D units with the same scalar response quantization
  step `1e-8`, centered basis recovery, active-gradient fit witnesses, and
  oracle-only access invariant. Current aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-q00000001-centered-gradfit-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `171298`, max raw oracle
  calls per case `85649`, min R2 `0.9957594628054579`, max MAE
  `0.0010088005020659832`, max worst error `0.009621486908707531`, and min
  subspace principal cosine `0.9999828597286717`.
- first higher-query finite-precision selected 8-unit SwiGLU block recovery on
  real local Mistral-family 4096D units at scalar response quantization step
  `1e-7`. This gate uses a larger `0.08` finite-difference step and `25`
  probe points; the lower-query `17`-probe attempt is not promoted. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-q0000001-centered-gradfit-step008-probe25-oracle-only/report.json`,
  pass `true`, raw oracle calls `118425`, R2 `0.9963343164135441`, MAE
  `0.0010131811501723571`, worst error `0.008584273469236195`, subspace min
  principal cosine `0.9999930621807005`, quantization step `1e-7`, basis
  moment `centered`, and active-gradient fit count `256`.
- first deterministic-random higher-query finite-precision 8-unit SwiGLU block
  sweep on real local Mistral-family 4096D units at scalar response
  quantization step `1e-7`. It uses the same `25` probe points and `0.08`
  finite-difference step. Current aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-q0000001-centered-gradfit-probe25-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `236850`, max raw oracle
  calls per case `118425`, min R2 `0.995860081804988`, max MAE
  `0.0010036287991970745`, max worst error `0.010330999089242423`, and min
  subspace principal cosine `0.9999857710403501`.
- first selected multi-step finite-precision 8-unit SwiGLU block recovery on
  real local Mistral-family 4096D units at scalar response quantization step
  `1e-6`. This uses the new multi-step finite-difference estimator with steps
  `0.08,0.12`. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8atom-forward-q000001-centered-gradfit-multistep0812-probe25-oracle-only/report.json`,
  pass `true`, raw oracle calls `220850`, R2 `0.9952414211113175`, MAE
  `0.0011636592796313008`, worst error `0.009295537729115112`, subspace min
  principal cosine `0.9993536447338561`, quantization step `1e-6`, and
  finite-difference steps `0.08,0.12`. This artifact alone is selected
  evidence; randomized `1e-6` robustness is reported separately below.
- first deterministic-random multi-step finite-precision 8-unit SwiGLU block
  sweep on real local Mistral-family 4096D units at scalar response
  quantization step `1e-6`. This uses multi-step finite differences
  `0.08,0.12`, increases the basis witness budget to `33` probe points, and
  reduces noisy gradient-fit weight to `0.075`.
  Current aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-q000001-centered-gradfit-multistep0812-probe33-grad0075-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `581508`, max raw oracle
  calls per case `290754`, min R2 `0.9958676370633143`, max MAE
  `0.0009856537193232476`, max worst error `0.014497109404921407`, and min
  subspace principal cosine `0.9994407684636628`.

Finite-precision interpretation and boundaries:

- These gates scale the input/oracle dimension to real Mistral-family 4096D
  hidden vectors, while the recovered block size is still a selected 8-unit
  SwiGLU scalar sum. They do not claim all-channel or full-FFN extraction.
- The `1e-8`, `1e-7`, and `1e-6` finite-precision gates are deterministic-random
  two-case sweeps. The `1e-6` sweep requires the larger `33`-probe witness
  budget; the lower `25`-probe setting is only selected-pass evidence.
- The observed finite-precision ladder degrades gracefully rather than
  cliff-failing under the gated cases: randomized `1e-8` min R2
  `0.9957594628054579`, randomized `1e-7` min R2 `0.995860081804988`, and
  randomized `1e-6` min R2 `0.9958676370633143`. This is benchmark-scoped evidence
  of noise-tolerant oracle-only recovery, not a universal API guarantee.
- The `1e-6` result required the multi-step finite-difference estimator; naive
  single-step `1e-6` attempts are not promoted by the release evidence.
- The finite-precision ladder now has a central audit command:
  `npm run benchmark:nn-live-blind-swiglu-finite-precision-ladder`. It reads
  existing clean, `1e-8`, `1e-7`, and `1e-6` selected/randomized Mistral
  oracle-only artifacts and reports R2, MAE, worst-error, subspace-cosine, and
  selected-vs-clean deltas in
  `benchmarks/release/nn-live-blind-swiglu-finite-precision-ladder/mistral-4096d-8unit-selected-randomized-ladder/report.json`.
  This is a measurement/ledger gate, not another hyperparameter-tuning claim.
- The live SwiGLU lane now has a real-model parity audit command:
  `npm run benchmark:nn-live-blind-swiglu-real-model-parity`. It reads existing
  Mistral-family 4096D and LLaMA-family 3072D selected/randomized 8-unit
  oracle-only artifacts, including finite-precision rows, and writes
  `benchmarks/release/nn-live-blind-swiglu-real-model-parity/mistral-4096d-llama-3072d-8unit-oracle-only/report.json`.
  This keeps the roadmap honest: the lane is not merely synthetic, but it is
  still selected-block recovery rather than full LLM FFN extraction.
- The live SwiGLU lane now has a deep multi-layer atlas command:
  `npm run benchmark:nn-live-blind-swiglu-deep-atlas`. It reads all current
  real-model oracle-only Mistral/LLaMA-family SwiGLU decompiler artifacts and
  builds a model/layer/unit/noise geometry atlas in
  `benchmarks/release/nn-live-blind-swiglu-deep-atlas/mistral-llama-multilayer-swiglu-atlas/report.json`.
  Current witness: `146` source reports, `127` selected rows, `19` aggregate rows,
  `60` layer-atlas rows, LLaMA-family `28/28` layers, Mistral-family `32/32`
  layers, early/middle/late buckets fully covered for both families, global min
  R2 `0.9951436177161573`, global max worst error `0.12080532366266805`, and
  global min subspace cosine `0.9992840966989263`. Companion command
  `npm run benchmark:nn-live-blind-swiglu-deep-atlas:backfill` audits the
  configured full-layer fill and refuses missing LLaMA/Mistral layers. Current
  backfill witness was a same-code deterministic configured-layer rerun:
  `mode=all`, requested `60`, passed `60`, skipped `0`, coverage complete
  `true`, refreshed atlas pass `true`, and no failed cases. The default command
  remains missing-layer gap fill/audit; use
  `GCL_ATLAS_BACKFILL_MODE=all GCL_ATLAS_BACKFILL_FORCE=1 npm run benchmark:nn-live-blind-swiglu-deep-atlas:backfill`
  when a full configured-layer regeneration witness is required.
  This is a real full configured-layer atlas of selected SwiGLU blocks, not
  all-channel FFN extraction or full-brain LLM extraction.
- first query-budget randomized 8-unit SwiGLU block sweep on real local
  Mistral-family 4096D units using forward-coordinate witnesses and explicit
  fit restart budgeting. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `162594`, max raw oracle
  calls per case `81297`, min R2 `0.9965490748792153`, max MAE
  `0.0008981839822401561`, max worst error `0.010140033299851102`, and min
  subspace principal cosine `0.9999975326762489`.
- first query-budget selected 8-unit SwiGLU block recovery on real local
  LLaMA-family 3072D units using forward-coordinate witnesses, active-gradient
  fit witnesses, and batched CUDA multi-start fitting. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-budgeted-oracle-only/report.json`,
  pass `true`, raw oracle calls `68241`, R2 `0.9999913818259203`, MAE
  `0.0017664391250281214`, worst error `0.016726727075836223`, and subspace min
  principal cosine `0.9999919539106246`.
- first finite-precision selected 8-unit SwiGLU block recovery on real local
  LLaMA-family 3072D units where the oracle deterministically quantizes every
  scalar response to step `0.00025`. The solver stays oracle-only, uses
  forward-coordinate witnesses with raw second-moment basis recovery, and fits
  only the quantized function values after subspace recovery. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-8atom-forward-q00025-nogradfit-oracle-only/report.json`,
  pass `true`, raw oracle calls `213793`, R2 `0.9995976100281595`, MAE
  `0.011652554395835905`, worst error `0.12080532366266805`, and subspace min
  principal cosine `0.9994415739539657`.
- first deterministic-random finite-precision 8-unit SwiGLU block sweep on real
  local LLaMA-family 3072D units with the same scalar response quantization step
  `0.00025` and oracle-only access invariant. Current aggregate artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-8unit-q00025-forward-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `427586`, max raw oracle
  calls per case `213793`, min R2 `0.9995742116828891`, max MAE
  `0.010256079823764875`, max worst error `0.07837874541579404`, and min
  subspace principal cosine `0.9992840966989263`.
- first query-budget randomized 8-unit SwiGLU block sweep on real local
  LLaMA-family 3072D units using forward-coordinate witnesses, active-gradient
  fit witnesses, and batched CUDA multi-start fitting. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-randomized-2block-8unit-forward-coordinate-sweep-oracle-only/report.json`,
  pass `true`, cases `2/2`, total raw oracle calls `136482`, max raw oracle
  calls per case `68241`, min R2 `0.9999781080257685`, max MAE
  `0.002327151614302204`, max worst error `0.02565795362142126`, and min
  subspace principal cosine `0.999936352892748`.
- first selected 16-unit layer-expansion parity gate on real local
  LLaMA-family 3072D units using the same oracle-only delegated sweep verifier.
  Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-layer-expansion/llama-layer20-16unit-expansion-oracle-only/report.json`,
  pass `true`, layer `20`, active rank `32`, unitCount `16`, raw oracle calls
  `129953`, R2 `0.996028807883924`, MAE `0.013147489844684133`, worst error
  `0.1197387336033246`, subspace min principal cosine
  `0.9996660748268255`, and oracle-only access invariant.
- randomized structure-gated 16-unit recovery on two real local LLaMA-family
  3072D scalar SwiGLU block oracles. The recovery process receives only batch
  `f(x)` values; source weights are requested only after recovery for the
  paired-atom witness. Current artifact:
  `benchmarks/release/nn-live-blind-swiglu-layer-expansion/llama-3072d-randomized-2block-16unit-structure-gated-residual-block-oracle-only/report.json`,
  pass `true`, cases `2/2`, min R2 `0.9993531509315209`, max MAE
  `0.005235521283167137`, max worst error `0.04507800353440916`, min subspace
  principal cosine `0.9999482868797673`, min one-to-one paired gate/up joint
  cosine `0.9942495918505425`, paired atoms above `0.99` `16/16` in both
  cases, total raw oracle calls `284482`, and oracle-only access invariant.
  The gate-norm/tail-responsibility weak-block selector is source-blind; the
  Hungarian source assignment is post-hoc evidence and does not enter fitting.
- process-separated opaque-oracle cold-start recovery for one real local
  Mistral-family 4096D 16-atom scalar block. The teacher manifest is delivered
  only to the oracle process through a private inherited descriptor. The
  dedicated solver receives no model/layer/unit/output-channel locator and no
  truth command; it uses a hash-chained eval-only binary Unix protocol. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-isolated-cold-start/mistral4096d-layer15-16unit-q000001-opaque-oracle/report.json`,
  pass `true`, R2 `0.9995787659698843`, all `16/16` post-hoc pairs above `0.99`,
  raw calls `1279238/1500000`, matching solver/server transcript digest,
  immutable frozen recovery, verifier after solver exit, and a complete
  hash-stamped file-access trace with no model/weight/control-file matches.
- cross-family confirmation that the opaque solver protocol is not tuned to one
  model family. The same protocol recovers a Mistral-family 4096D block through
  a `down-proj-output-channel` oracle and a LLaMA-family 3072D block through a
  `selected-unit-sum` oracle, with no solver-side protocol change. Current
  artifact:
  `benchmarks/release/nn-live-blind-swiglu-isolated-cold-start-court/mistral4096d-llama3072d-16unit-q000001-opaque-oracle/report.json`,
  pass `true`, cases `2/2`, min R2 `0.9995787659698843`, max MAE
  `0.0003064998140072596`, min paired joint cosine `0.9972788963034496`,
  `16/16` pairs above `0.99` per case, max raw calls per case
  `1279238/1500000`, total raw calls `2298770`, and a per-case file-access
  witness of at least `3101` lines with no model/weight or control-file
  matches. Each solver ran in a private randomized work directory, so no output
  path carried a model or layer name.

Measured boundary on block selection: the randomized stratified court
(`benchmark:nn-live-blind-swiglu-randomized-cold-start-court`, master seed
`260729`) runs the same opaque protocol on deterministically randomized blocks
across early/middle/late layer bands in both families and passes only `1/6`.
The failures are attributed, not blanket: `structure-not-recovered` 1,
`recovered-but-inaccurate` 2, `recovered-but-low-r2` 1, and
`recovered-but-worst-error-over-bar` 1. So the opaque cold-start recovery is
family-independent but not selection-independent, and the passing courts above
must be read as evidence for their configured blocks rather than for arbitrary
blocks. That lane is deliberately kept visible as BLOCKED.

Boundary: this is not yet full GPT-2 FFN extraction, arbitrary 3072-channel
blind decompilation, arbitrary all-channel 4096D production block
decompilation, an OS sandbox against a malicious same-UID process, or a
universal SOTA claim.

### 3. Provenance Rail

Purpose: answer "which neuron is this, where did it come from, and why does it
react" when the source side is available.

Current status: release-gated provenance-mode verification maps recovered GELU
atoms to exact source ids, layers, indices, duplicate source groups, and
why-reacts evidence. The local transformer MLP hook gate extends this from a
manifest-only controlled FFN source into captured activation rows from a local
transformer-style block. The real GPT-2 source-extractor gate extends the same
provenance rail to a compact top-20 source graph from actual GPT-2 layer-0 FFN
weights when the source manifest is available.

Boundary: exact source identity requires source manifests, hooks, weights, or
equivalent signed provenance. Output-only access can recover canonical
functional structure, but it cannot honestly prove original training-time
neuron identity without extra observability.

### 4. Live LLM Oracle Circuit Rail

Purpose: move from controlled FFN/source-structure gates into a
model-agnostic live LLM oracle layer that can run the same graph-discovery
protocol across transformer families. GPT-2 is the first fast proof gate, not
the endpoint.

Current status: the CLI can load real GPT-2 through the project-managed Python
runtime, run a real forward pass, hook MLP activations, ablate selected neurons,
and run negative-control ablations. The current local proof for
`The capital of France is` records 50 random-active controls, 50 random-inactive
controls, 50 off-prompt controls, same-neuron other-position controls, and a
cross-family overlap audit.

The first automatic graph-level gate now exists:
`npm run benchmark:nn-live-gpt2-fact-circuit-finder`. It loads real GPT-2,
scans configured factual prompts across selected layers/top MLP neurons, builds
token/neuron/target graph artifacts, and validates each selected graph through
joint causal ablation against graph-level controls. The current release artifact
is
`benchmarks/release/nn-live-gpt2-fact-circuit-finder/gpt2-50-facts-graph-layer-scan/report.json`:
50 valid facts, strict case definition `graphGate.pass && hardeningGate.pass`,
50 strict-passed cases, graph-passed cases `50`, hardening-passed cases `50`,
pass rate `1.0`, median joint graph target-logit drop
`0.25433349609375`, p10 graph drop `0.1052703857421875`, median
graph-control p95 positive target-logit drop `0.06884765625`, median graph
lift over control p95 `3.8081556082148498`, mean selected graph node count
`1.6`, and 46 unique selected graph neurons. Stage-I hardening passed with
paraphrase pass rate `1.0`, wrong-answer specificity `1.0`, negative-fact
specificity `1.0`, and unrelated-logit preservation `0.955`. The benchmark
uses primary layers `[6,9,11]` and adaptive fallback to `[0..11]` on strict
primary failures; the current report used fallback on Russia only. The current
artifact was generated with the explicit GPU command and records
`model.device = cuda`.

Boundary: hook reading, activation ablation, logit-delta scoring, and negative
controls are known mechanistic-interpretability measurement techniques. They
are not the GCL invention by themselves. In this rail they are the measurement
layer for GCL-native circuit discovery. The current graph finder is a
benchmark-scoped MLP circuit finder for the configured GPT-2 factual prompt set
and proves the first adapter gate only. The roadmap must not treat GPT-2
layer names, tokenizer behavior, GELU FFN shape, or prompt quirks as the
framework contract. It is not attention-head/path recovery, full GPT-2 FFN
extraction, arbitrary LLM understanding, or output-only original-neuron
identity.

The first attention-head/path gate now exists:
`npm run benchmark:nn-live-gpt2-attention-path-circuit-finder:gpu`. It loads
real GPT-2 through the CUDA runtime lane, scans configured attention heads,
builds token/head/target path artifacts, and validates selected heads through
causal attention-path ablation above attention-head controls. Current artifact:
`benchmarks/release/nn-live-gpt2-attention-path-circuit-finder/gpt2-50-facts-attention-path-layer-scan/report.json`.
Current result: 50 valid facts, 50 passed cases, pass rate `1.0`, median
attention graph target-logit drop `1.7271881103515625`, median control p95
`0.448638916015625`, median lift `3.7378692395502666`, mean selected heads
`1.04`, unique selected heads `6`, median wrong-answer collateral ratio
`0.7162562846461261`, skipped rows `33`, and `model.device = cuda`. Skipped
rows are separated from valid pass/fail cases and currently reflect
non-single-token answers or low-confidence target baseline rank. This is path
evidence, not answer-specific factual identity.

The first combined attention+MLP graph gate now exists:
`npm run benchmark:nn-live-gpt2-combined-circuit-finder:gpu`. It loads real
GPT-2 through the CUDA runtime lane, discovers selected MLP graph nodes and
attention heads, builds one token/head/neuron/target graph, and validates the
union through simultaneous attention+MLP causal ablation above combined
controls. Current artifact:
`benchmarks/release/nn-live-gpt2-combined-circuit-finder/gpt2-50-facts-attention-mlp-combined-graph/report.json`.
Current result: 50 valid facts, 50 passed cases, pass rate `1.0`, MLP
component pass rate `1.0`, attention component pass rate `1.0`, median
combined target-logit drop `1.92523193359375`, median full-control combined
p95 `0.417144775390625`, median lift `4.333544871993313`, mean selected MLP
nodes `1.56`, mean selected attention heads `1.02`, failed subjects `[]`, and
`model.device = cuda`. Combined pass/fail uses full component-replacement
controls; MLP-swap and attention-swap controls remain diagnostic.

The first output/logit-only whole-model behavior gate now exists:
`npm run benchmark:nn-live-blackbox-llm-behavior-reader:gpu`. It loads real
GPT-2 as a local oracle but gives the reader only prompt strings, target token
ids, and next-token logits. It builds factual behavior graphs from subject
swaps, adaptive capital-template paraphrase search, answer-token specificity,
and relation-control diagnostics. Current artifact:
`benchmarks/release/nn-live-blackbox-llm-behavior-reader/gpt2-output-only-10-facts-behavior-graph/report.json`.
Current result: 10 valid facts, 10 passed cases, pass rate `1.0` with gate
threshold `1.0`, query count `240`, median target rank `2`, max target rank
`10`, median subject-specificity lift `6.850424766540527`, min
subject-specificity lift `5.2599224090576175`, median answer-specificity
margin `3.499279022216797`, min answer-specificity margin
`2.0098876953125`, median relation-template margin `3.4460906982421875`, min
relation-template margin `1.2060317993164062`, and median paraphrase
retention `1.0`. Access audit: `solverAccess =
prompt-to-next-token-logits-only`, `weightsUsedBySolver = false`,
`hooksUsedBySolver = false`, `activationsUsedBySolver = false`.

Boundary: this is the first correct black-box whole-model behavior rail. It is
not hidden-neuron identity, not hook-based circuit recovery, not source
provenance, not full FFN extraction, and not arbitrary LLM understanding. It
exists to bridge user-visible model behavior and the deeper instrumented
circuit/FFN rails without pretending that hook evidence is black-box evidence.

## Current Mainline

The next development must stay on this order:

1. Extract a model-agnostic `LLMOracleAdapter` layer:
   DONE for GPT-2 adapter parity. GPT-2 remains the fast real-model proof gate,
   but model loading, tokenizer/model identity, block/MLP lookup,
   activation-family metadata, activation ablation, and negative-control
   selection now go through a typed adapter boundary. The circuit finder must
   not hard-code GPT-2. The adapter boundary now has targeted real-model
   witnesses for GPT-2, GPT-NeoX/Pythia, LLaMA-family SwiGLU, and
   Mistral-family SwiGLU.
   Runtime acceleration is explicit and hardware-gated: CPU remains the default
   release path, while NVIDIA CUDA runs use `.gcl-python-gpu`,
   `npm run python:setup:gpu`, and
   `npm run benchmark:nn-live-gpt2-fact-circuit-finder:gpu`. The GPU setup must
   prove `torch.cuda.is_available()` and a real CUDA tensor operation before GPU
   benchmark output is trusted.
2. Expand ordered residual path evidence beyond the first GPT-2 gate:
   the first ordered attention->residual->MLP->target gate is now present via
   `npm run benchmark:nn-live-gpt2-ordered-residual-path-circuit-finder:gpu`.
   The current artifact passed 50 valid GPT-2 factual prompts with median
   selected path activation delta `0.18021255731582642`, median full-control
   path p95 activation delta `0.0318724343786016`, and median ordered path
   lift `5.259007093850701`. The same protocol now has a second real-model
   adapter preview via
   `npm run benchmark:nn-live-pythia-ordered-residual-path-circuit-finder:gpu`:
   real EleutherAI/pythia-70m-deduped / GPT-NeoX, 20 valid facts, 20 passed
   ordered path cases, pass rate `1.0`, median selected path activation delta
   `0.51171875`, and failed subjects `[]`. Deterministic per-subject
   controls and multi-component ordered rescue close the previous Japan/Israel
   hard cases without threshold weakening. DONE for the first real
   LLaMA-family SwiGLU adapter gate: the real
   quantized Llama-3.2-3B-Instruct benchmark addresses
   `SiLU(gate_proj(x))*up_proj(x)` pair channels before `down_proj` and passes
   20/20 valid factual ordered-path cases on CUDA. Its fresh interpolated
   summary reports p10 selected path delta `0.023046875`, median control-path
   p95 delta `0.0017847249284386607`, and median combined target-logit drop
   `0.9375`. DONE for the first real Mistral-family SwiGLU adapter gate: the
   quantized Mistral-7B-Instruct benchmark addresses the same real
   `SiLU(gate_proj(x))*up_proj(x)` unit, passes 10/10 valid factual
   ordered-path cases, and records median selected path delta `0.203125`.
   DONE for the broadened ordered-path stability replay across GPT-2, Pythia,
   Llama, and Mistral: 4/4 model reports pass, with GPT-2/Pythia checking 5
   cases each, Llama/Mistral checking 4 cases each, 3 paraphrase probes per
   case, and layer shifts `[-2,-1,1,2]` with at least 2 valid layer controls.
   The next move is an explicit OOD paraphrase stress lane plus cleaner
   extraction of model-agnostic circuit-finder logic, not GPT-2-only special
   cases.
3. Keep the current live-reader/fact-finder runtime behind that adapter:
   prompt/token/layer/component/intervention -> logits, activations, logit
   deltas, controls, cache keys, runtime warnings, model/tokenizer identities,
   and query-budget records. This must keep using real model runtimes, not
   synthetic teachers and not fitted surrogates.
4. Expand the new output/logit-only behavior rail beyond GPT-2:
   same `prompt -> logits` interface, same access audit, larger factual sets,
   paraphrase/OOD prompt families, and then Pythia/LLaMA/Mistral parity. The
   solver must not receive hooks, hidden activations, weights, source ids, or
   intervention handles in this rail.
5. Continue blind real GPT-2 top-K FFN recovery and source-structure scaling
   only when it feeds the circuit finder or verifies source/provenance for
   circuits the live oracle selected.
6. Continue the 768D -> 4096D -> 16384D scaling ladder as a support lane for
   active-subspace structure recovery, not as the center of the product.

Do not make 7D/8D ReLU scaling the center of the project. ReLU ND work remains
important only when it improves the probing, adjacency, query-budget, or
failure-generation machinery needed by the smooth FFN decompiler.

Do not make another single-prompt hook-and-ablate report the center of the
project. The live LLM reader is useful only when it feeds graph-level circuit
discovery, graph stability tests, and path-level interventions.

## Proof Standard

A promoted neural-decompiler claim needs all of these:

- held-out MAE/RMSE/R2;
- worst-case error or a clear local-failure diagnostic;
- recovered atom count and canonical structure diagnostics;
- query-budget accounting;
- baseline comparison on the same task;
- failure/negative-case accounting;
- artifact paths under `benchmarks/release/...`;
- documentation of what the result does not prove.

Output fit alone is not enough for a structure claim.

## Non-Negotiable Boundaries

Do not claim:

- broad SOTA without a defined benchmark axis and strong baselines;
- "original neuron identity" from output-only access;
- full LLM block decompilation from controlled synthetic gates;
- scratch output as release evidence;
- a surrogate as a structure finder.
- instrumented source extraction as proof of output-only decompilation.
- single-prompt live GPT-2 hook/ablation instrumentation as proof of automatic
  circuit discovery.
- the current MLP-only fact-circuit graph gate as proof of attention-head/path
  recovery, full GPT-2 FFN extraction, arbitrary LLM understanding, or
  output-only original-neuron identity.
- GPT-2-specific implementation as proof that the circuit finder already scales
  to GPT-Neo, Pythia, LLaMA, Mistral, or arbitrary LLM architectures.

If a benchmark exposes a weakness, the weakness becomes a gate, a negative
case, or a visible non-promoted limitation. It must not be hidden by easier
settings.
