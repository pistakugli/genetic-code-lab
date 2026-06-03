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

Boundary: this is not yet full GPT-2 FFN extraction, arbitrary 3072-channel
blind decompilation, arbitrary 4096D production block decompilation, or a
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

Purpose: move from controlled FFN/source-structure gates into a real model
reader that treats GPT-2 as a live oracle and automatically discovers causal
circuits for prompts and facts.

Current status: the CLI can load real GPT-2 through the project-managed Python
runtime, run a real forward pass, hook MLP activations, ablate selected neurons,
and run negative-control ablations. The current local proof for
`The capital of France is` records 50 random-active controls, 50 random-inactive
controls, 50 off-prompt controls, same-neuron other-position controls, and a
cross-family overlap audit.

Boundary: hook reading, activation ablation, logit-delta scoring, and negative
controls are known mechanistic-interpretability measurement techniques. They
are not the GCL invention by themselves. In this rail they are the measurement
layer that feeds a GCL-native automatic circuit finder. A result in this rail
may not be promoted as "GCL understands GPT-2" until the system selects the
circuit automatically, validates it across prompt templates and negative facts,
and emits a graph with causal evidence and query-budget accounting.

## Current Mainline

The next development must stay on this order:

1. Build the live GPT-2 oracle control plane:
   prompt/token/layer/component/intervention -> logits, activations, logit
   deltas, controls, cache keys, and query-budget records. This must use the
   real model runtime, not synthetic teachers and not a fitted surrogate.
2. Build the automatic circuit finder:
   from a target such as `France -> Paris`, GCL must search candidate layers,
   token positions, MLP neurons, and later attention heads without a human
   hand-picking the neuron list. It must rank nodes by causal effect,
   control-normalized effect, prompt-template stability, and graph minimality.
3. Add the first `nn-live-gpt2-circuit-finder` release candidate gate:
   multiple factual prompts, paraphrases, negative facts, random-active,
   random-inactive, off-prompt, and other-position controls, with a graph
   artifact under `benchmarks/release/...`.
4. Continue blind real GPT-2 top-K FFN recovery and source-structure scaling
   only when it feeds the circuit finder or verifies source/provenance for
   circuits the live oracle selected.
5. Continue the 768D -> 4096D -> 16384D scaling ladder as a support lane for
   active-subspace structure recovery, not as the center of the product.

Do not make 7D/8D ReLU scaling the center of the project. ReLU ND work remains
important only when it improves the probing, adjacency, query-budget, or
failure-generation machinery needed by the smooth FFN decompiler.

Do not make another hook-and-ablate report the center of the project. The live
LLM reader is useful only if it becomes an oracle interface for automatic
circuit discovery.

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
- live GPT-2 hook/ablation instrumentation as proof of automatic circuit
  discovery before the circuit finder exists and passes its gate.

If a benchmark exposes a weakness, the weakness becomes a gate, a negative
case, or a visible non-promoted limitation. It must not be hidden by easier
settings.
