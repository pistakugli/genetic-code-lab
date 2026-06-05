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
Current result: 50 valid facts, 48 passed cases, pass rate `0.96`, median
attention graph target-logit drop `3.77752685546875`, median control p95
`0.8765945434570312`, median lift `4.1454169816074575`, mean selected heads
`3`, unique selected heads `20`, median wrong-answer collateral ratio
`0.7703779014285085`, and `model.device = cuda`. This is path evidence, not
answer-specific factual identity.

## Current Mainline

The next development must stay on this order:

1. Extract a model-agnostic `LLMOracleAdapter` layer:
   DONE for GPT-2 adapter parity. GPT-2 remains the fast real-model proof gate,
   but model loading, tokenizer/model identity, block/MLP lookup,
   activation-family metadata, activation ablation, and negative-control
   selection now go through a typed adapter boundary. The circuit finder must
   not hard-code GPT-2. The next adapter targets are GPT-Neo/Pythia-style
   HuggingFace models and then LLaMA/Mistral-style SwiGLU models.
   Runtime acceleration is explicit and hardware-gated: CPU remains the default
   release path, while NVIDIA CUDA runs use `.gcl-python-gpu`,
   `npm run python:setup:gpu`, and
   `npm run benchmark:nn-live-gpt2-fact-circuit-finder:gpu`. The GPU setup must
   prove `torch.cuda.is_available()` and a real CUDA tensor operation before GPU
   benchmark output is trusted.
2. Compose attention-head and MLP graph edges:
   the first attention-head/path gate is now present. The next move is to join
   attention path edges with the existing MLP graph nodes into a single
   token/head/neuron/target circuit graph, while keeping controls and claim
   boundaries separate for broad attention effects versus answer-specific MLP
   effects.
3. Keep the current live-reader/fact-finder runtime behind that adapter:
   prompt/token/layer/component/intervention -> logits, activations, logit
   deltas, controls, cache keys, runtime warnings, model/tokenizer identities,
   and query-budget records. This must keep using real model runtimes, not
   synthetic teachers and not fitted surrogates.
4. Continue blind real GPT-2 top-K FFN recovery and source-structure scaling
   only when it feeds the circuit finder or verifies source/provenance for
   circuits the live oracle selected.
5. Continue the 768D -> 4096D -> 16384D scaling ladder as a support lane for
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
