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

GCL now has three evidence rails.

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
- source-neuron provenance verification when source manifests or equivalent
  instrumentation are provided.

Boundary: this is not yet full GPT-2 FFN extraction, arbitrary 3072-channel
decompilation, arbitrary 4096D production block decompilation, or a universal
SOTA claim.

### 3. Provenance Rail

Purpose: answer "which neuron is this, where did it come from, and why does it
react" when the source side is available.

Current status: release-gated provenance-mode verification maps recovered GELU
atoms to exact source ids, layers, indices, duplicate source groups, and
why-reacts evidence.

Boundary: exact source identity requires source manifests, hooks, weights, or
equivalent signed provenance. Output-only access can recover canonical
functional structure, but it cannot honestly prove original training-time
neuron identity without extra observability.

## Current Mainline

The next development must stay on this order:

1. noisy/quantized mixed GELU block recovery;
2. package-backed baseline court for mixed structure recovery;
3. SwiGLU mixed-block structure recovery;
4. real local transformer MLP hook with captured activations and provenance;
5. scaling stress ladder: 768D -> 4096D -> 16384D;
6. GPT-2/full FFN research stress only after the generic mixed-block lane is
   strong enough.

Do not make 7D/8D ReLU scaling the center of the project. ReLU ND work remains
important only when it improves the probing, adjacency, query-budget, or
failure-generation machinery needed by the smooth FFN decompiler.

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

If a benchmark exposes a weakness, the weakness becomes a gate, a negative
case, or a visible non-promoted limitation. It must not be hidden by easier
settings.
