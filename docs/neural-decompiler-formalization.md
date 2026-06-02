# Neural Decompiler Formalization

This document defines the current scientific claim surface for the GCL neural
decompiler work. It is intentionally stricter than the implementation: if a
benchmark or report cannot satisfy these definitions, it must use weaker
language. The product direction is summarized in `docs/gcl-north-star.md`;
this document defines the terms and claim levels used to keep that direction
honest.

## Current Classification

The current system should be described as:

```text
Black-box canonical neural structure recovery under functional equivalence classes.
```

Short form:

```text
Structure-identifiable neural decomposition framework.
```

This is stronger than a plain output surrogate when a gate returns explicit
atoms and passes canonical invariance checks. It is still weaker than a true
semantic hidden-neuron decompiler.

## Access Model

The strict output-only setting is:

```text
oracle f: R^d -> R
query x -> f(x)
no hidden activations
no weights
no gradients unless estimated from oracle calls
```

Any lane that uses hidden activations, weights, or teacher metadata must state
that extra observability explicitly.

## Core Definitions

### Atom

An atom is one recovered functional component from an activation family. For the
current GELU FFN lane, an atom has the form:

```text
a * gelu(w dot x + b)
```

with direction `w`, input bias `b`, output scale `a`, and optional global
offset handled outside the atom list.

For the current clean SwiGLU FFN lane, the recovered unit is a value/gate pair:

```text
a * (u dot x + bu) * silu(g dot x + bg)
```

with value direction `u`, value bias `bu`, gate direction `g`, gate bias `bg`,
output scale `a`, and optional global offset handled outside the pair list.
This pair is a structure object, not only a local output predictor.

Two atoms can be equivalent up to transformations that preserve the same oracle
function under the benchmark's activation family and tolerance. Permutation of
atoms, duplicate splitting, and exact cancellation are representation changes,
not necessarily different observable functions.

### Functional Equivalence

Two candidate decompositions `g1` and `g2` are functionally equivalent on a
domain `D` when:

```text
E_{x in D}[abs(g1(x) - g2(x))] <= epsilon_mean
max_{x in H}[abs(g1(x) - g2(x))] <= epsilon_max
```

where `H` is a held-out oracle sample set not used for fitting. A release claim
must report the tolerances, sampling distribution, and query budget.

### Canonical Structure

A canonical structure is the simplest stable recovered atom set that remains
functionally equivalent to the oracle under the declared validation protocol.
For current GCL gates, "simplest" means:

1. lower effective atom count wins when output fidelity remains inside the
   allowed tolerance;
2. duplicate, near-duplicate, weak, and cancelling atoms are merged or pruned
   only when held-out oracle validation still passes;
3. canonical signatures are sorted and sign-normalized so permutation and
   harmless rescaling do not create false differences.

This is an Occam prior under oracle equivalence. It is not a proof that the
original teacher used the same hidden neurons.

### Canonical Signature

A canonical signature is a sorted summary of the recovered atom set:

```text
effective_atom_count
normalized_directions
normalized_biases
normalized_output_scales
duplicate_count
near_parallel_count
weak_atom_count
heldout_output_metrics
query_budget
```

Two equivalent teachers pass a cross-decomposition invariance gate only if their
canonical signatures remain close under the configured signature distance and
atom matching thresholds.

### Structure Finder vs Surrogate

A structure finder must return explicit atoms, effective channel count, atom
diagnostics, held-out output metrics, and query budget.

A surrogate only needs to approximate the output. A surrogate may be useful, but
it cannot be used as proof of neuron/block structure.

## Loss And Metrics

Every promoted neural decompiler report must separate these measurements:

| Metric | Purpose |
|---|---|
| MAE / RMSE / R2 | Functional output fidelity on held-out oracle samples. |
| Worst-case error | Detects local failure hidden by average metrics. |
| Direction projection | Measures whether recovered atom directions align with teacher directions when teacher directions are available. |
| Bias and scale error | Measures approximate parameter recovery when identifiable. |
| Effective atom count | Measures whether the solver recovered the intended canonical rank instead of an overcomplete basis. |
| Signature distance | Measures invariance across equivalent hidden decompositions. |
| Model-agreement MAE | Measures whether two independently recovered canonical structures compute the same function. |
| Raw oracle calls | Makes query budget part of the result, not an afterthought. |

Output metrics alone are insufficient for a structure claim.

## Claim Levels

### Level 0: Output Surrogate

Allowed when GCL approximates `f(x)` well on held-out samples but does not return
trusted atoms.

Allowed wording:

```text
active-subspace surrogate
output-equivalent approximation
```

### Level 1: Atom Recovery

Allowed when GCL recovers explicit atoms and parameter diagnostics under a
controlled identifiable teacher.

Allowed wording:

```text
controlled GELU atom recovery
explicit FFN atom structure finder
```

### Level 2: Canonical Structure Under Equivalence

Allowed when equivalent reparameterizations recover the same canonical
signature, with held-out oracle agreement and failure accounting.

Allowed wording:

```text
canonical structure recovery under oracle equivalence
canonicalized neural basis recovery
```

### Level 2.5: Instrumented Source-Neuron Provenance

Allowed when GCL first recovers explicit atoms and then verifies those atoms
against a source-neuron manifest, weight hook, activation hook, or equivalent
signed provenance record. The verifier may claim exact source ids, layers,
indices, duplicate source groups, and why-reacts evidence only for the supplied
instrumented source.

Allowed wording:

```text
source-neuron provenance verification
instrumented source-neuron mapping
why-reacts evidence for mapped source neurons
```

This is not an output-only claim. If the manifest or hook is removed, the claim
downgrades to Level 1 or Level 2 depending on the available gate.

### Level 3: Semantic Hidden-Neuron Identity

Not currently proven for output-only GCL lanes.

This would require either extra observability or a gated identifiability result
showing that no other hidden basis can generate the same oracle behavior under
the stated assumptions.

### Level 4: Production LLM Block Decompilation

Not currently proven.

This requires arbitrary high-dimensional transformer FFN/block tests, noisy and
quantized APIs, strong external baselines, realistic query budgets, and failure
cases where GCL is allowed to say "not identifiable" instead of returning a fake
structure.

## What Is Not Proven Yet

The current release-gated GELU/SwiGLU structure work does not prove:

- original hidden-neuron semantic identity from output-only access;
- uniqueness of the original teacher basis;
- an invertible mapping from oracle output back to training-time weights;
- full GPT-2 FFN extraction;
- arbitrary 3072-channel FFN decompilation;
- arbitrary 4096D production LLM block decompilation;
- noisy/cancellation-heavy mixed GELU or SwiGLU block recovery;
- broad SOTA status.

The provenance verifier separately proves benchmark-scoped source-neuron
mapping only when the source manifest or equivalent instrumentation is supplied.
That is a stronger engineering mode than a surrogate, but it is not evidence
that output-only access can recover original neurons in arbitrary networks.

## Required Negative Cases

Negative cases are not optional. A serious structure finder must know when it
cannot identify structure.

Each negative-case gate must record whether the solver:

1. correctly recovers a canonical structure;
2. correctly downgrades to a surrogate;
3. correctly reports non-identifiability;
4. fails by inventing unsupported structure.

The fourth outcome must be a hard failure for any structure claim.

Required negative-case families:

| Family | Failure Mode To Detect | Expected Correct Behavior |
|---|---|---|
| Exact duplicate split | One true atom represented as several identical atoms. | Collapse to one effective atom or report equivalence class. |
| Equal/opposite cancellation | Hidden atoms cancel in output. | Do not claim semantic hidden-neuron identity. |
| Near-collinear directions | Multiple bases fit the same oracle similarly. | Prefer minimum equivalent structure or report ambiguity. |
| Weak channel below noise floor | Real atom has output below measurable noise. | Downgrade or mark weak/non-identifiable. |
| Overcomplete high-rank budget | Solver is allowed more atoms than needed. | Select the simplest held-out-valid structure. |
| Mis-specified activation family | Teacher is not in the assumed GELU/SwiGLU family. | Return surrogate or fail the structure claim. |
| Adversarial quantization | Quantization erases small structural differences. | Report non-identifiability instead of hallucinating atoms. |

Current executable gate:

```bash
npm run benchmark:nn-gelu-ffn-negative-cases
```

Current coverage: duplicate/cancellation/weak canonicalization,
weak-under-noise non-identifiability, weak-but-resolvable channel recovery,
activation-family mismatch rejection, and minimum equivalent structure under
overcomplete budgets. The weak-channel pair is intentional: below the noise
floor the correct result is a smaller observable structure, while above the
noise floor the gate fails unless the third channel is recovered. Near-collinear
ambiguity under noisy mixed blocks and adversarial quantization still need
broader follow-up gates.

## Scaling Stress Ladder

Scaling must be measured separately from correctness. A high-dimensional pass is
not meaningful if it only works because the true active rank is tiny and clean.

Required ladder:

1. `768D`: GPT-2-width synthetic and fixture-backed stress.
2. `4096D`: LLM-width synthetic stress with explicit query budget.
3. `16384D`: sparse active-subspace stress, not brute-force sampling.
4. Multi-layer smooth composition: distinguish active subspace from true
   compositional structure.
5. Noise + quantization: repeated-query aggregation, worst-case error, and
   non-identifiability outcomes.

Promotion rule:

```text
Do not promote a dimension jump unless held-out fidelity, structure diagnostics,
negative-case behavior, and query budget all pass together.
```

## Review Standard

A reviewer should be able to answer these questions from every promoted report:

1. What oracle access was available?
2. What exactly is an atom in this lane?
3. Which equivalences are intentionally canonicalized away?
4. What loss function and holdout set were used?
5. What did GCL recover structurally, not only functionally?
6. What negative cases were tested?
7. When does the system downgrade to surrogate or non-identifiable?
8. How many oracle calls were required?
9. Which baselines were compared?
10. Which claims remain forbidden?
