# GCL Claim Policy & Reporting Standards

This document establishes the official terminology, validation rules, and reporting policies for all Genetic Code Lab (GCL) development, testing, and academic publications. The goal is to enforce strict scientific integrity and prevent hyperbole or selective bias.

## 1. Classification of Project Stages

Every feature, benchmark, or model in GCL must be explicitly categorized into one of these four categories in all reports and documentation:

1.  **PROVEN (Stable Core)**:
    *   *Definition*: Systematically passes non-regression gates, has deterministic or highly stable statistical convergence, and outperforms basic baseline models across multiple independent seeds.
    *   *Standard*: Must have a unit test or release gate script that runs automatically on CI/CD and exits non-zero on any deviation.
2.  **PROMISING**:
    *   *Definition*: Shows clear, verified performance benefits under specific, controlled scenarios, but has known limitations, sensitivity to hyper-parameters, or convergence stability issues across seeds.
    *   *Standard*: Must document exactly which configurations or seeds fail, and the exact delta in comparison to standard baselines.
3.  **EXPERIMENTAL (Sandbox)**:
    *   *Definition*: Active research script or hypothesis testing. May crash, fail, or be heavily outperformed by basic models.
    *   *Standard*: Experimental code must remain strictly inside `scratch/` or `trails/` directories. Experimental artifacts must not be treated as release evidence unless promoted by a release gate with a manifest.
4.  **NEEDS REPAIR / BLOCKED**:
    *   *Definition*: Empirical evaluations show the approach is systematically outperformed by standard baselines (e.g. Random Forest), suffers from numerical collapse, or has extreme overfitting.
    *   *Standard*: Must be openly documented in the `evidence-map.md` with clear explanation of the blocking mechanism.

---

## 2. Terminology Restrictions (The Banned List)

To maintain a sober, engineering-focused tone, the following words and phrases are strictly banned or restricted:

| Banned Phrase | Permitted Alternative | Reason |
| :--- | :--- | :--- |
| **"SOTA" / "State of the Art"** | "High-fidelity local approximation" / "Competitive baseline" | High-dimensional GP is rarely SOTA compared to deep ensembles or large GBDTs. |
| **"Exact reverse-engineering"** | "Approximate parameter recovery" / "Structural alignment" | Stochastic optimization results in visible numerical errors ($\pm 0.01 - 0.05$). |
| **"Zero double-dipping"** | "Strict out-of-sample partition" | Double-dipping can only be claimed if full train/validation/test isolation is programmatically locked. |
| **"Ultimate proof" / "Perfect fit"** | "Verified empirical validation" | Extravagant claims harm academic and industrial credibility. |
| **"Universal solver"** | "Targeted Symbolic Regression" | GP scales poorly to hundreds of dimensions without massive resources. |

---

## 2.1 Neural Decompiler Claim Terms

Output-only neural decompiler work must distinguish functional equivalence,
canonical structure recovery, and semantic hidden-neuron identity.

| Claim Phrase | Allowed Only When | Safer Wording |
| :--- | :--- | :--- |
| **"True neuron decompiler"** | Hidden activations/weights are observable, or a gated identifiability theorem/benchmark proves uniqueness under the stated assumptions. | "oracle-only canonical structure recovery" |
| **"Canonical structure recovery"** | A release gate compares two or more functionally equivalent reparameterizations and verifies matching canonical signatures on held-out oracle samples. | "canonicalized GELU atom recovery under oracle equivalence" |
| **"Semantic hidden-neuron identity"** | The benchmark has access to the original hidden basis or proves uniqueness beyond permutation/scaling/equivalent decompositions. | "effective atom identity" or "functional atom identity" |
| **"Surrogate"** | The model is optimized for output fidelity without claiming recovered atoms or interpretable channel count. | "active-subspace surrogate" |
| **"Structure finder"** | The gate reports explicit atoms, effective channel count, duplicate/near-parallel diagnostics, held-out output metrics, and query budget. | "GELU FFN structure finder" |
| **"Opaque oracle-only neural recovery"** | The solver process receives only an eval endpoint and public shape/budget metadata; model, layer, source ids, hooks, weights, truth commands, and input snapshots are absent from solver argv/environment/protocol, and a fresh runtime access witness confirms no model/control-file reads. | "process-separated, access-attested recovery for the selected block" |

The current GELU FFN structure gates may claim controlled oracle-only GELU atom
recovery, rank-adaptive mixed-channel recovery, near-parallel clean stability,
and clean cross-decomposition canonical invariance. They may not claim full
GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D
production LLM block decompilation, noisy mixed-block recovery, SwiGLU
mixed-block recovery, or general "GCL understands every LLM neuron/block"
status.

Formal definitions for "atom", "functional equivalence", "canonical
structure", "canonical signature", negative cases, and scaling stress live in
`docs/neural-decompiler-formalization.md`. Any release report that uses those
terms must follow that document or explicitly state a narrower local definition.
The first executable negative-case enforcement gate is
`npm run benchmark:nn-gelu-ffn-negative-cases`.

---

## 2.2 Output-Only LLM Behavior Reader Terms

Whole-model black-box LLM work must be separated from hook-based
mechanistic-interpretability work.

| Claim Phrase | Allowed Only When | Safer Wording |
| :--- | :--- | :--- |
| **"Black-box LLM behavior reader"** | The solver receives only prompt/input strings, target token ids, and output logits/probabilities from an oracle. | "output/logit-only behavior graph" |
| **"Circuit finder"** | The gate uses hooks, hidden activations, attention/MLP ablations, or residual path patching and says so explicitly. | "instrumented circuit finder" |
| **"LLM understands why"** | Never as a broad claim from one benchmark. | "benchmark-scoped behavior evidence for selected factual prompts" |
| **"Original neuron identity"** | Hidden source ids, weights, hooks, or an identifiability proof are available. | "output-only target behavior evidence" |
| **"SwiGLU mechanism evidence"** | A real model's gate/value/down-projection hooks reconstruct the measured forward pass and selected channel ablations are compared with controls. | "prompt-specific instrumented SwiGLU channel effects" |

The output/logit-only behavior rail may claim prompt-intervention evidence,
subject specificity, answer specificity, paraphrase/template stability, query
budget, and model/runtime identity. It may not claim hidden-neuron identity,
source-provenance, hook-based circuit recovery, full FFN extraction, or
arbitrary LLM understanding.

The interactive next-token decision mode additionally measures
whitespace-delimited-span deletion counterfactuals against a fixed
target/competitor set. Its "support" and
"suppress" labels mean only that the measured target-vs-competitor logit margin
decreased or increased under that exact text edit. Distribution-shift metrics
and exact edited prompts must be retained with the report. These observations
do not establish an internal reasoning step, semantic feature, or unique cause.

The autoregressive `--trace` mode records a greedy token-by-token output path
and bounded text-unit deletion counterfactuals against the exact token-ID
prefix. It can show how deleting a measured prompt or generated-prefix unit
changes the next-token distribution for that run. With `--response-probes`, it
also regenerates complete greedy responses after deleting selected prompt units,
reporting the exact counterfactual input, response, and first token divergence.
With `--explanation-probes`, it also searches for a budget-bounded subset of
tokenization-aware text units that preserves each observed next-token argmax.
Chat control markers and role headers are protected. A result is called
"one-minimal" only when every remaining editable unit was individually tested;
this is not a globally smallest subset, a semantic rationale, or proof of the
model's hidden reasoning. Exact token IDs, query budget, and whether the search
exhausted that budget must remain in the report.
This establishes an output change for that precise intervention, not semantic
necessity or an internal cause: token deletion may damage formatting or alter
the prompt in unintended ways. It does not expose hidden activations, prove why
a token was generated, identify a unique reasoning chain, or generalize beyond
the measured prompt, greedy decoding policy, model, and interventions.
With `--interaction-probes`, it also measures bounded pairs of already-probed
unit deletions, prioritizing units with the largest absolute single-deletion
margin changes. The interaction residual is the joint-deletion target-vs-fixed-
competitor margin minus the margin predicted by adding the two single-deletion
effects. This detects non-additivity for those exact edits; it is not evidence
of an internal circuit, semantic relation, or complete reasoning trace.
With `--branch-probes`, it forces selected non-greedy next-token candidates and
then resumes greedy decoding to measure their alternative continuations. These
are output-level decision counterfactuals for the exact prefix, not transcripts
of hidden reasoning or proof of why the model internally preferred one token.

The instrumented `explain-llm` SwiGLU path may report the measured
`SiLU(gate_proj(x))*up_proj(x)` channel values, largest signed input-weight
terms, each selected channel's actual full-model next-token logit changes when
ablated, and matched-active/inactive control ablations. These are local,
prompt-specific causal effects under the exact intervention. They do not prove
that a channel has a stable human semantic meaning, that its removal is
necessary for a concept, or that the complete model's reasoning has been
recovered. This path reads weights and hooks and is **instrumented**, not
black-box.

---

## 3. Mandatory Benchmarking Rules

Every benchmark run and report must adhere to the following rules:

1.  **Strict Split Protocol**: You must partition validation/test sets. Never fit hyper-parameters (such as Ridge/Lasso combination weights) on the same dataset used for reporting final generalization error.
2.  **Baselines Must Be Strong**: When comparing GCL to standard machine learning models, always include a tuned **Random Forest Regressor** or a **Gradient Boosting (XGBoost/LightGBM)** baseline. Never downplay or hide their scores when they beat GCL.
3.  **Physical Feasibility Checks**: Every evolved formula must be audited for "numerical mimickry" (such as high-order nested exponential terms $e^{e^x}$ that fit training data but explode OOD).
4.  **Artifact Hygiene**: Release benchmark scripts write to `benchmarks/release/<lane>/<variant>/report.{json,md}`. A file under `benchmarks/` stays outside the current release lane until its generating command, run date, pass/fail status, and claim boundary are listed in `docs/evidence-map.md`.
5.  **Scratch Isolation**: Scratch scripts may exist as research tools. Default scratch data/formula outputs belong under `benchmarks/scratch/<lane>/`, reports and plots belong under `artifacts/scratch/<lane>/`, and any scratch output left in the legacy `benchmarks/` root is legacy repair/rerun input until regenerated under the isolated layout with a manifest.
6.  **Identifiability Claims Need Invariance Tests**: Any claim that GCL recovered a canonical neural structure must include at least two equivalent decompositions, held-out oracle agreement, canonical-signature comparison, and failure accounting for overcomplete bases.
7.  **Output-Only FFN Claims Must Separate Meaning**: Documentation must explicitly separate output-equivalent recovery, canonical structure under an equivalence class, and semantic identity of original hidden neurons.
8.  **Negative Cases Are Mandatory For Structure Claims**: A structure claim must include cases where recovery should downgrade to surrogate or non-identifiable. Inventing atoms in a non-identifiable case is a failure, not a partial success.
9.  **Scaling Claims Need Joint Proof**: A dimension increase only counts when output fidelity, structure diagnostics, negative-case behavior, and query budget pass together.
