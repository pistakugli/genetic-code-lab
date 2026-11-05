# GCL novelty assessment

**Cutoff:** Public sources checked through 2026-10-30. This assessment is intentionally conservative. The detailed implementation map and bibliography are in [GCL_PRIOR_ART_AUDIT.md](GCL_PRIOR_ART_AUDIT.md); claim-level support is in [GCL_CLAIM_EVIDENCE_LEDGER.md](GCL_CLAIM_EVIDENCE_LEDGER.md).

## Bottom line

**No new general solver or unique causal-intervention method is established by the current evidence.** The white-box GPT-2 reader integrates known instrumentation, activation patching, ablation, causal tracing and circuit-analysis techniques. The specific 50-case autoregressive and 744-pair transfer packages are credible reproducible evaluation contributions, but the audit did not establish historical priority or direct superiority over a compatible baseline.

The black-box arm is more technically distinctive: it applies oracle-only fitting to selected real 3,072D/4,096D SwiGLU functions. This is a different target from established shallow-ReLU extraction and generic model stealing. It remains a **narrow application demonstration**, not evidence of an algorithmically novel general neural-network decompiler. The tested solver is given the activation family and structural order/rank information; the oracle exposes unit count; selected cases do not generalize to a strict randomized court (1/6 pass).

## Classification of the main GCL capabilities

| GCL feature | Classification | What can be claimed |
|---|---|---|
| Transformer hooks, activations, residual/head/MLP tracing | **Established method + engineering integration** | GCL has a reproducible custom pipeline for recorded GPT-2 cases. Internal access and activation caching are widely available, including TransformerLens. |
| Factual path localization and ablation | **Established causal-analysis family + constrained GCL adaptation** | Frozen selected paths influence many measured scores. ROME, Knowledge Neurons and circuit papers precede this method class. |
| Token-by-token trace and actual greedy replay | **Potential reproducible empirical contribution** | 50 cases / 600 steps replay and preserve per-step contexts. It is a useful machine-readable dataset, not a new causal principle. |
| 744-pair cross-factual transplant | **Potential protocol/evaluation contribution** | The scale, pair selection, alignment, restoration, controls and reporting may be distinctive as a package. Interchange/activation patching itself is established. No exact matching protocol was found in the reviewed sources; that is not proof of priority. |
| Path necessity/sufficiency/specificity | **Partial, with important negatives** | Influence is observed; strict greedy necessity is 3/45; complete sufficiency is not established; global specificity is not established. |
| Causal interaction estimates | **Established measurement class + repository integration** | Group-minus-singleton effects are measured. Additivity labels do not uniquely identify redundancy or synergy. |
| Identifiability verdicts/source alignment | **Useful diagnostic implementation, not general uniqueness proof** | Rank/cancellation/duplicate warnings and post-hoc atom pairing exist. Synthetic tests and source-assisted validation do not establish an identifiable decomposition in general. |
| Selected SwiGLU oracle recovery | **Potential new application/scale; algorithmic novelty unknown** | A restricted real-model black-box target is distinct from the prior art reviewed. Strong claims require blinded order and input-distribution assumptions. |
| Full neural decompilation | **Unsupported** | Neither selected GPT-2 paths nor selected SwiGLU functions reconstruct all layers, attention, residual dependencies, or a complete trained model. |

## What is and is not genuinely differentiating

### Established before GCL

- Direct extraction of restricted neural-network classes from prediction access [R1], API model stealing [R2], and functionality substitutes [R3].
- Mathematical neural-network reconstruction and identifiability questions [R4, R5].
- Factual causal localization/editing [R9, R10].
- Activation/path patching and source-to-base interchange intervention [R11, R12].
- Automated task-circuit localization [R13, R14].
- Circuit hypothesis testing by resampling [R15].
- White-box GPT-2 circuit reconstruction, transformer activation tooling, sparse feature dictionaries, attribution graphs and agentic circuit explanation [R17–R22]. The OpenAI neuron-explanation page [R19] is relevant public work, but was not retrievable for detailed claims in this audit.

### GCL-specific evidence package

The strongest distinctiveness lies in **combining**:

1. A frozen factual ordered-path source set.
2. Autoregressive trace replay that preserves each actual baseline context.
3. Same-family paired facts with exact fixed-text token alignment.
4. Bidirectional transfer, restoration, necessity measures and structural controls.
5. A raw archive and verifier capable of checking the recorded measurements.

The 744-pair scale is valuable because it reveals direction asymmetry, strict-necessity failures, inadequate matching coverage, and a narrow specificity-positive subset that a five-case demonstration would obscure. Scale alone is not novelty; the defensible contribution is the **pre-specified, auditable measurement protocol and its mixed results**. A paper or report should provide a stable dataset/protocol citation and avoid “first” claims until a broader search and direct baseline comparison support them.

### Distinct black-box target, bounded conclusion

The reviewed extraction literature does not contain a direct matching report for scalar-oracle recovery of a selected real Llama/Mistral SwiGLU function at the stated hidden dimensions. This makes the target interesting. But:

- The solver knows the model family and receives atom-count/rank inputs.
- The oracle readiness response exposes the number of selected units.
- Training samples in the strict isolated court use Gaussian vectors, not only empirical residual-stream activations.
- The 2/2 selected-court result is not representative evidence after the randomized strict 1/6 result.
- Functional fit, parameter/source alignment, and uniqueness are different claims.

Therefore, the strongest safe claim is: **GCL has implemented an oracle-only solver for a structurally specified SwiGLU function and has obtained successful recovery on selected cases, with poor pass rate under one stricter randomized court.** “General black-box decompiler” and “novel extraction algorithm” are not currently supported.

## Unsupported claims to reject

1. “GCL invented activation transplantation/counterfactual patching.” **No**—IIT, causal abstraction and activation patching establish the method family.
2. “The discovered path uniquely explains the prediction.” **No**—influence does not entail specificity; matching coverage fails on nearly all transfer directions.
3. “The path is necessary in 44/45 facts.” **No**—44/45 is a likelihood decrease; strict correct-greedy necessity is 3/45.
4. “The path is sufficient because donor likelihood rises.” **No**—no complete sufficient-circuit criterion is met.
5. “GCL's 50-case GPT-2 trace is a black-box decompiler.” **No**—local weights and hooks are used.
6. “Residual reconstruction is exact across the 50 cases.” **No**—the all-50 maximum reported error is approximately `3.05e-5`; zero belongs to a separate five-case report.
7. “The OOD circuit is robust.” **No**—the measured OOD pilot passes 13/30 prompts and 2/5 cases.
8. “The SwiGLU decompiler generalizes across random real blocks.” **No**—the strict randomized 16-atom court passes 1/6.
9. “Source-atom matching proves a unique decomposition.” **No**—matching to revealed source truth is a post-hoc witness; alternative functionally equivalent decompositions remain possible.
10. “No prior art exists for GCL's exact protocol.” **Unverified**—the review found no exact match in the checked corpus; absence is not established.

## Answer to the project-level question

GCL is best described today as:

- **A reproducibility-focused integration of established mechanistic-interpretability methods** for an instrumented GPT-2 path, with a specific broad counterfactual-transfer evaluation; and
- **An experimental black-box function-recovery system** for selected SwiGLU sums under known structural assumptions.

It is **not yet** a complete neural decompiler. Its meaningful next advance is to remove hidden solver-side structural hints and make evaluation reflect actual model activation inputs, while separating function equivalence from atom identity. See [the proposed next development direction](GCL_NEXT_DEVELOPMENT_DIRECTION.md).

## References

See the numbered full bibliography [R1–R21](GCL_PRIOR_ART_AUDIT.md#bibliography-and-primary-sources). The method-level novelty boundary is particularly informed by Jagielski et al. [R1], ROME [R9], Knowledge Neurons [R10], IIT/Causal Abstraction [R11–R12], ACDC/EAP [R13–R14], and Circuit Tracing [R20].
