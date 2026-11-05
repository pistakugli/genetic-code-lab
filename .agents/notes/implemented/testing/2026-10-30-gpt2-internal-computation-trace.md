# Agent Note: GPT-2 Internal Computation Trace

Status: implemented

## Context

The existing ordered-path artifact records candidate GPT-2 attention and MLP
components, but path replay alone does not demonstrate which internal
quantities the model computed or whether ablating a recorded component changes
the target score. Without direct tensor checks and removal/restoration
interventions, GCL cannot distinguish an instrumented trace from an
evidence-backed causal claim.

## Decision

GCL has a separate five-case GPT-2 internal-trace experiment. It consumes the
exact existing source prompts and ordered paths, records layerwise residual,
attention, MLP, normalization, and logit tensors, then removes/restores
selected activations and compares recorded controls. A separate verifier
checks the report against the raw tensor archive and source artifact. The
experiment is labeled `instrumented-witness`; it does not claim black-box
decompilation or a complete semantic circuit.

## Alternatives considered

**Treat ordered-path replay or activation magnitude as causal evidence.**
Rejected because neither intervenes on the computation nor proves restoration
and control behavior.

**Change the original 50-fact path artifact or its selection thresholds.**
Rejected because that would confound the capability test with a newly tuned
path and would modify the baseline being evaluated.

**Claim an oracle-only or complete LLM reconstruction.** Rejected because this
experiment uses local model weights and hooks as its measurement mechanism;
that access mode cannot prove black-box reconstruction.

## Consequences

The capture provides reproducible quantitative evidence for the selected
inputs and interventions, at the cost of requiring local GPT-2 weights and
framework hooks. Attention probabilities and exhaustive neuron/head ablations
remain out of scope. Random-active-neuron controls limit claims of selected
MLP specificity, and three of five source targets are not greedy next tokens.
Broader prompts and amplitude-matched controls remain future gates.

## Verification

The focused instrumentation tests pass; the independent verifier checks the
five-case JSON, source identity, tokenization, raw NPZ inventory, quantitative
tensor relationships, interventions, and research claim boundaries. The
benchmark is reproducible with the `five-facts` variant. TypeScript integration
type-check and project build are required gates.

## Cross-references

- [Five-case research report](../../../../docs/research/2026-10-30-gpt2-internal-computation-trace.md)
