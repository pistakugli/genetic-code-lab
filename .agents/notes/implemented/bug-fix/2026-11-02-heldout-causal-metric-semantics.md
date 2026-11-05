# Agent Note: Heldout causal metric semantics

Status: implemented

## Context

The first decompiler report published dependency and interaction reconstruction
accuracy by reusing the counterfactual-effect and interaction-residual-sign
scores. Those labels therefore did not describe independent measurements, and
the single paired-deletion case could not distinguish unseen single effects
from an unseen joint effect.

## Decision

Validation reports measure paired counterfactual change, previously unqueried
single-deletion dependencies, held-out paired-deletion next-token outcomes, and
held-out pairwise logit-margin residuals separately. A held-out additive
interaction case is selected only when its constituent single-deletion
index groups were measured as input-deletion effects but the joint context was
not. An independent dependency case uses individual deletions absent from
training observations.
Case-level oracle evidence maps each measurement to its context and query ID.
Validation budget estimation reserves capacity for the formulation, dependency,
and additive-interaction cases.

## Alternatives considered

- **Reuse the existing aggregate scores under separate metric names** —
  rejected because a duplicated score creates the appearance of independent
  causal verification without providing it.
- **Use the same paired-deletion example for both single-dependency and
  interaction validation** — rejected because the single deletions may already
  be training evidence, or may be unknown to the reconstruction, confounding
  the two claims.

## Consequences

Reports can now expose an unavailable metric as `null` when no qualifying
held-out intervention exists, rather than fabricating a score. The extra
validation cases consume more oracle queries. Passing these local checks still
does not establish generalization across arbitrary prompts or recovery of
hidden reasoning.

## Verification

`tests/test_blackbox_decision.py` verifies distinct scores, additive-model
selection, and evidence provenance for synthetic held-out intervention cases.
The Qwen2.5-0.5B-Instruct run in `/tmp/gcl-decompile-qwen-run9` uses 69 unique
queries within budget 100 with matching worker accounting. It predicts the
held-out interaction residual as `-0.125` when the oracle measures `+0.875`,
and its full held-out response fidelity is `0.0`; this is negative
generalization evidence, not a decompilation-success claim.

## Cross-references

- [Black-box decompiler implementation](../../../../tools/llm_reader/blackbox_decompiler.py)
- [NN decompiler roadmap](../../../../docs/nn-decompiler-roadmap.md)
- [Black-box decompiler tests](../../../../tests/test_blackbox_decision.py)
