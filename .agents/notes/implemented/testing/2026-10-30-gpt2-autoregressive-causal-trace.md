# Agent Note: GPT-2 Autoregressive Causal Trace

Status: implemented

## Context

The five-case internal trace showed that hook-level access and a one-step path
ablation do not explain free-running token sequences or establish that a
discovered path is selective. The 50-fact source artifact and its paths must
remain frozen while those claims are tested against actual greedy contexts
and structurally matched controls.

## Decision

The GPT-2 trace runner records a full-context greedy prediction trace and
one-step path, component, restoration, and 100-control interventions for each
prediction. Controls preserve path component count, type, layer, and token
positions; magnitude deviations and deterministic sample identities are
recorded. The runner reports signed and absolute logit-effect percentiles,
uses the original signed-effect percentile for the strict specificity gate,
and does not test path sufficiency. A separate verifier replays all 50 cases
against the local model and frozen source artifact.

## Alternatives considered

**Treat activation magnitude or a changed prediction as path specificity.**
Rejected because either can show influence without showing selectivity versus
structurally matched random components.

**Feed intervened tokens into later steps.** Rejected because later contexts
would diverge from the unmodified greedy baseline, confounding one-step causal
effects.

**Relax the magnitude-match or percentile gate after seeing outcomes.**
Rejected because the resulting specificity claim would be tuned to the
observed paths; unmatched attention controls are instead reported as a
specificity failure.

## Consequences

The experiment provides a verified token-by-token witness, but it uses local
weights and hooks rather than oracle-only access. The 12-token generation cap
truncates all 50 outputs. Specificity passes at only 9 of 600 individual
steps under the signed-effect gate and at no complete case; sufficiency
remains untested. The raw tensors and detailed report are large, so
reproducibility requires preserving their manifest and hashes.

## Verification

The final all-50 report contains 600 generated steps and 60,000 control draws.
The independent verifier reproduces the greedy logits and checks source
prompts, paths, activation traces, ablation arithmetic, control structure,
and report claim boundaries. The five-case legacy regression, 13 focused
tests, TypeScript checking, and project build are separate checks.

## Cross-references

- [Autoregressive causal trace research report](../../../../docs/research/2026-10-30-gpt2-autoregressive-causal-trace.md)
- [Initial GPT-2 internal-trace decision](../2026-10-30-gpt2-internal-computation-trace.md)
- [All-50 machine-generated report](../../../../benchmarks/release/nn-live-gpt2-autoregressive-causal-trace/all-50/report.md)
