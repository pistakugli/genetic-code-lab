# Agent Note: Blackbox decision evidence

Status: implemented

## Context

The output-only reader previously analyzed only a fixed factual dataset. It
could not accept an arbitrary prompt or provide measured evidence about why one
next-token candidate outweighed alternatives in that prompt.

## Decision

The existing reader adds an interactive decision mode that queries the
baseline prompt and bounded whitespace-delimited-span deletion counterfactuals. It records
the target-vs-competitor logit-margin change, target rank/probability,
next-token distribution shift, exact edited prompt, and oracle call count. The
`explain-llm-blackbox` CLI and `llm:read:blackbox` package command expose the
mode. CLI runtime selection respects explicit Python overrides and chooses the
GPU environment for `cuda`/`auto` when available. It is an output-behavior
report, not an explanation of hidden internal reasoning.

## Alternatives considered

**Continue with curated fact prompts only:** rejected because it cannot answer
a user's question about a prompt of their choice.

**Use hooks or model weights:** rejected for this feature because that would
cross the output-only access boundary and conflate behavior measurement with
instrumented circuit analysis.

## Consequences

This adds a directly usable prompt-analysis path while preserving the original
factual benchmark as its default mode. Span deletion can create
unnatural text, so its score changes are labeled as exact text-edit effects
and cannot be presented as semantic causes or internal reasoning steps.

## Verification

Unit tests exercise target/competitor margins, positive and negative effects,
query accounting, probe bounds, control-token exclusion, and invalid input
handling. A local Qwen2.5-0.5B-Instruct CLI run records a rank-1 target, eight
measured counterfactuals, nine oracle calls, and oracle-only access metadata.

## Cross-references

- [Claim policy](../../../../docs/claim-policy.md)
- [NN decompiler roadmap](../../../../docs/nn-decompiler-roadmap.md)
