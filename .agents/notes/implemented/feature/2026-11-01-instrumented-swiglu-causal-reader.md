# Agent Note: Instrumented SwiGLU causal reader

Status: implemented

## Context

The existing live LLM reader inspected GPT-2 MLP units, while GCL's Qwen2,
LLaMA, and Mistral adapters already exposed SwiGLU projections. The oracle-only
decompilation experiments did not recover the real Qwen module's hidden
components. Without a usable instrumented path, GCL could neither inspect the
actual gate/value computation nor test whether removing a selected channel
changes a model decision.

## Decision

The existing `explain-llm` command routes supported SwiGLU model types through
their registered adapters. It captures the MLP input, gate and up projections,
their SiLU-gated product, and down-projection output; checks reconstruction of
the observed product and projection; ranks channels; and measures full-model
next-token logit changes from individual channel ablations with active and
inactive controls. `--layers` selects explicit layers or all layers. The
reader explicitly identifies this path as instrumented access, not black-box
analysis.

## Alternatives considered

- **Continue with oracle-only recovery as the internal explanation path** —
  the Qwen SwiGLU recovery experiments did not identify useful hidden
  components, and input/output observations do not expose the real module's
  internal activations used by this reader.
- **Add another standalone Qwen analysis script** — rejected because the
  existing reader, adapter registry, and `explain-llm` command already provide
  the correct integration points; a parallel entry point would duplicate
  model-loading, reporting, and CLI behavior.
- **Keep the reader GPT-2-only** — rejected because it leaves the existing
  SwiGLU adapters unused and cannot inspect the target Qwen MLP computation.

## Consequences

This provides inspectable, testable evidence about individual MLP channels
for supported architectures and measures causal next-token effects in the
complete model. It requires instrumented access to weights and hooks, so it
does not satisfy the oracle-only boundary. It analyzes a selected prompt
position and individual channels; attention, multi-channel interactions,
complete autoregressive reasoning traces, and semantic generalization are not
established by these measurements.

## Verification

The targeted reader and black-box tests pass, Python modules compile, and
`npx tsc --noEmit` succeeds. A real Qwen2.5-0.5B-Instruct run with
`--layers all` checks the SwiGLU product and down projection in all 24 layers
and accounts for every forward pass. The JSON and text report formats expose
the same bounded claims.

## Cross-references

- [Instrumented reader implementation](../../../tools/llm-reader-gpt2-live.py)
- [SwiGLU adapter registry](../../../tools/llm_reader/adapters/registry.py)
- [Qwen SwiGLU roadmap and measured results](../../../docs/nn-decompiler-roadmap.md)
