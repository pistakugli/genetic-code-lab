# Agent Note: Real Qwen oracle contract tests

Status: implemented

## Context

The measured cubic recovery path had a synthetic oracle regression but no focused contract test for the Qwen2 adapter or for refusing unsafe full-width recovery before making oracle calls.

## Decision

The existing `tests/test_cubic_oracle_recovery.py` exercises Qwen2 adapter registration and SwiGLU activation exposure with a small deterministic module, and verifies that workspace and query-budget limits reject recovery before making oracle calls. A separate local GPU smoke run verifies model loading and direct Qwen layer replay without making the unit suite depend on GPU availability or downloading model weights.

## Alternatives considered

- **Load the full pretrained Qwen model in every unit test** — rejected because the test would require a large cached checkpoint and CUDA, making the contract suite slow and machine-dependent.
- **Rely on the synthetic recovery test alone** — rejected because it does not cover Qwen adapter dispatch or the real-model workspace boundary.

## Consequences

The inexpensive contract suite can run independently of the Qwen checkpoint and proves the resource guard does not touch the oracle. Exact model/runtime equivalence remains a separate integration witness rather than a portable unit-test claim.

## Verification

The six Qwen adapter/oracle/artifact contracts and two synthetic oracle-only regression tests pass together with `./.gcl-python-gpu/bin/python tests/test_cubic_oracle_recovery.py`. The five `test:cubic-live-access` invariants also pass. A local CUDA capture and independent post-hoc replay of the full 175-sample activation artifact recorded zero input/output delta for all four natural splits and zero output delta for 32 generated oracle probes.

## Cross-references

- [Qwen2 SwiGLU adapter](../../../../tools/llm_reader/adapters/qwen2_swiglu.py)
- [Measured cubic recovery](../../../../tools/llm_reader/cubic_identification_probe.py)
- [Real Qwen activation capture and verification](../feature/2026-11-01-real-qwen-activation-capture-verification.md)
- [Real Qwen pilot gates](../../../../GATES.md)
