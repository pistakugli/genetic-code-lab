# Agent Note: Real Qwen activation capture and verification

Status: implemented

## Context

The real-Qwen SwiGLU adapter and opaque full-module oracle could be smoke-tested,
but no activation artifact separated recovery data from held-out functional,
structural, and distributional samples, and no independent check proved the
saved tensors came from actual inference.

## Decision

The existing `tools/llm_reader/cubic_identification_probe.py` owns both
`--capture-real-activations` and `--verify-real-activation-capture`. Capture
writes four disjoint prompt-based natural-activation NPZ files, a separately
labelled generated-probe NPZ, and a JSON manifest with artifact hashes,
prompt/token provenance, direct-replay deltas, oracle accounting, recovery
status, and CUDA measurements.

The independent verification mode is a separate CLI invocation. It validates
the manifest and NPZ contents, reloads the model, reconstructs each sequence
from saved token IDs, and compares actual target-layer inputs and outputs to
the saved arrays. It also recomputes outputs for generated oracle probes. Its
ground-truth access is post-capture and is not supplied to a recovery solver.

## Alternatives considered

- **Add a new capture runner and verifier script** — rejected because the
  repository already has a live cubic probe runner and Qwen adapter; adding a
  parallel runner would duplicate model and oracle setup.
- **Trust capture-time hashes and replay deltas alone** — rejected because
  self-reported checks from the capture process do not independently establish
  that saved token IDs reproduce the saved hidden states.

## Consequences

The five NPZ files keep natural activations distinct from synthetic oracle
inputs, and four natural prompt partitions remain disjoint. The verifier
provides a post-hoc white-box witness for artifact provenance and exact replay.
The artifacts do not imply recovery: with the current 896-dimensional cubic
design, the workspace guard stops the solver before oracle queries, so all
three reconstruction objectives remain unevaluated.

## Verification

The existing `tests/test_cubic_oracle_recovery.py` covers artifact hash and
shape validation, natural/generated provenance, split disjointness, and the
synthetic recovery regression. The real Qwen integration command and its fresh
verification report are recorded in [the pilot report](../../../../docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md)
and [GATES.md](../../../../GATES.md).

## Cross-references

- [Qwen2 SwiGLU adapter](../../../../tools/llm_reader/adapters/qwen2_swiglu.py)
- [Existing cubic probe runner](../../../../tools/llm_reader/cubic_identification_probe.py)
- [Qwen oracle contract tests](../testing/2026-11-01-real-qwen-oracle-contract-tests.md)
- [Pilot research report](../../../../docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md)
