# Agent Note: Unknown-order empirical SwiGLU recovery integration

Status: implemented

## Context

The recommended next development direction (blind recovery on real
activation distributions, hidden atom count) had been executed as a Kaggle
T4 kernel campaign on 2026-11-18: a pool-sampler kernel (v4, COMPLETE)
recorded and validated all six preregistered empirical MLP-input pools, and
the first ladder kernel (`gcl-unknown-order-ladder-m-e1` v5, COMPLETE)
recorded case M-E1. The session ended before the outputs were downloaded or
landed: origin/main had none of the tools, the artifacts, or the referenced
preregistration doc (which lived only in ephemeral kernel-side commits
`db5f33f`/`b3a906b`).

## Decision

Land the completed first case as recorded evidence, with the campaign's own
frozen protocol, plus a new independent verifier to bring the lane up to the
repo's verification standard:

- `tools/llm_reader/unknown_order_swiglu_recovery.py`,
  `empirical_activation_sampler.py`, and
  `synthetic_swiglu_oracle_server.py` landed from the input dataset snapshot
  (`kuglipista/gcl-unknown-order-input`, pinned `b3a906b`).
- `blind_swiglu_oracle_decompiler.py` gained the empirical sampling mode
  (`--train-sampling-mode empirical` + `--empirical-inputs-path`, pool splits
  train/selection/validation-gate/validation-fairness) and the unknown-order
  protocol (ready frame without `unitCount`, candidate `--atom-count` not
  cross-checked against a disclosed truth, report fields `unknownOrder`
  and `unitCount: null`); `swiglu_oracle_server.py` gained
  `--hide-unit-count` (protocol tag switch) and the null-control
  `--permute-responses-seed`.
- New `tools/llm_reader/verify_unknown_order_empirical.py` independently
  recomputes the selection rule, the upper-region non-identifiability flag,
  every fit's grades from the frozen fits plus recorded teacher-side
  response tables, the pool sha256, and teacher-secret consistency
  (`tests/test_unknown_order_empirical.py`, 27 tests).
- Artifact landed at
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-m-e1-2026-11-18/`
  with kernel-evidence provenance (heavy files stay in Kaggle kernel output
  and the pools dataset, sha256 recorded); kernel sources landed under
  `scripts/gpu/kaggle/unknown-order-*/`.
- Research doc
  `docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md`
  reconstructs the preregistration from the frozen protocol echoes and
  discloses the doc-late gap; the five remaining cases are re-registered
  (mechanical m = 6/8/12 by layer position, seed pairs advancing by two)
  before any run against them.

## Result

M-E1 (Mistral-7B layer 1, true m = 6): ladder 4..24, frozen gate
r_selection <= 0.05. Inferred order 10; candidates 4-8 sit at
0.058-0.079 and candidate 6 at 0.0737; candidates 12 and 14 are
sufficient within 2x (non-identifiable upper region). Verdict
`inferred-order`, recorded as an honest over-estimate against the hidden
order, not order recovery. The verifier passes against the downloaded
kernel output including the heavy response tables and structurally against
the repo artifact.

The batch kernel (`scripts/gpu/kaggle/unknown-order-ladder-batch/kernel.py`,
Kaggle `gcl-unknown-order-ladder-batch` v1, 2 h 02 m) then completed the
campaign: the five freshly preregistered cases plus the M-E1
null-permutation control. L-L27 inferred the exact hidden order (12 of 12);
L-E1 over-estimated (8 of 6), L-M14 over-estimated (16 of 8), M-M15
over-estimated (18 of 8), M-L28 recorded no sufficient order at all, and
the M-E1 null control recorded no-sufficient-order (the gate does not fit
permuted responses). Every case was verified by
`tools/llm_reader/verify_unknown_order_empirical.py` before integration
(`scripts/gpu/integrate-unknown-order-batch.py`); roadmap item 7 and the
capability matrix record the campaign as mostly negative under the frozen
threshold, per the preregistered stop condition.

