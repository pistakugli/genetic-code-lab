# Agent Note: Adaptive oracle swiglu distillation

Status: implemented

## Context

The existing cubic recovery path materializes a dense regression design whose
estimated full-width workspace for Qwen2.5 SwiGLU is 308.4365 PiB. It therefore
cannot query the 896-dimensional oracle under the configured workspace limit.
Reducing the dimension would remove the real module's hidden structure, while
leaving the target unqueried would not test whether an alternative formulation
can run end to end.

## Decision

The existing runner includes an adaptive-width, oracle-only SwiGLU student
distillation path. It fits candidate widths geometrically, selects by generated
held-out oracle observations, and adaptively spends more oracle queries on the
validation shell with the largest residual. Natural recovery activations are
reused as observations; generated probes and natural provenance remain
distinguished. The solver receives only the callable oracle, dimensions,
observations, and query budget. A separate post-hoc evaluator reloads the
teacher only after the solver artifact is frozen.

This is an experimental path around the dense cubic workspace, not a successful
recovery method. On Qwen2.5-0.5B SwiGLU it queried 73,728 full-width points and
selected a 256-component student, but held-out functional error remained high
and no structural matches reached absolute cosine 0.99. The selected student
width is not an estimate of the teacher component count.

## Alternatives considered

- **Keep the dense cubic regression unchanged as the only full-width route** —
  rejected for this experiment because the measured workspace is infeasible
  and the guard prevents any full-width oracle query.
- **Use the teacher's hidden width to initialize the student** — rejected
  because it would leak forbidden structure to the solver and would not test
  whether component count emerges from measurements.
- **Treat a lower memory estimate as recovery success** — rejected because the
  student must independently pass functional, structural, and distributional
  evaluation; this attempt did not.

## Consequences

The path avoids allocating the cubic design and gives an end-to-end oracle
experiment with explicit query accounting. It gives up the existing algebraic
solver's exact gate-recovery guarantee: random student fitting is approximate,
can be unstable at larger widths, and does not identify teacher components by
construction. The final run's measured process peak RSS is 8,605,700,096 bytes,
above the 4-GiB target. A matrix-free structure-aware method remains necessary.

## Verification

`tests/test_cubic_oracle_recovery.py` covers oracle-only inputs, query budgets,
output shape, synthetic solver regression, and artifact provenance. The
preserved `adaptive-oracle-distillation-streaming-batches` solver report
records 73,728 calls and access-boundary fields; the independent evaluation
report records held-out natural metrics, post-hoc structural matching, and
resource measurements. The reports show that the implementation executed but
did not achieve recovery.

## Cross-references

- [Qwen SwiGLU recovery research report](../../../../docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md)
- [Qwen recovery roadmap](../../../../docs/nn-decompiler-roadmap.md)
- [Focused oracle recovery tests](../../../../tests/test_cubic_oracle_recovery.py)
