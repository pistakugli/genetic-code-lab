# Agent Note: Measured cubic oracle recovery

Status: implemented

## Context

The overcomplete cubic extractor had been tested on an analytically constructed span, while the
measured-span path lived in a synthetic harness that also held the teacher's hidden weights and
unit count. That left no reusable function boundary proving the extraction could operate on
oracle responses without those values.

## Decision

`recover_gates_from_measured_oracle` accepts a callable, input/response dimensions, and
measurement settings but no teacher object or unit count. It measures the lifted span from
oracle responses, reports the count implied by an even positive span rank, and extracts gates.
`oracle_recovery` passes an eval-only callable to it and reads synthetic truth only after the
recovery returns. The focused test uses a locked wrapper that raises on early truth or count
access.

## Alternatives considered

- **Keep recovery inside `oracle_recovery(units, ...)`** — rejected because the solver and
  post-hoc harness would continue sharing the teacher object and known unit count, so the
  black-box access boundary would remain implicit.
- **Use the analytic-span extractor as the overcomplete proof** — rejected because it skips the
  measured-span step that the current roadmap identifies as the remaining link.
- **Wire the solver directly into a full live layer now** — rejected for this step because the
  live protocol still exposes unit-count metadata; this small CPU gate isolates the solver
  boundary before that integration.

## Consequences

The recovery API no longer needs access to a teacher class and can be called with an eval-only
oracle. `rank / 2` is only the count implied by the measured span; it is not guaranteed to be the
full count when units are silent or the span is incomplete. The 8D synthetic gate does not prove
live-model or whole-layer recovery.

## Verification

The focused test passes for a 10-unit, 8D teacher through a locked oracle, verifies the
span-implied count and all ten post-hoc gate matches, confirms 8640 oracle points, and rejects
malformed response shapes. The five existing live-access tests pass. The CPU CLI lane reports
measured rank 20, span-implied count 10, ten gates above cosine 0.99, and 8640 oracle points.
All three checks in `GATES.md` pass.

## Cross-references

- [Cubic identification probe](../../../tools/llm_reader/cubic_identification_probe.py)
- [Oracle-only recovery test](../../../tests/test_cubic_oracle_recovery.py)
- [NN decompiler roadmap](../../../docs/nn-decompiler-roadmap.md)
