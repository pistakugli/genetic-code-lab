# Agent Note: Boundary conditioned active discovery

Status: implemented

## Context

The existing query-selection methods can reduce prediction error without
identifying every effective boundary. A separate experiment is needed to test
whether queries selected for competing boundary explanations improve boundary
recovery without changing the existing solver baselines or using teacher
metadata as solver input.

## Decision

BCAD is implemented as an independent solver in
`src/solvers/boundary-conditioned-active-discovery.ts`. It proposes multiple
boundary planes from oracle-only finite-difference gradients and affine
intercepts, acquires queries using boundary uncertainty, prediction
disagreement, or gradient-jump criteria, and compares fixed, stable-adaptive,
delayed-reduction, and full-dimensional rank policies. Candidate verification
uses paired probes on both sides of a candidate plane from the same tangent
anchor. Each gradient observation uses `2d+1` oracle calls; fitting does not
query the oracle. Reconstruction is marked complete only when supported
hypotheses, rank stability, and the fit residual satisfy their gates.

The solver remains experimental and is not the default. The fixed 100-seed
16D/32D campaign did not meet the predefined BCAD success criteria; the
multi-crossing cells also show elevated false boundary acceptance. The raw
records remain the source of the reported measurements.

## Alternatives considered

- **Modify ASAD or an existing line-probe mode** — rejected because the
  experiment must distinguish a new boundary-query strategy from changes to
  its baselines.
- **Automatically fall back to an existing solver** — rejected because it
  would obscure BCAD's own boundary-recovery and query-cost results.
- **Infer completeness from a stable low-rank estimate** — rejected because
  stable rank does not establish that all boundary hypotheses are identified.

## Consequences

BCAD retains uncertain alternatives rather than forcing one explanation, at
the cost of extra query-selection and verification work. Its paired
same-anchor probes make local side comparisons more controlled but do not
guarantee identifiability when several boundaries affect the same observations.
The benchmark verifier reconciles 11,000 raw runs across 110 cells with zero
oracle-budget overruns and zero available-diagnostic mismatches. Under the
predefined criteria, the measured verdict is `not-supported`; BCAD therefore
remains an experiment and is not promoted as a performance improvement.

## Verification

The BCAD unit tests exercise acquisition strategies, rank policies, separate
near-parallel boundaries, and incomplete reconstruction. The 100-seed
benchmark is reconciled against actual oracle invocations, shared teacher and
evaluation hashes, budgets, and its report by `verify-bcad-results.ts`.

## Cross-references

- [BCAD solver](../../../../src/solvers/boundary-conditioned-active-discovery.ts)
- [BCAD tests](../../../../tests/boundary-conditioned-active-discovery.test.ts)
- [BCAD benchmark report](../../../../docs/research/2026-10-29-bcad-experimental-report.md)
- [BCAD raw-data verifier](../../../../scripts/verify-bcad-results.ts)
