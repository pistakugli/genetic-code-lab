# Agent Note: Global Spectral-Moment Inversion

Status: implemented

## Context

Local kink probes and direct value fitting do not make global distributional structure explicit. A ReLU sum has a Hessian supported on its kink planes, but an oracle-only solver cannot directly inspect that distribution.

## Decision

GSMI queries a fixed, bounded Gaussian sample from the scalar oracle, estimates second- and third-order Gaussian-windowed distributional-Hessian moments from function values by integration by parts, and inverts a matrix pencil on CUDA to estimate plane vectors and offsets. It then fits the affine output layer using the same sample values. The `second-only` benchmark ablation omits the third moment and pencil inversion. The solver receives no teacher term count, labels, or hidden geometry.

The public API is in [the GSMI solver](../../../../src/solvers/global-spectral-moment-inversion.ts); its CUDA backend is the standalone [spectral worker](../../../../scripts/gsi-spectral-worker.py) reached through [the worker client](../../../../scripts/gsi-cuda-client.ts). The paired screening benchmark and raw-data verifier are [the benchmark](../../../../scripts/benchmark-gsmi.ts) and [the verifier](../../../../scripts/verify-gsmi-results.ts).

## Alternatives considered

- **Add another local gradient or boundary-query policy** - rejected because it retains local geometric probing instead of exploiting global distributional moments.
- **Optimize latent ReLU normals against function values** - rejected because it repeats the direct parameter-fitting mechanism used by GFD-VP.
- **Use only the second Hessian moment** - retained only as a control ablation; it aggregates non-orthogonal planes and does not encode offsets.

## Consequences

The solver has a direct algebraic path from global value moments to plane geometry when the moment rank and matrix-pencil eigenstructure are identifiable. It does not guarantee unique recovery: zero offsets, dependent normals, signed cancellation, repeated eigenvalues, near-parallel conditioning, and non-ReLU smooth contamination remain explicit limitations. The bounded Gaussian sampling policy approximates the untruncated Gaussian identities with a small tail truncation.

## Verification

The focused tests check deterministic bounded sampling, actual oracle-call accounting, zero-budget behavior, and CUDA recovery on an analytic one-plane control. The ten-seed paired pilot records all raw runs; the independent verifier recomputes per-cell metrics and run-status totals, then checks fixture pairing, actual oracle counts, budget limits, CUDA evidence, and the declared verdict before accepting the report.

## Cross-references

- [GSMI mathematical derivation](../../../../docs/research/2026-10-30-global-spectral-moment-inversion.md)
- [GSMI pilot report](../../../../docs/research/2026-10-30-global-spectral-inversion-pilot.md)
