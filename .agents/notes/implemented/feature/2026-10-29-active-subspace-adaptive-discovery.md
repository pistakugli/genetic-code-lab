# Agent Note: Active-Subspace Adaptive Discovery

Status: implemented

## Context

The ND ReLU solvers probe random lines and then use Star-Probe or RANSAC to recover candidate hyperplanes. The existing active-subspace solver targets an FFN surrogate and does not provide a separate adaptive hyperplane-discovery path. ASAD tests whether local affine disagreement can direct queries toward informative regions while measured gradient variation identifies a lower-dimensional search space.

## Decision

`src/solvers/active-subspace-adaptive-discovery.ts` implements ASAD as an additive oracle-only solver exported from `src/index.ts`. It estimates gradients with coordinate finite differences, estimates active directions from the eigensystem of observed gradient covariance, and selects center queries by disagreement among observed local affine hypotheses plus a novelty term. If the rank is not identifiable from enough observations, it uses global exploration instead of treating a low observed rank as established.

Candidate hyperplanes are inferred from local paired affine models and accepted only when the gradient jump, within-side flatness, and local separation agree. A single gradient probe consists of one center evaluation and two evaluations per input coordinate, for `2 * dimension + 1` actual oracle calls. A single central wrapper enforces `oracleBudget`; no fitting or validation oracle queries are made after gradient acquisition. The coefficient fit must be overdetermined by at least `max(4, ceil(fitWidth * 0.5))` observations and have relative training residual at most `0.02` before the reconstruction can be marked complete. Insufficient or ambiguous evidence returns a partial model with `reconstructionComplete: false`; the residual and validation observation count are included in solver and raw benchmark diagnostics.

The RANSAC, Star-Probe, and Adaptive implementations remain separate and unchanged by the ASAD feature. `scripts/benchmark-asad.ts` generates same-seed teachers for all four methods, counts actual calls through an independent wrapper, and scores predictions on independently seeded test points. Query-saving comparisons use symmetric relative MAE difference with a `1e-8` absolute floor, while the non-key regression gate is relative to the best baseline with the same floor.

## Alternatives considered

- **Adapt the existing line-sweep direction scheduler** — rejected because the discovery loop would remain coupled to the existing one-dimensional kink extraction and Star-Probe/RANSAC path, making it a weak test of a genuinely different mechanism.
- **Use the instrumented FFN active-subspace decompiler** — rejected because it returns a local surrogate from gradient probes and does not reconstruct effective ReLU hyperplanes; its training samples also do not define the requested competitive query-selection rule.
- **Accept every gradient-difference plane** — rejected because a pair of observations can cross several boundaries or a smooth contaminant. Local consistency and explicit uncertainty avoid converting that ambiguity into asserted reconstruction success.

## Consequences

ASAD can spend fewer oracle calls on isolated low-rank boundaries, but its coordinate gradient stencils still cost `2 * dimension + 1` calls per observation. Gradient covariance and affine consensus are finite-sample estimates; multiple crossings, near-parallel facets, and contaminated local behavior can therefore leave a result incomplete. The solver intentionally does not fall back to a legacy algorithm, so benchmark comparisons remain attributable to the new mechanism.

## Verification

Unit tests compare actual mock-oracle invocations with diagnostics, verify budget limits, recover an isolated low-rank facet, reject an ambiguous composite jump, and check full-rank and smooth-function behavior. The reproducible benchmark records raw per-run outcomes, diagnostics, oracle calls, errors, structure scores, runtime, and verdict inputs in `scratch/results/asad/`; the 100-seed report is generated at `docs/research/2026-10-29-asad-experimental-report.md`.

## Cross-references

- [ASAD solver](../../../src/solvers/active-subspace-adaptive-discovery.ts)
- [ASAD benchmark](../../../scripts/benchmark-asad.ts)
- [ASAD falsification tests](../../../tests/active-subspace-adaptive-discovery.test.ts)
- [ASAD experimental report](../../../docs/research/2026-10-29-asad-experimental-report.md)
