# Agent Note: Multi-Hypothesis Boundary Continuation and Disambiguation

Status: implemented

## Context

The BCAD campaign reduced oracle use in some difficult cells without recovering reliable geometry. Its rank estimates and candidate boundaries could look decisive even when competing explanations remained viable. A follow-up must test whether explicit uncertainty and contradiction-seeking measurements improve boundary identification, not merely prediction efficiency.

## Decision

MBCD is implemented as a separate oracle-only solver. It retains plausible active-rank and geometric boundary alternatives; reports detection, normal identification, offset identification, uniqueness, and completion separately; selects contradiction-seeking probes; and validates candidates on later oracle measurements. ASAD, BCAD, RANSAC, Star-Probe, and Adaptive remain separate and unchanged.

The benchmark compares the full method with ablations removing rank alternatives, contradiction-seeking acquisition, or independent candidate validation. It uses paired deterministic teachers, budgets, evaluation points, and seed derivation across 16D/32D and the five BCAD focus scenarios.

The predeclared criterion requires at least two of three primary scenario classes to qualify at both dimensions. A cell qualifies only if MBCD achieves at least 25% lower MAE, or at least 30% fewer oracle calls while MAE is no more than 5% worse than the best scored existing mode; false-accepted-run rate may rise by at most one percentage point. The campaign uses 100 seeds per cell and requires zero budget overruns.

## Alternatives considered

- **Add more acquisition weights to BCAD** — rejected because the observed failure is premature certainty about rank and boundary identity, not merely an unoptimized query score.
- **Select one rank or boundary as soon as it leads the current score** — rejected because an early winner can be unsupported by the oracle data and conceal unresolved alternatives.
- **Use teacher normals, offsets, or test points to guide the solver** — rejected because it would invalidate the black-box experiment.

## Consequences

The solver can return incomplete reconstructions even when it has a useful predictive fit; this is intentional when geometric evidence is ambiguous. Explicit alternatives and later validation consume query budget and may lower completion rates. Matched raw results, rather than tests or a favorable subset of seeds, determine whether MBCD is supported.

## Verification

Actual mock-oracle invocations are reconciled against solver diagnostics and budgets. The matched benchmark records per-run MAE, completion, plane counts, normal/offset errors, unresolved hypotheses, oracle use, runtime, and exceptions. An independent verifier recomputes cell metrics and the predeclared verdict from raw rows.

The 100-seed campaign recorded 9,000 runs; independent verification found zero budget overruns and zero diagnostic/oracle-count mismatches. MBCD did not meet the predeclared criterion: none of the six primary cells qualified, and full MBCD's reconstruction-complete rate was 0% in each. The result is experimental evidence against the current method, not support for its boundary-identification hypothesis.

## Cross-references

- [BCAD experimental report](../../../../docs/research/2026-10-29-bcad-experimental-report.md)
- [MBCD experimental report](../../../../docs/research/2026-10-29-mbcd-experimental-report.md)
