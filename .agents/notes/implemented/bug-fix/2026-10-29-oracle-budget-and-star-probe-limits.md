# Agent Note: Oracle budget and star probe limits

Status: implemented

## Context

The ND ReLU line-probe solver previously counted oracle calls for reporting but
did not enforce a global limit across kink collection, star probing, and final
fit sampling. Its adaptive star-probe limit was part of the config type but was
not applied. A reported budget therefore did not guarantee the number of calls
actually received by an oracle, and a partial reconstruction could be returned
without an explicit incomplete marker.

## Decision

The solver routes kink collection, primary and normal-consistency star probes,
and fit sampling through one budget-checked oracle wrapper. When the budget is
exhausted, the active query phase stops; previously collected kink points and
recovered geometric candidates remain available. A model whose fit did not
finish is marked `reconstructionComplete: false` and reports the call count,
budget, stop phase, candidate count, and per-phase call totals in
`adaptiveDiagnostics`.

`adaptiveMaxStarProbes` limits primary kink-candidate probes. A candidate's
gradient stencil and any normal-consistency stencils are part of that one probe
for this limit, while all their individual oracle calls count against
`oracleBudget`.

## Alternatives considered

- **Limit only star-probe entry checks and final fitting** — rejected because
  kink collection and nested line-decompiler queries would still exceed the
  advertised global call cap.
- **Throw on budget exhaustion** — rejected because it discards already
  collected kink points and valid plane candidates, and does not provide a
  controlled incomplete result.
- **Count validation stencils as separate primary probes** — rejected because
  the configured limit is intended to bound how many kink candidates are
  examined; the stencils still consume the independent global oracle budget.

## Consequences

Every solver-owned oracle invocation is accounted for centrally, and the
budget cap cannot be exceeded even when exhaustion happens inside a nested
stencil or repeated fit sample. Partial models retain candidate geometry but
use zero-valued fit coefficients until fitting completes; consumers must check
`reconstructionComplete` before treating a budget-limited result as a fitted
reconstruction. RANSAC operates only on collected kink data and consumes no
additional oracle calls.

## Verification

Mock-oracle tests compare the oracle's actual invocation count with solver
diagnostics for budgets 0 and 1, interruption during kink collection,
interruption during a star-probe stencil, RANSAC fallback, and final fit
sampling. They also verify zero/one/multiple star-probe limits and phase-count
reconciliation. The targeted line-probe suites, full Jest suite, and TypeScript
build are run for the change.

## Cross-references

- [ND ReLU line-probe solver](../../../../src/solvers/blackbox-relu-nd-line-probe.ts)
- [Star-probe regression tests](../../../../tests/blackbox-relu-nd-star-probe.test.ts)
