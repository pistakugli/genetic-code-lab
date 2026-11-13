# Agent Note: CPU device and full rescue coverage

Status: implemented

## Context

Two behaviours blocked a CPU-live stress run and complete break coverage.
First, the independent verifier rejected any artifact whose model device was
not `cuda`, although a live CPU run of a small model is a legitimate lane
execution. Second, the stability runner's rescue budget defaulted to 2, so an
artifact with more broken paraphrases left recorded breaks without rescue
measurements or why-reports, contradicting the lane's north-star that every
recorded break gets a why-report: the Pythia-70m run broke five of six
families and would have silenced three of them.

## Decision

The verifier's device check accepts exactly `cuda` and `cpu` and still rejects
unknown device strings, so a fabricated or misrecorded device fails
verification. The runner's `--max-rescue-attempts` default rises from 2 to 6
(one per OOD family) and the budget is recorded in the report config as
`maxRescueAttempts`, so an artifact states how many breaks it could have
covered.

## Alternatives considered

- **Keep CUDA-only verification and relabel CPU runs** — rejected because the
  artifact must state the device it actually ran on.
- **Retry rescues until every break is covered regardless of budget** —
  rejected because expensive GPU lanes need an explicit bound.

## Verification

A new test verifies that `cuda` and `cpu` case reports pass and that an
unknown device string is rejected with one error; the suite grows to 11 tests.
Both the recorded GPT-2 (CUDA) and the new Pythia-70m (CPU) artifacts verify.

## Cross-references

- [Pythia-70m OOD replacement transfer note](../feature/2026-11-14-pythia-70m-ood-replacement-transfer.md)
