# Agent Note: Why-report rejected candidate scope

Status: implemented

## Context

When a replacement search records its best candidate but that candidate fails
the replacement gates, the why-report presented the candidate's components and
drop in a table headed `replacement path`, immediately after a sentence saying
no passing path was found. A reader could mistake the rejected candidate for
the path that carried the effect. The Pythia-70m artifact made this concrete:
its service-question entry showed a candidate (MLP L4 n234, attention L2 h4 /
L3 h4, drop 1.0) under the replacement-path header while `found` was false.

## Decision

Each why entry's replacement block now carries an explicit `pathChangeScope`
with three values: `replacement path` when the search passed, `best rejected
candidate` when a candidate is recorded but failed the gates, and `no
candidate recorded` when the search produced no interventions. The markdown
table header uses the scope, and the why statement names a rejected candidate
and its drop, stating that the comparison describes a candidate, not a carried
effect.

## Alternatives considered

- **Drop the comparison for failed searches** — rejected because the best
  rejected candidate is exactly the evidence that the effect did not move
  there.
- **Mark the scope only in the markdown** — rejected because the JSON blocks
  feed downstream consumers that never read the markdown.

## Verification

The independent verifier recomputes the found flag and path comparison from
the raw search record, so the scope cannot contradict the measurements; the
full cycle verification (positive, negative, backward-compat) and the 11-test
suite pass, and the regenerated Pythia-70m artifact verifies with scopes
`best rejected candidate` and `no candidate recorded` on record.

## Cross-references

- [Pythia-70m OOD replacement transfer note](../feature/2026-11-14-pythia-70m-ood-replacement-transfer.md)
- [Why-report layer note](../feature/2026-10-30-ordered-path-ood-stress.md)
