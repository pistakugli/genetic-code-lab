# Agent Note: Ordered path ood stress

Status: implemented

## Context

The existing ordered-path stability lane checks standard paraphrase replay but
does not separate results by prompt family or exercise a held-out set of
surface forms. A path can pass the original prompt and neighboring-layer
controls while failing under rephrasing, so a single aggregate hides a useful
failure mode.

## Decision

The Python stability runner accepts `--paraphrase-profile standard|ood`;
`standard` remains the default. The separate OOD profile generates six
deterministic capital-question families from the subject alone, measures
per-prompt token counts and causal replay drops, and aggregates outcomes by
family. For a bounded number of broken prompts, the stability runner now
reselects MLP neurons and attention heads on that exact prompt using the
existing component and ordered-path solvers, over the source and configured
rescue-shift layers. A replacement passes only when the existing combined,
ordered-path, and control gates pass. Same-component layer shifts remain
separate diagnostic hints. For an OOD replacement path that passes discovery,
the runner replays the same interventions on every other measured OOD family,
remapping the read-token position and excluding the discovery family. This
cross-family transfer remains diagnostic and does not change the original-path
stability gate. Standard paraphrase runs and searches without held-out families
record transfer as not applicable. The TypeScript wrapper selects models,
bounds the replacement candidate/attempt budgets, and writes OOD runs to their
own benchmark variant. An independent verifier checks raw transfer effects,
threshold decisions, family coverage, and aggregates.

## Alternatives considered

- **Replace the existing standard prompts** — rejected because existing
  stability evidence and default behavior must remain comparable.
- **Treat a layer-shifted replay as replacement discovery** — rejected because
  moving the same neuron/head identities does not find a new prompt-specific
  path.

## Consequences

The separate artifact lane preserves standard results and makes family-level
failures visible. The templates are curated capital questions, so a passing
result would not establish general semantic or arbitrary-prompt robustness.
When rescue probes are enabled, Markdown reports also show their best
layer-shift hints separately from prompt-specific replacement paths. The
replacement search is instrumented and layer-bounded; it is not an
oracle-only or full-model circuit recovery claim.
The matched five-subject GPT-2 standard baseline passes 30/30 paraphrase
replays; the OOD profile passes 13/30 and fails its configured gate, despite
perfect original replay and neighboring-layer specificity. This is evidence of
prompt-family sensitivity, not a claim about circuit absence. Cross-family
replacement transfer has unit-test coverage but no fresh real-model artifact
yet, so its empirical rate remains unknown.

## Verification

`tests/test_ordered_path_ood_stress.py` covers deterministic prompt families,
answer non-leakage, path rediscovery, remapped held-out replay, query-free
abstention, aggregates, and verifier rejection of inconsistent raw
measurements. The five-case CUDA pilot and independent verifier are documented
in the research report; the standard profile remains the CLI default. The
historical five-case pilot predates replacement-path transfer. A fresh
single-fact GPT-2 run found a replacement path on the imperative prompt; its
target-logit drop transferred above threshold to four of the other five OOD
families. The independent verifier accepted the run artifact. This validates
the end-to-end transfer measurement for one fact only, not a population-level
claim.

## Cross-references

- [OOD pilot report](../../../../docs/research/2026-10-30-ordered-path-ood-stress.md)
