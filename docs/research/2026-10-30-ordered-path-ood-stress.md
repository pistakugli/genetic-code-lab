# Ordered-Path OOD Paraphrase Stress

## Purpose

This lane tests whether an already-discovered real-model ordered circuit
continues to have a causal effect when the same fact is phrased using
capital-question families not used by the existing standard paraphrase
templates. It replays the original interventions; it does not rediscover a
circuit or fit a surrogate.

The existing `standard` profile remains the default. The separate `ood`
profile uses six deterministic, answer-blind prompt families:

- service question
- possessive question
- imperative
- embedded question
- sentence completion
- context-prefixed question

The OOD helper uses the subject to generate prompts; the answer is used only
as the replay target. Each measured prompt records its token count, target-logit
drop, ratio to the original prompt, and pass/fail result. Both case-level and
family-level results are retained.

## Matched pilot setup

- Model: local GPT-2 (`gpt2`), `GPT2LMHeadModel`
- Device: CUDA (`model.device = cuda`); local-files-only loading
- Source: existing 50-fact GPT-2 ordered-residual-path artifact
- Subjects: France, Germany, Italy, Portugal, Greece
- Paraphrases: six per subject (30 replays per profile)
- Original-path replay cases: 5
- Neighbor-layer shifts: -2, -1, +1, +2
- OOD pass-rate threshold: 0.66 per case; existing original replay and layer
  specificity gates were unchanged

The standard and OOD runs use the same five source cases, model, intervention
paths, layer shifts, six paraphrases per case, and thresholds. Only the prompt
profile changes. The standard run is written to a separate matched-comparison
variant; the existing standard artifact is not overwritten.

No teacher parameters, hidden weights, or scenario labels are supplied to the
prompt-family generator. This experiment evaluates replay of a circuit
previously found by the existing reader, not blind circuit discovery.

## Results

| Metric | Standard profile | OOD profile |
|---|---:|---:|
| Strictly passed path-replay cases | 5 / 5 | 2 / 5 |
| Original replay pass rate | 100% | 100% |
| Paraphrase replays passed | 30 / 30 (100%) | 13 / 30 (43.3%) |
| Neighbor-layer specificity rate | 100% | 100% |
| Maximum original replay error | 0 | 0 |

| Prompt family | Passed | Mean drop / original |
|---|---:|---:|
| Context-prefixed question | 2 / 5 | 0.2585 |
| Embedded question | 2 / 5 | 0.1002 |
| Imperative | 0 / 5 | -0.3728 |
| Possessive question | 3 / 5 | 0.0652 |
| Sentence completion | 3 / 5 | 0.1668 |
| Service question | 3 / 5 | 0.0460 |

France and Italy passed the configured OOD case gate. Germany, Portugal, and
Greece failed it. The imperative family failed on every subject and had a
negative mean target-logit-drop ratio. The suite correctly reports
`pass: false`; the 100% original replay and layer-control rates do not override
the OOD failure.

## Interpretation and claim boundary

The matched standard baseline passes 30/30 paraphrase replays; the OOD profile
passes 13/30. This pilot therefore does **not** support the claim that the
discovered path is robust to these OOD paraphrases. It does show that the
failure is family-dependent: the original circuit replay and neighboring-layer
controls remain stable while OOD replay often does not. That is diagnostic
evidence about prompt sensitivity, not proof that the model's behavior or all
equivalent circuits are absent.

The six families are manually authored templates for capital questions, and
the pilot covers five facts from one model. It does not establish general
semantic OOD robustness, broad language coverage, circuit completeness, or
stability across model families. The next useful experiment is to hold the
template families fixed and expand the independently selected fact set and
model adapters, rather than tuning templates or thresholds to this result.

## Replacement-path cross-family transfer

The stability runner now measures a newly discovered replacement path on the
other OOD prompt families for the same fact, excluding the family used during
path selection. It remaps the read-token intervention position for each held
out prompt and records every target-logit drop, discovery-relative ratio, and
threshold decision. This transfer score is diagnostic and does not change the
original-path OOD gate.

The historical five-case report above predates this addition. A fresh
single-fact GPT-2 run on France selected the imperative prompt for rediscovery.
The replacement path's discovery target-logit drop was `1.90994`; on the other
five OOD families it passed the configured drop and relative-ratio thresholds
on four (`4/5`, pass rate `0.8`). The service-question transfer was below the
relative threshold (`0.0813` vs `0.1`) despite a positive target-logit drop.
The independent artifact verifier passed. This is one fact and one discovered
path, not evidence of population-level or cross-model transfer.

Unit tests also cover family exclusion, position remapping, threshold
accounting, no-path abstention, and verifier rejection of inconsistent raw
measurements.

Runtime and VRAM were not measured in this lane. CUDA use is verified from the
model report; it is not evidence of a speedup.

## Why-report layer

The rescue/replacement lane records what happened when a paraphrase breaks a
path: the broken replay drop, layer-shift attempts, the rediscovered
replacement path, and its held-out family transfer. It did not explain the
finding. The north-star reader step "explain why selected components react"
requires the artifact to say why the path broke and where the measured effect
went, not only that it broke.

The stability runner now attaches a why-report to every case with rescue
measurements **or at least one broken paraphrase**, computed by the
pure-stdlib module
[`llm_reader/why_report.py`](../../tools/llm_reader/why_report.py) from values
already recorded in the artifact (no model queries, no new measurements):

- break classification: `sign-flip` (ablation raises the target logit, so the
  recorded components are counterproductive under the new phrasing),
  `magnitude-collapse` (positive drop below the absolute gate), or
  `relative-falloff` (clears the absolute gate but misses the configured
  ratio);
- layer-shift rescue outcome: whether shifting the same components across
  layers recovered the effect, which distinguishes a layer misplacement from a
  component re-binding;
- replacement-path comparison against the source circuit: source vs
  replacement layers, mean layer direction (earlier/later/same), MLP and
  attention component overlap, ablate-position movement, and a change kind
  (`component-retention`, `position-rebinding`, `layer-local-rebinding`,
  `component-rebinding`);
- transfer diagnosis for the replacement path: which held-out families failed
  and which threshold (absolute drop vs discovery-relative ratio) they missed;
- a narrative `whyStatement` assembled only from recorded numbers, plus a
  per-entry claim boundary;
- a per-paraphrase break classification for **every** broken paraphrase, not
  only the ones the run selected for a rescue search, with an honest
  `rescueSought` flag and, when a rescue search exists, its layer-shift
  outcome;
- an `attempted` flag on the replacement comparison so a historical run
  without a replacement search reads "not run" instead of the misleading
  "did not find".

The markdown report gains a "Why the path broke" section with the same
content, and the artifact config gains a `whyReportEnabled` marker. The
independent verifier recomputes the classification, overlap counts, change
kind, direction, and transfer diagnosis from the raw rescue measurements, so a
fabricated or stale why block fails verification; artifacts recorded before
this layer (no marker) keep verifying under the previous rules.

Applied to the recorded France replacement-transfer artifact: the imperative
break is classified as a `sign-flip` (broken drop `-0.8927` against the source
`1.8188`), the layer-shift attempts all remain negative, and the replacement
path is a `component-rebinding` that moves the effect earlier (layers `9/11`
to `6`, positions `4` to `9`, zero of two components retained) with `4/5`
held-out family transfer (service-question below the relative ratio). That is
the previously missing "why and where" statement for that recorded finding,
produced by the repo's own gate tooling rather than by hand.

Claim boundary for the why-report layer: it is a comparative diagnosis of
recorded interventions between two prompts of one subject. It is not semantic
proof of what components compute, not proof of circuit uniqueness, and not
evidence beyond the measured families, subjects, and models.

## Family-level break atlas

The why-report explains one break at a time. The atlas layer
([`llm_reader/why_atlas.py`](../../tools/llm_reader/why_atlas.py)) aggregates
**every recorded paraphrase replay** across the recorded artifacts into a
family-level break taxonomy, recomputed from the raw fields and verified by an
independent recomputation, so a fabricated atlas fails `verify_break_atlas`.

Applied to the four recorded GPT-2 artifacts (five-case OOD suite, one-case
preflight, replacement-transfer run, and the matched standard baseline), the
atlas restates the whole recorded population:

- 72 measured paraphrase replays: 53 passed, 19 broken;
- break kinds across all 19 breaks: 18 `sign-flip`, 1 `relative-falloff`,
  0 `magnitude-collapse` — when a recorded OOD replay broke, ablation almost
  always *raised* the target logit instead of merely fading;
- the `imperative` family broke 7/7 measured replays, all sign-flips (mean
  broken ratio `-0.41`); no other family broke more than 3/7;
- the matched standard-profile baseline broke 0/30, so the taxonomy is a
  property of the OOD phrasings under this protocol, not of the replay
  machinery;
- rescue coverage: only 10 of the 19 broken paraphrases were rescue-searched
  in the recorded runs; of those 10 searches, 6 recovered the effect by
  shifting the same components to a neighbouring layer (layer misplacement),
  while the 4 failures are the Germany embedded-question break and the France
  imperative break (recorded in three separate runs). The one replacement
  search that ran found a `component-rebinding` (the France case above).

That population statement corrects an implicit bias in the earlier per-case
reading: the France replacement-transfer case, which motivated the why-report
layer, is *atypical* — for the majority of the rescue-searched breaks in the
recorded population the measured effect simply moved with the components to a
neighbouring layer. The atlas keeps that visible by reporting rescue coverage
(10/19) rather than generalising the searched subset.

Claim boundary for the atlas: it aggregates the four listed artifacts of one
model and one protocol; three of them share the France subject, and the
rescue-searched subset is not a random sample of breaks. It is a restatement
of recorded numbers, not a semantic claim about prompt families and not
evidence about unmeasured subjects, families, or models.

## Artifacts

- [Five-case suite report](../../benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2-5case/report.json)
- [Five-case GPT-2 raw report](../../benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2-5case/gpt2/report.json)
- [Matched standard suite report](../../benchmarks/release/nn-live-ordered-path-stability/standard-matched-gpt2-5case-6prompt/report.json)
- [Matched standard GPT-2 raw report](../../benchmarks/release/nn-live-ordered-path-stability/standard-matched-gpt2-5case-6prompt/gpt2/report.json)
- [One-case preflight report](../../benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2/gpt2/report.json)
- [Replacement-path transfer suite report](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-2026-11-04/report.json)
- [Replacement-path transfer raw report](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-2026-11-04/gpt2/report.json)
- [Independent artifact verifier](../../tools/llm_reader/verify_ordered_path_ood_stress.py)
- [Family-level break atlas module](../../tools/llm_reader/why_atlas.py)
