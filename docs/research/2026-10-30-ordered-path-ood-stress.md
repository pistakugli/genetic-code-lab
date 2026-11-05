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

## Artifacts

- [Five-case suite report](../../benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2-5case/report.json)
- [Five-case GPT-2 raw report](../../benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2-5case/gpt2/report.json)
- [Matched standard suite report](../../benchmarks/release/nn-live-ordered-path-stability/standard-matched-gpt2-5case-6prompt/report.json)
- [Matched standard GPT-2 raw report](../../benchmarks/release/nn-live-ordered-path-stability/standard-matched-gpt2-5case-6prompt/gpt2/report.json)
- [One-case preflight report](../../benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2/gpt2/report.json)
- [Replacement-path transfer suite report](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-2026-11-04/report.json)
- [Replacement-path transfer raw report](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-2026-11-04/gpt2/report.json)
- [Independent artifact verifier](../../tools/llm_reader/verify_ordered_path_ood_stress.py)
