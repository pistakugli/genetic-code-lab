# Output-Only OOD Paraphrase Behavior Across Model Families

## Purpose

The output/logit-only behavior rail (roadmap item 5,
[`nn-decompiler-roadmap.md`](../nn-decompiler-roadmap.md)) had one recorded
gate: a GPT-2 10-fact factual behavior graph over standard same-relation
paraphrase templates, selected by margin. The roadmap's next gates for this
rail were a larger fact set, the OOD paraphrase families, and
Pythia/LLaMA/Mistral parity, all through the unchanged
prompt-to-next-token-logits-only solver boundary. This lane records the first
two gates on GPT-2 and the Pythia-70m family parity, on CPU, in one session:

- GPT-2 (`gpt2`), full 20-fact pool, OOD paraphrase profile,
- Pythia-70m (`EleutherAI/pythia-70m-deduped`, GPT-NeoX family), same protocol.

The runner records a pass or a fail honestly; the Pythia family failure below
is a finding, not a rerun target.

## Protocol

The solver boundary is unchanged: the black-box behavior reader receives
prompt strings, target token ids, and next-token logits only. No weights,
hooks, hidden activations, source ids, or ablation handles cross the
boundary; the access audit fields stay `prompt-to-next-token-logits-only`.

The paraphrase protocol changes from the standard profile in one way: the
six deterministic answer-blind OOD capital-question families from
[`llm_reader/ood_paraphrases.py`](../../tools/llm_reader/ood_paraphrases.py)
(service-question, possessive-question, imperative, embedded-question,
sentence-completion, context-prefixed-question) replace the same-relation
completion templates. Every family prompt of every fact is scored and kept:
there is no margin-based selection, so the gate cannot cherry-pick an easy
phrasing, and the per-family table restates the whole recorded population.
A family item is retained when the target answer keeps a strictly positive
answer-pool margin under the new phrasing; the per-case and suite gates
(retention rate, relation-template margin, subject lift, answer margin,
target rank) keep the standard-profile thresholds.

Token-validity limits are recorded, not hidden: Brasilia is not a single
GPT-2 token and Canberra and Brasilia are not single Pythia-70m tokens, so
the artifact reports 19 valid GPT-2 facts (one skipped, `skippedCases=1`)
and 18 valid Pythia facts (two skipped) out of the same full 20-fact pool.

## Results

### GPT-2: full-pool OOD pass

19 valid facts, 19 passed (`gpt2-output-only-ood-20-facts-behavior-graph`,
380 oracle queries). Median baseline target rank 7, max 23; median
paraphrase retention 1.0; minimum relation-template margin 0.672 against the
0.02 gate.

| family | retained | retention rate | mean margin | min margin | median target rank |
|---|---:|---:|---:|---:|---:|
| service-question | 18/19 | 0.947 | 1.864 | -0.396 | 31 |
| possessive-question | 19/19 | 1.000 | 2.739 | 1.054 | 7 |
| imperative | 19/19 | 1.000 | 1.717 | 0.433 | 108 |
| embedded-question | 19/19 | 1.000 | 2.080 | 0.521 | 16 |
| sentence-completion | 18/19 | 0.947 | 4.318 | -0.468 | 2 |
| context-prefixed-question | 19/19 | 1.000 | 2.511 | 0.915 | 12 |

The only broken family items in the whole population are Austria's
service-question (-0.396) and sentence-completion (-0.468); Austria still
retains 4/6 and passes. The imperative family keeps a positive margin
everywhere (median target rank 108) even though it is the weakest family by
rank: the answer survives the phrasing without being the top prediction.

### Pythia-70m: an honestly recorded family-level failure

18 valid facts, 7 passed (France, Germany, Spain, Japan, China, Russia,
Sweden), 11 failed; 360 oracle queries. Median baseline target rank 95.5,
max 730 (Portugal exceeds the 100 rank gate). Pythia-70m margins are coarse
integer-valued logit differences, so several facts sit exactly at margin 0
and fail the 0.02 relation gate.

| family | retained | retention rate | mean margin | min margin | median target rank |
|---|---:|---:|---:|---:|---:|
| service-question | 3/18 | 0.167 | -0.167 | -3.000 | 884 |
| possessive-question | 17/18 | 0.944 | 1.778 | -1.000 | 21 |
| imperative | 9/18 | 0.500 | 0.500 | -2.000 | 452 |
| embedded-question | 4/18 | 0.222 | -1.111 | -3.000 | 4518 |
| sentence-completion | 11/18 | 0.611 | 0.889 | -1.000 | 14 |
| context-prefixed-question | 2/18 | 0.111 | -1.222 | -3.000 | 1466 |

## Interpretation and claim boundary

Cross-family reading, bounded to what was measured: through the identical
logit-only boundary and the identical answer-blind family protocol, the base
web-text-trained GPT-2 keeps a positive answer-pool margin under every OOD
family for 18/19 facts, while the Pile-trained Pythia-70m retains mainly the
possessive-question phrasing (17/18) and collapses on the question-style
families (2-4 of 18). Which template family preserves the answer is
model-relative, not template-intrinsic - the same conclusion the hook-based
ordered-path OOD replacement lane measured for Pythia-70m from the
intervention side (paraphrase pass rate 0.43, service-question 1/5). Two
access boundaries, one family-level finding, recorded from both sides.

The Pythia run also measures the practical floor of the logit-only rail: a
70M-parameter model whose factual margins are coarse integer logits cannot
support the 0.02 margin gates for most capital facts regardless of phrasing
(median baseline rank 95.5 against GPT-2's 7).

This lane does not prove semantic understanding, instruction-following
ability, circuit identity, robustness beyond the six capital-question
templates and the selected 20-fact pool, or anything about instruct-tuned
families; those are the next recorded gate (LLaMA-3.2 / Mistral-7B on Kaggle
T4 through the same boundary). The family table is a restatement of recorded
measurements, not a claim about unmeasured subjects or models.

## Artifacts

- GPT-2 OOD report: [`gpt2-output-only-ood-20-facts-behavior-graph/report.json`](../../benchmarks/release/nn-live-blackbox-llm-behavior-reader/gpt2-output-only-ood-20-facts-behavior-graph/report.json)
- Pythia-70m OOD report: [`pythia-70m-output-only-ood-20-facts-behavior-graph/report.json`](../../benchmarks/release/nn-live-blackbox-llm-behavior-reader/pythia-70m-output-only-ood-20-facts-behavior-graph/report.json)
- GPT-2 standard reference: [`gpt2-output-only-10-facts-behavior-graph/report.json`](../../benchmarks/release/nn-live-blackbox-llm-behavior-reader/gpt2-output-only-10-facts-behavior-graph/report.json)
- Runner: [`llm-reader-blackbox-behavior.py`](../../tools/llm-reader-blackbox-behavior.py) (`--paraphrase-profile standard|ood`)
- OOD family templates: [`llm_reader/ood_paraphrases.py`](../../tools/llm_reader/ood_paraphrases.py)
- Cross-family hook-based lane: [`2026-11-17-ood-replacement-transfer-cross-family.md`](2026-11-17-ood-replacement-transfer-cross-family.md)
