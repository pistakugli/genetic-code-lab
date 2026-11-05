# Agent Note: Chat template and response completeness

Status: implemented

## Context

The integrated reader encoded chat-tuned model prompts as raw text, even when
the selected tokenizer supplied a chat template. This produced responses that
continued into transcript-like text on Qwen. Separately, bounded generation
could end at `max_new_tokens` without recording that the oracle answer was
incomplete, allowing identical truncated prefixes to count as full-response
matches.

## Decision

The decompiler and saved-reconstruction predictor share a prompt encoder. Its
default `auto` mode applies the tokenizer's user-message chat template when one
is present and otherwise preserves raw-text encoding; explicit `raw` and
`chat-template` modes are available. The effective format is stored with the
reconstruction and used for prediction.

Oracle traces, reconstruction predictions, alternative routes, and validation
cases record whether generation reached the tokenizer's configured EOS or the
token budget. Full-response fidelity is calculated only over oracle outputs
that reached EOS; capped outputs remain available as bounded-prefix matches.
When an oracle response ends before its configured token cap, the remaining
safe query budget first widens single-deletion probes, then expands pairwise
probes, while retaining query budget for formulations and validation. When at
least five editable input units permit it, the decompiler selects a central
pair whose singleton contexts differ and whose joint context is unique among
input-unit pairs. The central pair is the singleton dependency holdout; its
units are excluded from single, pair, and minimal-context training probes.
When a disjoint pair can also be selected while retaining at least three
trainable units, it becomes the interaction holdout: its component singleton
probes are prioritized in training and only its exact joint pairwise probe is
withheld. Minimal-context evidence records that its search excludes the
dependency holdout, so one-minimality is not misrepresented as covering the
full editable prompt. The validation-case selector honors the preferred
interaction pair instead of replacing it with an endpoint pair. If a disjoint
pair is unavailable, the central pair remains a dependency holdout and no
separate interaction holdout is claimed. Token IDs that make component
contexts indistinguishable are rejected.
Held-out dependency assessments also record the baseline and counterfactual
prediction methods, confidence, and reconstruction evidence query IDs separately
from the oracle validation query IDs. Validation aggregates report prediction
method counts and the fraction backed by intervention-effect models, separating
causal-model coverage from generic observation or formulation voting.
Automatic pair expansion preserves an unqueried pair at every decision where
the requested probe count permits it. For unmeasured pairs, the executor
estimates interaction residuals from similar measured pair corrections using
each component's single-deletion effect signature, token overlap, and prompt
position. When exactly two editable deletion groups are present, this pair
model can also use component effects produced by calibrated local transfer to
unseen spans. Leave-one-pair-out calibration compares the residual estimate
with zero against the target pair's additive decision frontier: the winner,
top alternatives, and the tokenizer's EOS token when it is a candidate. Each
frontier margin is calibrated independently against the zero-residual
baseline; a correction is applied only when its leave-one-pair-out MAE improves
by at least the existing 10% threshold. For a rejected alternative, the
predictor applies the anchor residual to both tokens so their additive margin
is unchanged. If no margin passes, all residuals are zeroed. The artifact
records the policy and accepted/rejected margin counts. Transferred component
sources and pair-measurement provenance are retained separately. Token IDs
from pair residuals are normalized before updating scores so serialized string
keys do not create duplicate candidates.

## Alternatives considered

- **Keep raw prompts and only increase `max_new_tokens`** — rejected because a
  larger cap does not correct a chat model's missing conversation framing and
  still cannot prove completion when EOS is not reached.
- **Treat every bounded token sequence as a complete response** — rejected
  because it mislabels matching prefixes as exact full-response fidelity.
- **Hardcode a Qwen conversation wrapper** — rejected because it couples the
  shared decompiler to one model family; the tokenizer's own template is the
  adapter-independent source of prompt framing.

## Consequences

New chat-tuned model runs use their model-provided conversation format by
default, while tokenizers without a chat template keep raw behavior. Existing
reconstructions without a stored format retain raw prompt encoding when
executed. Full-response exact-match rates can now be `null` when no held-out
oracle response reaches EOS; bounded prefix equality remains separately
available. EOS remains the tokenizer's configured stop condition, not a claim
that every model-specific end marker has been discovered.

## Verification

The focused Python suite covers chat-template selection, raw override,
EOS-complete traces, token-cap truncation, incomplete-response metric handling,
adaptive context and interaction allocation, held-out input reservation,
target-frontier interaction calibration, accepted interaction-kernel transfer,
rejection when leave-one-pair-out calibration fails, and mixed-frontier
application with a nonzero common offset for rejected margins. A full local
Qwen2.5-0.5B-Instruct run for
`What is 9 - 4? Reply with only the result.` expanded interaction probes
from one to five per generated token after EOS arrived in two tokens. It used
51 unique queries within budget 512 with matching session and worker counts.
Token, rollout, and full-response fidelity were `1.0` over three complete
held-out cases; dependency accuracy was `1.0` over two non-dependency cases
and held-out pair next-token accuracy was `1.0`. Interaction residual sign
accuracy was `0.0` with mean absolute error `2.6875`, so the run does not
establish generalization of interaction magnitudes. The saved reconstruction
predicted the same complete response with `oracleAccess = false`. These
arithmetic observations cover one prompt, not generalization across model
capabilities. A separate comparison prompt generated `12` correctly, but
held-out response fidelity was `2/3` and the held-out interaction case
predicted `12` where the oracle returned `7`; this is a measured limit on
causal generalization. The new interaction kernel reduced its held-out
interaction residual MAE from `0.81745` to the zero-residual baseline `0.75`
by rejecting the uncalibrated residual estimate; it still failed to predict
the oracle's held-out `7`, and interaction sign accuracy stayed `0.0`.

A subsequent run of the same comparison prompt used a 120-query budget. The
adaptive allocator increased context probes from 4 to 12 and interaction
probes from 1 to 15 per generated-token decision, reserved two editable input
units from training interventions, and used 92 unique oracle queries with
matching session and worker accounting. The access report records
`process-separated-token-ids-to-next-token-logits`, with weights, hooks, and
activations all false and `postHocWitnessOnly = true`. Among three EOS-complete
validation responses, token and rollout fidelity were `0.875`, and full
response exact match was `2/3`. Held-out interaction next-token accuracy was
`1.0` over two cases; held-out dependency accuracy remained `0.0` over two
cases, and interaction residual sign accuracy remained `0.0` over one
measurement (MAE `1.25`). The saved reconstruction reproduced the baseline
`12` response through the oracle-free predictor, but that path used an exact
observed context. The run therefore verifies the expanded probe allocation
and one prompt's held-out interaction outputs, not dependency generalization or
general-purpose decompilation. The report and executable reconstruction are
retained as `qwen-chat-scope-comparison-v4-report.json` and
`qwen-chat-scope-comparison-v4-reconstruction.json` in the session artifact
directory.

A fresh isolated run of `Compare 8 and 13. Which number is larger? Reply with
only the number.` used 77 unique queries within budget 96, with matching
session and worker accounting. Validation aggregation counts dependency
assessments across all cases, including when a later case has no dependency
assessment. The report recorded two
`calibrated_local_intervention_transfer` predictions and intervention-model
coverage `1.0`; dependency accuracy was `1.0` over two cases. Interaction
accuracy was `1.0` over two cases, while residual sign accuracy was `0.0` on
one measurement (MAE `1.0`). Token and rollout fidelity were `0.8333`, and
full-response exact match was `0.0` over the one EOS-complete oracle response.
The results verify the aggregate and its regression test, not broad
generalization.

A fresh isolated run with calibrated transferred-span pair interactions again
used 77 unique queries within budget 96, with matching session and worker
accounting. The saved oracle-free predictor applied the pair model at the
second held-out counterfactual token, and the EOS token was included in its
calibration frontier. The counterfactual response nevertheless remained
incorrect: oracle `[16, 151645]`, reconstruction `[16, 18]`. Across the
validation set, token and rollout fidelity were `0.8333`, dependency accuracy
was `1.0` over two cases, interaction accuracy was `1.0` over two cases, and
interaction residual sign accuracy was `0.0` over one measurement (MAE `1.0`);
full-response exact match was `0.0` over one EOS-complete response. This run
verifies that the transferred interaction path executes and is evidence-linked,
but does not demonstrate improved held-out response fidelity.

Held-out counterfactual validation also compares baseline and counterfactual
oracle and reconstruction decisions at every actual generated-token prefix.
It stores per-token dependency outcomes, prediction methods and confidence,
and separate oracle evidence IDs for the baseline and counterfactual contexts.
The Markdown explanation exports these assessments. A synthetic regression
has no dependency at token one and a dependency at token two, guarding against
the prior first-token-only assessment.

A fresh isolated Qwen2.5-0.5B-Instruct run used 77 unique queries within budget
96 (142 logical calls); session and worker accounting matched. It measured
autoregressive dependency accuracy `0.5` over two held-out token positions.
At the second position, with prefix `[16]`, the oracle counterfactual emitted
EOS `151645` while the baseline emitted `18`; the reconstruction predicted
`18` for both. Token and rollout fidelity were `0.8333`, and full-response
exact match remained `0.0`. This is a measured later-token reconstruction
failure, not evidence of broad generalization.

The factorial-holdout regression uses seven distinct synthetic input units.
It verifies that the dependency and interaction pairs are disjoint, at least
three units remain trainable, both interaction-component singleton queries are
training evidence, and the joint interaction query is independent validation
evidence. Five units cannot support both holdouts while preserving the minimum
training-unit count; duplicate token IDs can also make nominally different
component contexts observationally identical.

The fresh paired Qwen run used 77 unique queries within budget 96, with
matching worker accounting. The reserved interaction pair's singleton query
IDs (`q000004`, `q000005`) were training-only and its joint query (`q000076`)
was validation-only. The oracle generated `[785, 1372]` for that joint context;
the reconstruction generated `[4498, 1249]`. Its token and rollout fidelity
were `0.5`, and its complete counterfactual response remained wrong
(`[16, 151645]` versus `[16, 18]`). The frozen report's interaction case used
a different pair, so its `1.0` interaction accuracy is not a paired comparison
to the new `0.5`. The new residual measurement is `0.75` MAE with sign
accuracy `0.0`; its component margins are supported, but this run does not
show improved predictive fidelity.

An alternate role assignment that trained the central pair's singleton
components was also tried once. On that same central joint counterfactual, it
predicted `[23, 16]` instead of `[16, 151645]`, versus `[16, 18]` before the
role swap; the first-token match regressed from one of two to zero of two.
The role swap was therefore not retained. The overall aggregate used different
holdout roles and is not a paired comparison.

Interaction residual evaluation now uses reconstructed log-scores for the
entire available candidate set, not only the displayed top ten. If either
compared token is unsupported in any required context, that residual is marked
unavailable and excluded from the residual-score denominator rather than
receiving a `-1e9` sentinel. `interactionResidualEvaluations` records the
number of fully score-supported cases.

The per-margin policy was evaluated on the same Qwen2.5-0.5B-Instruct prompt,
96-query budget, two-token cap, held-out pair, and 77 unique queries as the
frozen all-or-nothing version. Session and worker counts matched; training
component evidence was `q000004` and `q000005`, while the joint validation
observation remained independent at `q000076`. The held-out interaction
changed from oracle `[785, 1372]` versus frozen `[4498, 1249]` to prediction
`[785, 1249]`: token/rollout fidelity improved from `0.5` to `0.6667`, and
interaction next-token accuracy from `0.5` to `1.0`. Residual sign accuracy
remained `0.0` and residual MAE remained `0.75`; the separate counterfactual
still predicted `[16, 18]` against `[16, 151645]`, and full-response exact
match remained `0.0`. The saved reconstruction generated `[16, 18]` at the
same cap with `oracleAccess = false`. The report records two of three frontier
margins accepted under the unchanged per-margin threshold. This is a paired
improvement on one prompt, not evidence that the full decompiler succeeds or
generalizes broadly.

## Cross-references

- [Black-box decision trace](../../../../tools/llm_reader/blackbox_decision.py)
- [Black-box decompiler](../../../../tools/llm_reader/blackbox_decompiler.py)
- [CLI adapter](../../../../tools/llm-reader-blackbox-behavior.py)
- [Black-box decompiler tests](../../../../tests/test_blackbox_decision.py)
- [NN decompiler roadmap](../../../../docs/nn-decompiler-roadmap.md)
