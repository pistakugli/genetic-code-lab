# Agent Note: Measured prompt formulation probes

Status: implemented

## Context

The integrated black-box decompiler validated its executable reconstruction on
a held-out prompt wrapper, but its original training evidence came from one
prompt formulation. We needed to measure whether additional oracle-observed
formulations improve local prediction without treating prompt variation as
evidence of hidden reasoning.

## Decision

`gcl decompile --blackbox` accepts `--formulation-probes` to generate and
observe up to two generic training wrappers around the supplied prompt. Their
greedy token paths enter the saved executable reconstruction as oracle
observations, and their logical calls count toward budget allocation and
provenance. A third, separate wrapper remains held out for independent
validation. Requests above the number of available wrappers are rejected
before the oracle worker starts.

## Alternatives considered

- **Silently cap a larger requested count:** rejected because the report would
  suggest that more probes ran than were actually executed.
- **Reuse the held-out wrapper as training evidence:** rejected because that
  would contaminate the independent formulation check.
- **Claim generalization from exact prompt reproduction:** rejected because
  reproducing the measured prompt does not test a different formulation.

## Consequences

The report records each observed wrapper and generated path, and query
accounting includes the corresponding oracle calls. The current predictor still
uses nearest-observation voting outside the measured formulation domain;
adding wrapper observations does not establish a general semantic mapping
between formulations. Within that domain, the artifact now stores
formulation-specific experts with prompt prefix/suffix surfaces and
prefix-conditioned token observations. A character n-gram kernel combines
experts for a held-out wrapper and reports the contributing oracle query IDs.
The report also compares each wrapper's greedy token path with the baseline at
the first divergence. Those are whole-prompt contrasts and do not isolate the
effect of any individual wrapper token.

The saved reconstruction can retain multiple response routes with
`--prediction-beam-width`. This is a set-valued alternative, not a claim that
the highest-ranked response is correct. The two-route validation metric reports
candidate response coverage separately from exact top-response fidelity.

## Verification

`tests/test_blackbox_decision.py` verifies formulation-path inclusion,
query accounting and provenance, pairwise margin-shift calculations, wrapper
similarity selection, response-beam retention, and rejection above the two
available wrappers. Python compilation and TypeScript type-checking pass.

For Qwen2.5-0.5B-Instruct, the two-wrapper `6 + 7` run used 85 unique oracle
queries with matching worker and logical accounting. The single-route
held-out response remained incorrect, but the public `predict-blackbox`
command with beam width two returned the exact held-out response as its
top-ranked route with `oracleAccess = false`; two-route coverage was `1/3`
across all validation cases. A second arithmetic prompt had `0/3` two-route
coverage at 140 queries. Expanding to three branch probes at width three used
148 unique queries and 298 logical calls, with matching accounting, but still
did not include the held-out response in the validation set. The experiments
verify that measured alternatives can be retained; they do not verify broad
response generalization.

## Cross-references

- [Black-box decompiler](../../../../tools/llm_reader/blackbox_decompiler.py)
- [Black-box CLI wrapper](../../../../tools/llm-reader-blackbox-behavior.py)
- [NN decompiler roadmap](../../../../docs/nn-decompiler-roadmap.md)
- [Black-box decompiler tests](../../../../tests/test_blackbox_decision.py)
