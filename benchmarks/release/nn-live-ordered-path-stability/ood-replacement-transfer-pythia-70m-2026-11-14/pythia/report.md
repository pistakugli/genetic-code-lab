# Live Ordered Path OOD Stress

This gate replays already-discovered real-model circuits. It does not fit a surrogate.

- Pass: `False`
- Cases: `1`
- Original replay pass rate: `1.0`
- Paraphrase profile: `ood`
- Paraphrase pass rate: `0.16666666666666666`
- Neighbor-layer specificity rate: `1.0`
- Maximum replay error: `0.0`

| subject | replay | paraphrase | layer margin | status |
|---|---:|---:|---:|---:|
| France | 2.000000 | 0.166667 | 2.000000 | failed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 1 | 0 | 0.000000 | 0.000000 | 20.00 |
| embedded-question | 1 | 0 | 0.000000 | 0.000000 | 19.00 |
| imperative | 1 | 1 | 1.000000 | 8.500000 | 10.00 |
| possessive-question | 1 | 0 | 0.000000 | 0.000000 | 7.00 |
| sentence-completion | 1 | 0 | 0.000000 | 0.000000 | 17.00 |
| service-question | 1 | 0 | 0.000000 | 0.000000 | 9.00 |

## Paraphrase rescue

Same-component layer-shift hints; these are controls, not replacement-path discovery.

| subject | found | best shift | broken drop | best drop |
|---|---:|---:|---:|---:|
| France | True | 1 | 0.000000 | 1.000000 |
| France | False | 1 | 0.000000 | 0.000000 |
| France | False | 1 | 0.000000 | 0.000000 |
| France | False | 1 | 0.000000 | 0.000000 |
| France | False | 1 | 0.000000 | 0.000000 |

## Replacement path rediscovery

Components are reselected on each broken prompt from the loaded source and configured rescue-shift layers. A pass requires the existing combined-ablation, ordered-path and control gates.

| subject | status | MLP nodes | attention heads | target drop | path delta | path margin |
|---|---:|---:|---:|---:|---:|---:|
| France | failed | 1 | 2 | 1.000000 | 0.776367 | 0.776367 |
| France | failed | 0 | 0 | 0.000000 | n/a | n/a |
| France | failed | 0 | 0 | 0.000000 | n/a | n/a |
| France | failed | 0 | 0 | 0.000000 | n/a | n/a |
| France | failed | 0 | 0 | 0.000000 | n/a | n/a |

## Held-out family transfer

Each replacement path is measured on the other OOD prompt families, excluding the family used for path selection.

| subject | source family | status | held-out families passed |
|---|---|---:|---:|
| France | service-question | not_applicable | 0/0 |
| France | possessive-question | not_applicable | 0/0 |
| France | embedded-question | not_applicable | 0/0 |
| France | sentence-completion | not_applicable | 0/0 |
| France | context-prefixed-question | not_applicable | 0/0 |

## Why the path broke

Per broken paraphrase: why the replay failed, where the measured effect went, and whether the replacement path carries it. All values come from the recorded artifact; this is a comparative diagnosis, not a semantic claim.

### France

**service-question** - Which city serves as the capital of France?

Under the service-question phrasing of the France prompt the original path (MLP L1 n309 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. A layer shift of 1 alone recovered the effect. The replacement search did not find a passing path. The search's best rejected candidate (MLP L4 n234 + attention L2 h4, L3 h4) with target-logit drop 1.0000 did not pass the replacement gates, so its comparison below describes a candidate, not a carried effect.

| field | source circuit | best rejected candidate |
|---|---|---|
| layers | [0, 1] | [2, 3, 4] |
| MLP components | [(1, 309)] | [(4, 234)] |
| attention components | [(0, 6)] | [(2, 4), (3, 4)] |
| ablate positions | [4] | [8] |
| target-logit drop | 2.000000 | 1.000000 |

**possessive-question** - What city is France's capital?

Under the possessive-question phrasing of the France prompt the original path (MLP L1 n309 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | no candidate recorded |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 309)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

**embedded-question** - I am checking a country profile. Could you tell me which city is the capital of France?

Under the embedded-question phrasing of the France prompt the original path (MLP L1 n309 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | no candidate recorded |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 309)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

**sentence-completion** - Complete this sentence with a city name: The city that is the capital of France is

Under the sentence-completion phrasing of the France prompt the original path (MLP L1 n309 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | no candidate recorded |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 309)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

**context-prefixed-question** - For a geography quiz, answer with a city name only. Which city is the capital of France?

Under the context-prefixed-question phrasing of the France prompt the original path (MLP L1 n309 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | no candidate recorded |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 309)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

