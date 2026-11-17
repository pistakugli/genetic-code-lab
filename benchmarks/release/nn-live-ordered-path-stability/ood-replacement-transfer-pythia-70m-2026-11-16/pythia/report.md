# Live Ordered Path OOD Stress

This gate replays already-discovered real-model circuits. It does not fit a surrogate.

- Pass: `False`
- Cases: `5`
- Original replay pass rate: `1.0`
- Paraphrase profile: `ood`
- Paraphrase pass rate: `0.43333333333333335`
- Neighbor-layer specificity rate: `1.0`
- Maximum replay error: `0.0`

| subject | replay | paraphrase | layer margin | status |
|---|---:|---:|---:|---:|
| Spain | 2.000000 | 0.333333 | 2.000000 | failed |
| Japan | 2.000000 | 0.333333 | 2.000000 | failed |
| Russia | 2.000000 | 1.000000 | 1.000000 | passed |
| Norway | 2.000000 | 0.166667 | 2.000000 | failed |
| Austria | 2.000000 | 0.333333 | 2.000000 | failed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 5 | 2 | 0.400000 | 0.100000 | 20.00 |
| embedded-question | 5 | 1 | 0.200000 | 0.100000 | 19.00 |
| imperative | 5 | 5 | 1.000000 | 5.900000 | 10.00 |
| possessive-question | 5 | 2 | 0.400000 | 0.300000 | 7.00 |
| sentence-completion | 5 | 2 | 0.400000 | 0.300000 | 17.00 |
| service-question | 5 | 1 | 0.200000 | 0.000000 | 9.00 |

## Paraphrase rescue

Same-component layer-shift hints; these are controls, not replacement-path discovery.

| subject | found | best shift | broken drop | best drop |
|---|---:|---:|---:|---:|
| Spain | True | 1 | -1.000000 | 1.000000 |
| Spain | False | 1 | 0.000000 | 0.000000 |
| Japan | False | 1 | 0.000000 | 0.000000 |
| Japan | False | 1 | 0.000000 | 0.000000 |
| Norway | False | 1 | -1.000000 | 0.000000 |
| Norway | False | 1 | 0.000000 | 0.000000 |
| Austria | True | 1 | -1.000000 | 1.000000 |
| Austria | False | 1 | 0.000000 | 0.000000 |

## Replacement path rediscovery

Components are reselected on each broken prompt from the loaded source and configured rescue-shift layers. A pass requires the existing combined-ablation, ordered-path and control gates.

| subject | status | MLP nodes | attention heads | target drop | path delta | path margin |
|---|---:|---:|---:|---:|---:|---:|
| Spain | failed | 1 | 2 | 5.000000 | 1.593750 | 1.593750 |
| Spain | failed | 0 | 0 | 0.000000 | n/a | n/a |
| Japan | failed | 1 | 2 | 0.000000 | 0.994148 | 0.994148 |
| Japan | failed | 0 | 0 | 0.000000 | n/a | n/a |
| Norway | failed | 0 | 0 | 0.000000 | n/a | n/a |
| Norway | failed | 0 | 0 | 0.000000 | n/a | n/a |
| Austria | failed | 1 | 2 | 1.000000 | 5.917969 | 5.768628 |
| Austria | failed | 0 | 0 | 0.000000 | n/a | n/a |

## Held-out family transfer

Each replacement path is measured on the other OOD prompt families, excluding the family used for path selection.

| subject | source family | status | held-out families passed |
|---|---|---:|---:|
| Spain | service-question | not_applicable | 0/0 |
| Spain | possessive-question | not_applicable | 0/0 |
| Japan | service-question | not_applicable | 0/0 |
| Japan | embedded-question | not_applicable | 0/0 |
| Norway | service-question | not_applicable | 0/0 |
| Norway | possessive-question | not_applicable | 0/0 |
| Austria | service-question | not_applicable | 0/0 |
| Austria | possessive-question | not_applicable | 0/0 |

## Why the path broke

Per broken paraphrase: why the replay failed, where the measured effect went, and whether the replacement path carries it. All values come from the recorded artifact; this is a comparative diagnosis, not a semantic claim.

### Spain

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| service-question | sign-flip | -1.000000 | -0.500000 | yes | recovered (shift 1) |
| possessive-question | magnitude-collapse | 0.000000 | 0.000000 | yes | not recovered (best shift 1) |
| embedded-question | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |
| sentence-completion | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |

**service-question** - Which city serves as the capital of Spain?

Under the service-question phrasing of the Spain prompt the original path (MLP L3 n983 + attention L0 h6) broke as a sign-flip: the remapped replay drop was -1.0000 against 2.0000 on the source prompt. A layer shift of 1 alone recovered the effect. The replacement search did not find a passing path.

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 3] | [1, 5] |
| MLP components | [(3, 983)] | [(5, 1132)] |
| attention components | [(0, 6)] | [(1, 0), (1, 4)] |
| ablate positions | [4] | [8] |
| target-logit drop | 2.000000 | 5.000000 |

**possessive-question** - What city is Spain's capital?

Under the possessive-question phrasing of the Spain prompt the original path (MLP L3 n983 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 3] | [] |
| MLP components | [(3, 983)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

### Japan

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| service-question | magnitude-collapse | 0.000000 | 0.000000 | yes | not recovered (best shift 1) |
| embedded-question | magnitude-collapse | 0.000000 | 0.000000 | yes | not recovered (best shift 1) |
| sentence-completion | sign-flip | -1.000000 | -0.500000 | no | not sought |
| context-prefixed-question | sign-flip | -1.000000 | -0.500000 | no | not sought |

**service-question** - Which city serves as the capital of Japan?

Under the service-question phrasing of the Japan prompt the original path (MLP L3 n92, L3 n1348 + attention L0 h6, L2 h1) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path.

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 2, 3] | [0, 2, 3] |
| MLP components | [(3, 92), (3, 1348)] | [(3, 92)] |
| attention components | [(0, 6), (2, 1)] | [(0, 0), (2, 1)] |
| ablate positions | [4] | [8] |
| target-logit drop | 2.000000 | 0.000000 |

**embedded-question** - I am checking a country profile. Could you tell me which city is the capital of Japan?

Under the embedded-question phrasing of the Japan prompt the original path (MLP L3 n92, L3 n1348 + attention L0 h6, L2 h1) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 2, 3] | [] |
| MLP components | [(3, 92), (3, 1348)] | [] |
| attention components | [(0, 6), (2, 1)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

### Norway

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| service-question | sign-flip | -1.000000 | -0.500000 | yes | not recovered (best shift 1) |
| possessive-question | magnitude-collapse | 0.000000 | 0.000000 | yes | not recovered (best shift 1) |
| embedded-question | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |
| sentence-completion | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |
| context-prefixed-question | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |

**service-question** - Which city serves as the capital of Norway?

Under the service-question phrasing of the Norway prompt the original path (MLP L1 n1023 + attention L0 h6) broke as a sign-flip: the remapped replay drop was -1.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 1023)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

**possessive-question** - What city is Norway's capital?

Under the possessive-question phrasing of the Norway prompt the original path (MLP L1 n1023 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 1023)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

### Austria

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| service-question | sign-flip | -1.000000 | -0.500000 | yes | recovered (shift 1) |
| possessive-question | magnitude-collapse | 0.000000 | 0.000000 | yes | not recovered (best shift 1) |
| embedded-question | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |
| context-prefixed-question | magnitude-collapse | 0.000000 | 0.000000 | no | not sought |

**service-question** - Which city serves as the capital of Austria?

Under the service-question phrasing of the Austria prompt the original path (MLP L1 n699 + attention L0 h6) broke as a sign-flip: the remapped replay drop was -1.0000 against 2.0000 on the source prompt. A layer shift of 1 alone recovered the effect. The replacement search did not find a passing path.

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 1] | [2, 3, 5] |
| MLP components | [(1, 699)] | [(5, 1476)] |
| attention components | [(0, 6)] | [(2, 3), (3, 2)] |
| ablate positions | [4] | [8] |
| target-logit drop | 2.000000 | 1.000000 |

**possessive-question** - What city is Austria's capital?

Under the possessive-question phrasing of the Austria prompt the original path (MLP L1 n699 + attention L0 h6) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 2.0000 on the source prompt. Replaying the same components shifted across layers (1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search did not find a passing path (ordered path requires selected MLP and attention interventions).

| field | source circuit | replacement path |
|---|---|---|
| layers | [0, 1] | [] |
| MLP components | [(1, 699)] | [] |
| attention components | [(0, 6)] | [] |
| ablate positions | [4] | [] |
| target-logit drop | 2.000000 | 0.000000 |

