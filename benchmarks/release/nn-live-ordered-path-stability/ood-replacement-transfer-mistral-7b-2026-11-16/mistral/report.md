# Live Ordered Path OOD Stress

This gate replays already-discovered real-model circuits. It does not fit a surrogate.

- Pass: `True`
- Cases: `4`
- Original replay pass rate: `1.0`
- Paraphrase profile: `ood`
- Paraphrase pass rate: `0.8333333333333334`
- Neighbor-layer specificity rate: `1.0`
- Maximum replay error: `0.0625`

| subject | replay | paraphrase | layer margin | status |
|---|---:|---:|---:|---:|
| France | 0.500000 | 1.000000 | 0.500000 | passed |
| Germany | 0.437500 | 0.666667 | 0.437500 | passed |
| Spain | 0.187500 | 0.833333 | 0.062500 | passed |
| Japan | 0.125000 | 0.833333 | 0.062500 | passed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 4 | 3 | 0.750000 | 0.385417 | 23.00 |
| embedded-question | 4 | 3 | 0.750000 | 0.302083 | 20.00 |
| imperative | 4 | 4 | 1.000000 | 0.514881 | 11.00 |
| possessive-question | 4 | 4 | 1.000000 | 0.732143 | 9.00 |
| sentence-completion | 4 | 2 | 0.500000 | 0.223214 | 18.00 |
| service-question | 4 | 4 | 1.000000 | 0.613095 | 10.00 |

## Paraphrase rescue

Same-component layer-shift hints; these are controls, not replacement-path discovery.

| subject | found | best shift | broken drop | best drop |
|---|---:|---:|---:|---:|
| Germany | True | -1 | 0.000000 | 0.062500 |
| Germany | True | -3 | 0.000000 | 0.125000 |
| Spain | False | -2 | 0.000000 | 0.000000 |
| Japan | True | 1 | 0.000000 | 0.062500 |

## Replacement path rediscovery

Components are reselected on each broken prompt from the loaded source and configured rescue-shift layers. A pass requires the existing combined-ablation, ordered-path and control gates.

| subject | status | MLP nodes | attention heads | target drop | path delta | path margin |
|---|---:|---:|---:|---:|---:|---:|
| Germany | passed | 1 | 1 | 0.312500 | 0.250000 | 0.153125 |
| Germany | passed | 1 | 1 | 0.250000 | 0.031250 | 0.009033 |
| Spain | passed | 1 | 1 | 0.687500 | 0.093750 | 0.048438 |
| Japan | passed | 1 | 1 | 0.312500 | 0.203125 | 0.197461 |

## Held-out family transfer

Each replacement path is measured on the other OOD prompt families, excluding the family used for path selection.

| subject | source family | status | held-out families passed |
|---|---|---:|---:|
| Germany | embedded-question | failed | 4/5 |
| Germany | context-prefixed-question | passed | 5/5 |
| Spain | sentence-completion | failed | 2/5 |
| Japan | sentence-completion | failed | 2/5 |

## Why the path broke

Per broken paraphrase: why the replay failed, where the measured effect went, and whether the replacement path carries it. All values come from the recorded artifact; this is a comparative diagnosis, not a semantic claim.

### Germany

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| embedded-question | magnitude-collapse | 0.000000 | 0.000000 | yes | recovered (shift -1) |
| context-prefixed-question | magnitude-collapse | 0.000000 | 0.000000 | yes | recovered (shift -3) |

**embedded-question** - I am checking a country profile. Could you tell me which city is the capital of Germany?

Under the embedded-question phrasing of the Germany prompt the original path (MLP L31 n9943 + attention L20 h24) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 0.4375 on the source prompt. A layer shift of -1 alone recovered the effect. The replacement search found a path (MLP L31 n8967 + attention L31 h22) with target-logit drop 0.3125; relative to the source circuit this is a component-rebinding: layers [20, 31] -> [31] (later), 0/2 components retained, ablate positions [5] -> [19]. On held-out families the replacement path passed 4/5; failing families: sentence-completion.

| field | source circuit | replacement path |
|---|---|---|
| layers | [20, 31] | [31] |
| MLP components | [(31, 9943)] | [(31, 8967)] |
| attention components | [(20, 24)] | [(31, 22)] |
| ablate positions | [5] | [19] |
| target-logit drop | 0.437500 | 0.312500 |

**context-prefixed-question** - For a geography quiz, answer with a city name only. Which city is the capital of Germany?

Under the context-prefixed-question phrasing of the Germany prompt the original path (MLP L31 n9943 + attention L20 h24) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 0.4375 on the source prompt. A layer shift of -3 alone recovered the effect. The replacement search found a path (MLP L28 n6042 + attention L17 h22) with target-logit drop 0.2500; relative to the source circuit this is a component-rebinding: layers [20, 31] -> [17, 28] (earlier), 0/2 components retained, ablate positions [5] -> [22]. On held-out families the replacement path passed 5/5.

| field | source circuit | replacement path |
|---|---|---|
| layers | [20, 31] | [17, 28] |
| MLP components | [(31, 9943)] | [(28, 6042)] |
| attention components | [(20, 24)] | [(17, 22)] |
| ablate positions | [5] | [22] |
| target-logit drop | 0.437500 | 0.250000 |

### Spain

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| sentence-completion | magnitude-collapse | 0.000000 | 0.000000 | yes | not recovered (best shift -2) |

**sentence-completion** - Complete this sentence with a city name: The city that is the capital of Spain is

Under the sentence-completion phrasing of the Spain prompt the original path (MLP L24 n8600 + attention L20 h21) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 0.1875 on the source prompt. Replaying the same components shifted across layers (-3, -2, -1, 1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search found a path (MLP L18 n3733 + attention L17 h19) with target-logit drop 0.6875; relative to the source circuit this is a component-rebinding: layers [20, 24] -> [17, 18] (earlier), 0/2 components retained, ablate positions [5] -> [17]. On held-out families the replacement path passed 2/5; failing families: service-question, imperative, embedded-question.

| field | source circuit | replacement path |
|---|---|---|
| layers | [20, 24] | [17, 18] |
| MLP components | [(24, 8600)] | [(18, 3733)] |
| attention components | [(20, 21)] | [(17, 19)] |
| ablate positions | [5] | [17] |
| target-logit drop | 0.187500 | 0.687500 |

### Japan

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| sentence-completion | magnitude-collapse | 0.000000 | 0.000000 | yes | recovered (shift 1) |

**sentence-completion** - Complete this sentence with a city name: The city that is the capital of Japan is

Under the sentence-completion phrasing of the Japan prompt the original path (MLP L20 n7999 + attention L20 h21) broke as a magnitude-collapse: the remapped replay drop was 0.0000 against 0.1250 on the source prompt. A layer shift of 1 alone recovered the effect. The replacement search found a path (MLP L23 n6542 + attention L19 h29) with target-logit drop 0.3125; relative to the source circuit this is a component-rebinding: layers [20] -> [19, 23] (later), 0/2 components retained, ablate positions [5] -> [17]. On held-out families the replacement path passed 2/5; failing families: possessive-question, imperative, embedded-question.

| field | source circuit | replacement path |
|---|---|---|
| layers | [20] | [19, 23] |
| MLP components | [(20, 7999)] | [(23, 6542)] |
| attention components | [(20, 21)] | [(19, 29)] |
| ablate positions | [5] | [17] |
| target-logit drop | 0.125000 | 0.312500 |

