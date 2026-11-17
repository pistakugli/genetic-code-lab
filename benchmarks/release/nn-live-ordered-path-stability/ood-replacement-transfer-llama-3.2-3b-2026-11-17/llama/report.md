# Live Ordered Path OOD Stress

This gate replays already-discovered real-model circuits. It does not fit a surrogate.

- Pass: `True`
- Cases: `4`
- Original replay pass rate: `1.0`
- Paraphrase profile: `ood`
- Paraphrase pass rate: `0.9166666666666666`
- Neighbor-layer specificity rate: `1.0`
- Maximum replay error: `0.125`

| subject | replay | paraphrase | layer margin | status |
|---|---:|---:|---:|---:|
| France | 0.625000 | 1.000000 | 0.500000 | passed |
| Germany | 1.000000 | 1.000000 | 1.000000 | passed |
| Italy | 0.875000 | 0.833333 | 0.750000 | passed |
| Spain | 0.250000 | 0.833333 | 0.250000 | passed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 4 | 3 | 0.750000 | 0.689732 | 21.00 |
| embedded-question | 4 | 4 | 1.000000 | 0.705357 | 20.00 |
| imperative | 4 | 3 | 0.750000 | 0.785268 | 11.00 |
| possessive-question | 4 | 4 | 1.000000 | 0.892857 | 8.00 |
| sentence-completion | 4 | 4 | 1.000000 | 0.798661 | 18.00 |
| service-question | 4 | 4 | 1.000000 | 0.949107 | 10.00 |

## Paraphrase rescue

Same-component layer-shift hints; these are controls, not replacement-path discovery.

| subject | found | best shift | broken drop | best drop |
|---|---:|---:|---:|---:|
| Italy | False | -1 | 0.375000 | 0.125000 |
| Spain | False | -3 | 0.062500 | 0.000000 |

## Replacement path rediscovery

Components are reselected on each broken prompt from the loaded source and configured rescue-shift layers. A pass requires the existing combined-ablation, ordered-path and control gates.

| subject | status | MLP nodes | attention heads | target drop | path delta | path margin |
|---|---:|---:|---:|---:|---:|---:|
| Italy | passed | 1 | 1 | 0.250000 | 0.023438 | 0.021055 |
| Spain | passed | 1 | 1 | 0.562500 | 0.007812 | 0.004247 |

## Held-out family transfer

Each replacement path is measured on the other OOD prompt families, excluding the family used for path selection.

| subject | source family | status | held-out families passed |
|---|---|---:|---:|
| Italy | imperative | passed | 5/5 |
| Spain | context-prefixed-question | passed | 5/5 |

## Why the path broke

Per broken paraphrase: why the replay failed, where the measured effect went, and whether the replacement path carries it. All values come from the recorded artifact; this is a comparative diagnosis, not a semantic claim.

### Italy

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| imperative | relative-falloff | 0.375000 | 0.428571 | yes | not recovered (best shift -1) |

**imperative** - Give the name of the capital city of Italy.

Under the imperative phrasing of the Italy prompt the original path (MLP L27 n5089 + attention L27 h5) broke as a relative-falloff: the remapped replay drop was 0.3750 against 0.8750 on the source prompt. Replaying the same components shifted across layers (-3, -2, -1) reached at best 0.1250, so the effect did not simply move with those components. The replacement search found a path (MLP L18 n4625 + attention L17 h0) with target-logit drop 0.2500; relative to the source circuit this is a component-rebinding: layers [27] -> [17, 18] (earlier), 0/2 components retained, ablate positions [5] -> [10]. On held-out families the replacement path passed 5/5.

| field | source circuit | replacement path |
|---|---|---|
| layers | [27] | [17, 18] |
| MLP components | [(27, 5089)] | [(18, 4625)] |
| attention components | [(27, 5)] | [(17, 0)] |
| ablate positions | [5] | [10] |
| target-logit drop | 0.875000 | 0.250000 |

### Spain

Measured paraphrase breaks for this subject:

| family | break kind | broken drop | ratio to original | rescue sought | layer-shift outcome |
|---|---|---|---|---|---|
| context-prefixed-question | relative-falloff | 0.062500 | 0.250000 | yes | not recovered (best shift -3) |

**context-prefixed-question** - For a geography quiz, answer with a city name only. Which city is the capital of Spain?

Under the context-prefixed-question phrasing of the Spain prompt the original path (MLP L20 n5491 + attention L20 h3) broke as a relative-falloff: the remapped replay drop was 0.0625 against 0.2500 on the source prompt. Replaying the same components shifted across layers (-3, -2, -1, 1, 2, 3) reached at best 0.0000, so the effect did not simply move with those components. The replacement search found a path (MLP L21 n3044 + attention L17 h3) with target-logit drop 0.5625; relative to the source circuit this is a component-rebinding: layers [20] -> [17, 21] (earlier), 0/2 components retained, ablate positions [5] -> [20]. On held-out families the replacement path passed 5/5.

| field | source circuit | replacement path |
|---|---|---|
| layers | [20] | [17, 21] |
| MLP components | [(20, 5491)] | [(21, 3044)] |
| attention components | [(20, 3)] | [(17, 3)] |
| ablate positions | [5] | [20] |
| target-logit drop | 0.250000 | 0.562500 |

