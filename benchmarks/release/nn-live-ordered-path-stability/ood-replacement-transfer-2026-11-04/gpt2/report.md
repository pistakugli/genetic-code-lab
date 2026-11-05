# Live Ordered Path OOD Stress

This gate replays already-discovered real-model circuits. It does not fit a surrogate.

- Pass: `True`
- Cases: `1`
- Original replay pass rate: `1.0`
- Paraphrase profile: `ood`
- Paraphrase pass rate: `0.8333333333333334`
- Neighbor-layer specificity rate: `1.0`
- Maximum replay error: `0.0`

| subject | replay | paraphrase | layer margin | status |
|---|---:|---:|---:|---:|
| France | 1.818840 | 0.833333 | 1.464195 | passed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 1 | 1 | 1.000000 | 0.850893 | 20.00 |
| embedded-question | 1 | 1 | 1.000000 | 1.499360 | 19.00 |
| imperative | 1 | 0 | 0.000000 | -0.490782 | 10.00 |
| possessive-question | 1 | 1 | 1.000000 | 0.610598 | 7.00 |
| sentence-completion | 1 | 1 | 1.000000 | 1.793552 | 17.00 |
| service-question | 1 | 1 | 1.000000 | 0.361923 | 9.00 |

## Paraphrase rescue

Same-component layer-shift hints; these are controls, not replacement-path discovery.

| subject | found | best shift | broken drop | best drop |
|---|---:|---:|---:|---:|
| France | False | -3 | -0.892654 | -0.093651 |

## Replacement path rediscovery

Components are reselected on each broken prompt from the loaded source and configured rescue-shift layers. A pass requires the existing combined-ablation, ordered-path and control gates.

| subject | status | MLP nodes | attention heads | target drop | path delta | path margin |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | 1 | 1 | 1.909943 | 0.069018 | 0.046838 |

## Held-out family transfer

Each replacement path is measured on the other OOD prompt families, excluding the family used for path selection.

| subject | source family | status | held-out families passed |
|---|---|---:|---:|
| France | imperative | failed | 4/5 |
