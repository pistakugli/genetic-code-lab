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
