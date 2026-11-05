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
| France | 1.818840 | 0.833333 | 1.464195 | passed |
| Germany | 1.388084 | 0.500000 | 1.247429 | failed |
| Italy | 3.218582 | 0.833333 | 3.085922 | passed |
| Portugal | 1.984413 | 0.000000 | 1.848541 | failed |
| Greece | 2.556099 | 0.000000 | 2.378494 | failed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 5 | 2 | 0.400000 | 0.258486 | 20.00 |
| embedded-question | 5 | 2 | 0.400000 | 0.100210 | 19.00 |
| imperative | 5 | 0 | 0.000000 | -0.372764 | 10.00 |
| possessive-question | 5 | 3 | 0.600000 | 0.065180 | 7.00 |
| sentence-completion | 5 | 3 | 0.600000 | 0.166774 | 17.00 |
| service-question | 5 | 3 | 0.600000 | 0.046023 | 9.00 |
