# Live Ordered Path OOD Stress

This gate replays already-discovered real-model circuits. It does not fit a surrogate.

- Pass: `True`
- Cases: `4`
- Original replay pass rate: `1.0`
- Paraphrase profile: `ood`
- Paraphrase pass rate: `1.0`
- Neighbor-layer specificity rate: `1.0`
- Maximum replay error: `0.0`

| subject | replay | paraphrase | layer margin | status |
|---|---:|---:|---:|---:|
| France | 1.330673 | 1.000000 | 1.314781 | passed |
| Germany | 1.012051 | 1.000000 | 0.994217 | passed |
| Spain | 1.617285 | 1.000000 | 1.617285 | passed |
| Japan | 0.923326 | 1.000000 | 0.923326 | passed |

## Paraphrase family results

| family | count | passed | pass rate | mean drop ratio | mean tokens |
|---|---:|---:|---:|---:|---:|
| context-prefixed-question | 4 | 4 | 1.000000 | 1.178771 | 20.00 |
| embedded-question | 4 | 4 | 1.000000 | 0.384180 | 19.00 |
| imperative | 4 | 4 | 1.000000 | 0.554377 | 10.00 |
| possessive-question | 4 | 4 | 1.000000 | 0.725027 | 7.00 |
| sentence-completion | 4 | 4 | 1.000000 | 1.325423 | 17.00 |
| service-question | 4 | 4 | 1.000000 | 0.989460 | 9.00 |
