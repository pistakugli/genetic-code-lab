# Live GPT-2 Combined Attention+MLP Circuit Finder

This benchmark uses a real GPT-2 model, real attention-path hooks, real MLP activation hooks, automatic component discovery, and simultaneous attention+MLP causal ablation. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `48`
- Pass rate: `0.96`
- MLP component pass rate: `1.0`
- Attention component pass rate: `0.96`
- Median combined target-logit drop: `2.0570144653320312`
- Median combined control p95: `1.8040924072265625`
- Median combined lift over control p95: `1.0844234406017157`
- Mean selected MLP nodes: `1.64`
- Mean selected attention heads: `3.0`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can discover selected MLP nodes and attention heads, build one token/head/neuron/target graph, and validate the union through simultaneous causal ablation above combined controls across a factual prompt set.

This is not full GPT-2 circuit recovery, not ordered path-patching through every residual stream edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | mlp | attention | combined drop | control p95 | margin |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | passed | 1.818840 | 1.757874 | 0.060966 |
| Germany | passed | passed | passed | 1.445564 | 1.242264 | 0.203300 |
| Italy | passed | passed | passed | 3.230293 | 3.057693 | 0.172600 |
| Spain | passed | passed | passed | 3.272522 | 3.218300 | 0.054222 |
| Portugal | passed | passed | passed | 1.925232 | 1.804092 | 0.121140 |
| Greece | passed | passed | passed | 2.251404 | 2.123108 | 0.128296 |
| Egypt | passed | passed | passed | 2.357124 | 2.202637 | 0.154488 |
| Japan | passed | passed | passed | 0.959023 | 0.696495 | 0.262527 |
| China | passed | passed | passed | 2.633568 | 2.557335 | 0.076233 |
| Russia | passed | passed | passed | 2.481956 | 1.695992 | 0.785965 |
| Canada | passed | passed | passed | 1.751297 | 1.644257 | 0.107040 |
| Australia | passed | passed | passed | 1.314934 | 1.163048 | 0.151886 |
| Sweden | passed | passed | passed | 2.136024 | 1.821648 | 0.314377 |
| Norway | passed | passed | passed | 1.412010 | 1.088043 | 0.323967 |
| Denmark | passed | passed | passed | 1.741798 | 1.336784 | 0.405014 |
| Finland | passed | passed | passed | 1.640099 | 1.317924 | 0.322174 |
| Poland | passed | passed | passed | 1.727982 | 1.364212 | 0.363770 |
| Hungary | passed | passed | passed | 2.008759 | 1.888161 | 0.120598 |
| Austria | passed | passed | passed | 2.045082 | 1.765091 | 0.279991 |
| Belgium | failed | passed | failed | 1.381111 | 1.263863 | 0.117249 |
| Netherlands | passed | passed | passed | 1.718933 | 1.503975 | 0.214958 |
| Ireland | passed | passed | passed | 1.410110 | 1.256989 | 0.153122 |
| Turkey | passed | passed | passed | 3.818199 | 3.147217 | 0.670982 |
| Iran | passed | passed | passed | 6.871803 | 6.845604 | 0.026199 |
| Iraq | passed | passed | passed | 2.364708 | 2.137276 | 0.227432 |
| Syria | passed | passed | passed | 2.058083 | 1.406792 | 0.651291 |
| Israel | passed | passed | passed | 1.628532 | 1.487083 | 0.141449 |
| Saudi Arabia | passed | passed | passed | 3.160912 | 2.574028 | 0.586884 |
| Pakistan | passed | passed | passed | 2.057014 | 1.687782 | 0.369232 |
| Thailand | passed | passed | passed | 1.453979 | 1.397034 | 0.056946 |
| Indonesia | passed | passed | passed | 2.170708 | 2.085419 | 0.085289 |
| Philippines | passed | passed | passed | 4.014038 | 3.916855 | 0.097183 |
| South Korea | passed | passed | passed | 1.141998 | 0.905663 | 0.236336 |
| Czech Republic | passed | passed | passed | 2.121323 | 1.990555 | 0.130768 |
| Switzerland | passed | passed | passed | 2.803993 | 2.742180 | 0.061813 |
| Tunisia | passed | passed | passed | 2.581497 | 2.312607 | 0.268890 |
| Chile | passed | passed | passed | 3.654984 | 3.265518 | 0.389465 |
| Peru | passed | passed | passed | 2.185226 | 2.080185 | 0.105042 |
| Cuba | passed | passed | passed | 2.383675 | 2.328590 | 0.055084 |
| Jamaica | passed | passed | passed | 3.127769 | 2.976128 | 0.151642 |
| largest planet | passed | passed | passed | 2.154495 | 2.063141 | 0.091354 |
| smallest planet | passed | passed | passed | 2.336189 | 2.228287 | 0.107903 |
| nearest star | passed | passed | passed | 1.961548 | 1.906479 | 0.055069 |
| grass color | passed | passed | passed | 1.344994 | 1.140800 | 0.204193 |
| opposite of hot | passed | passed | passed | 2.113075 | 1.948570 | 0.164505 |
| number after nine | passed | passed | passed | 0.837044 | 0.651588 | 0.185455 |
| month after January | failed | passed | failed | 0.323662 | 1.094910 | -0.771248 |
| frozen water | passed | passed | passed | 2.510033 | 2.266991 | 0.243042 |
| baby dog | passed | passed | passed | 1.585350 | 1.493187 | 0.092163 |
| female parent | passed | passed | passed | 0.664459 | 0.626770 | 0.037689 |
