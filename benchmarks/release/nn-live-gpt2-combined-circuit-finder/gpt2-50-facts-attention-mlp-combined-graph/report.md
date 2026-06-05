# Live GPT-2 Combined Attention+MLP Circuit Finder

This benchmark uses a real GPT-2 model, real attention-path hooks, real MLP activation hooks, automatic component discovery, and simultaneous attention+MLP causal ablation. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- MLP component pass rate: `1.0`
- Attention component pass rate: `1.0`
- Median combined target-logit drop: `1.92523193359375`
- Median combined control p95: `0.417144775390625`
- Median combined lift over control p95: `4.333544871993313`
- Mean selected MLP nodes: `1.56`
- Mean selected attention heads: `1.02`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can discover selected MLP nodes and attention heads, build one token/head/neuron/target graph, and validate the union through simultaneous causal ablation above combined controls across a factual prompt set.

This is not full GPT-2 circuit recovery, not ordered path-patching through every residual stream edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | mlp | attention | combined drop | control p95 | margin |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | passed | 1.818840 | 0.743576 | 1.075264 |
| Germany | passed | passed | passed | 1.445564 | 0.623680 | 0.821884 |
| Italy | passed | passed | passed | 3.230293 | 0.909477 | 2.320816 |
| Spain | passed | passed | passed | 3.154991 | 0.417145 | 2.737846 |
| Portugal | passed | passed | passed | 1.925232 | 0.601982 | 1.323250 |
| Greece | passed | passed | passed | 2.251404 | 1.046951 | 1.204453 |
| Egypt | passed | passed | passed | 2.357124 | 0.545853 | 1.811272 |
| Japan | passed | passed | passed | 0.959023 | 0.419083 | 0.539940 |
| China | passed | passed | passed | 2.633568 | 0.592896 | 2.040672 |
| Russia | passed | passed | passed | 2.481956 | 0.695061 | 1.786896 |
| Canada | passed | passed | passed | 0.504021 | 0.439362 | 0.064659 |
| Australia | passed | passed | passed | 1.168922 | 0.428452 | 0.740471 |
| Sweden | passed | passed | passed | 2.136024 | 0.492905 | 1.643120 |
| Norway | passed | passed | passed | 1.412010 | 0.285973 | 1.126038 |
| Denmark | passed | passed | passed | 1.741798 | 0.350983 | 1.390816 |
| Finland | passed | passed | passed | 1.157379 | 0.296951 | 0.860428 |
| Poland | passed | passed | passed | 1.727982 | 0.419594 | 1.308388 |
| Hungary | passed | passed | passed | 2.008759 | 0.876656 | 1.132103 |
| Austria | passed | passed | passed | 2.045082 | 0.593170 | 1.451912 |
| Belgium | passed | passed | passed | 1.381111 | 0.551788 | 0.829323 |
| Netherlands | passed | passed | passed | 1.718933 | 0.596542 | 1.122391 |
| Ireland | passed | passed | passed | 1.410110 | 0.357277 | 1.052834 |
| Turkey | passed | passed | passed | 3.818199 | 0.593842 | 3.224358 |
| Iran | passed | passed | passed | 1.582626 | 0.294586 | 1.288040 |
| Iraq | passed | passed | passed | 2.258224 | 0.055214 | 2.203011 |
| Syria | passed | passed | passed | 1.508110 | 0.473900 | 1.034210 |
| Israel | passed | passed | passed | 1.690529 | 0.407814 | 1.282715 |
| Saudi Arabia | passed | passed | passed | 2.714005 | 0.468628 | 2.245377 |
| Pakistan | passed | passed | passed | 2.057014 | 0.402145 | 1.654869 |
| Thailand | passed | passed | passed | 1.453979 | 0.274185 | 1.179794 |
| Indonesia | passed | passed | passed | 2.316467 | 0.319984 | 1.996483 |
| Philippines | passed | passed | passed | 1.487564 | 0.376579 | 1.110985 |
| South Korea | passed | passed | passed | 0.847572 | 0.286041 | 0.561531 |
| Czech Republic | passed | passed | passed | 2.075897 | 0.550049 | 1.525848 |
| Tunisia | passed | passed | passed | 2.612411 | 0.928879 | 1.683533 |
| Chile | passed | passed | passed | 3.306816 | 0.755188 | 2.551628 |
| Peru | passed | passed | passed | 2.246483 | 0.418495 | 1.827988 |
| Cuba | passed | passed | passed | 2.275627 | 0.162842 | 2.112785 |
| Jamaica | passed | passed | passed | 3.127769 | 0.044998 | 3.082771 |
| largest planet | passed | passed | passed | 2.397034 | 0.026596 | 2.370438 |
| smallest planet | passed | passed | passed | 2.336189 | 0.129532 | 2.206657 |
| nearest star | passed | passed | passed | 2.239586 | 0.000000 | 2.239586 |
| grass color | passed | passed | passed | 1.146965 | 0.544144 | 0.602821 |
| opposite of hot | passed | passed | passed | 2.024422 | 0.054924 | 1.969498 |
| frozen water | passed | passed | passed | 2.510033 | 0.141998 | 2.368034 |
| baby dog | passed | passed | passed | 1.689445 | 0.125359 | 1.564087 |
| female parent | passed | passed | passed | 0.664459 | 0.000000 | 0.664459 |
| male parent | passed | passed | passed | 0.581787 | 0.000801 | 0.580986 |
| medical profession | passed | passed | passed | 1.258743 | 0.235672 | 1.023071 |
| school profession | passed | passed | passed | 1.406326 | 0.256981 | 1.149345 |
