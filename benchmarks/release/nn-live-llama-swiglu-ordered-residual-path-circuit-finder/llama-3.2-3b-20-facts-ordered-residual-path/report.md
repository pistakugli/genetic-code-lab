# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `20`
- Passed cases: `20`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.046875`
- Median control path p95 activation delta: `0.0017847249284386607`
- Median ordered path lift over control p95: `39.35359036498139`
- Median combined target-logit drop: `0.9375`

## Claim Boundary

Benchmark-scoped evidence that the real llama-swiglu pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.078125 | 0.003696 | 0.074429 | 21.138310 |
| Germany | passed | passed | 0.046875 | 0.000000 | 0.046875 | 46875000.000000 |
| Italy | passed | passed | 0.117188 | 0.007422 | 0.109766 | 15.789474 |
| Spain | passed | passed | 0.035156 | 0.001416 | 0.033740 | 24.827586 |
| Portugal | passed | passed | 0.242188 | 0.000000 | 0.242188 | 242187500.000000 |
| Greece | passed | passed | 0.093750 | 0.001801 | 0.091949 | 52.049339 |
| Egypt | passed | passed | 0.031250 | 0.004321 | 0.026929 | 7.231638 |
| Japan | passed | passed | 0.031250 | 0.008105 | 0.023145 | 3.855422 |
| China | passed | passed | 0.039062 | 0.000000 | 0.039062 | 39062500.000000 |
| Russia | passed | passed | 0.046875 | 0.001758 | 0.045117 | 26.657842 |
| Canada | passed | passed | 0.019531 | 0.002295 | 0.017236 | 8.510638 |
| Australia | passed | passed | 0.156250 | 0.001953 | 0.154297 | 80.000000 |
| Sweden | passed | passed | 0.046875 | 0.000000 | 0.046875 | 46875000.000000 |
| Norway | passed | passed | 0.115234 | 0.001768 | 0.113466 | 65.167706 |
| Denmark | passed | passed | 0.023438 | 0.002649 | 0.020789 | 8.849335 |
| Finland | passed | passed | 0.054688 | 0.000000 | 0.054688 | 54687500.000000 |
| Poland | passed | passed | 0.023438 | 0.002602 | 0.020836 | 9.008465 |
| Hungary | passed | passed | 0.046875 | 0.000000 | 0.046875 | 46875000.000000 |
| Austria | passed | passed | 0.015625 | 0.000000 | 0.015625 | 15625000.000000 |
| Belgium | passed | passed | 0.101562 | 0.004053 | 0.097510 | 25.060241 |
