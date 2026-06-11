# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `20`
- Passed cases: `20`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.046875`
- Median control path p95 activation delta: `0.0014201669022440894`
- Median ordered path lift over control p95: `55.04681300180998`
- Median combined target-logit drop: `0.9375`

## Claim Boundary

Benchmark-scoped evidence that the real llama-swiglu pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.078125 | 0.002832 | 0.075293 | 27.586207 |
| Germany | passed | passed | 0.046875 | 0.003516 | 0.043359 | 13.333147 |
| Italy | passed | passed | 0.117188 | 0.000000 | 0.117188 | 117187500.000000 |
| Spain | passed | passed | 0.035156 | 0.004053 | 0.031104 | 8.674699 |
| Portugal | passed | passed | 0.242188 | 0.000000 | 0.242188 | 242187500.000000 |
| Greece | passed | passed | 0.093750 | 0.010811 | 0.082939 | 8.671884 |
| Egypt | passed | passed | 0.031250 | 0.002612 | 0.028638 | 11.962617 |
| Japan | passed | passed | 0.031250 | 0.000000 | 0.031250 | 31250000.000000 |
| China | passed | passed | 0.039062 | 0.000000 | 0.039062 | 39062500.000000 |
| Russia | passed | passed | 0.046875 | 0.001795 | 0.045080 | 26.115217 |
| Canada | passed | passed | 0.019531 | 0.000000 | 0.019531 | 19531250.000000 |
| Australia | passed | passed | 0.156250 | 0.012598 | 0.143652 | 12.403101 |
| Sweden | passed | passed | 0.046875 | 0.000000 | 0.046875 | 46875000.000000 |
| Norway | passed | passed | 0.115234 | 0.000000 | 0.115234 | 115234375.000000 |
| Denmark | passed | passed | 0.023438 | 0.000000 | 0.023438 | 23437500.000000 |
| Finland | passed | passed | 0.054688 | 0.000000 | 0.054688 | 54687500.000000 |
| Poland | passed | passed | 0.023438 | 0.002412 | 0.021026 | 9.717293 |
| Hungary | passed | passed | 0.046875 | 0.012207 | 0.034668 | 3.840000 |
| Austria | passed | passed | 0.015625 | 0.001609 | 0.014016 | 9.708684 |
| Belgium | passed | passed | 0.101562 | 0.001231 | 0.100332 | 82.507419 |
