# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `20`
- Passed cases: `20`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.234375`
- Median control path p95 activation delta: `0.03992919921874999`
- Median ordered path lift over control p95: `4.966984592809982`
- Median combined target-logit drop: `0.3125`

## Claim Boundary

Benchmark-scoped evidence that the real mistral-swiglu pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 4.000000 | 0.829687 | 3.170313 | 4.821092 |
| Germany | passed | passed | 0.132812 | 0.069531 | 0.063281 | 1.910112 |
| Italy | passed | passed | 0.390625 | 0.242187 | 0.148438 | 1.612903 |
| Spain | passed | passed | 0.125000 | 0.000586 | 0.124414 | 213.256304 |
| Japan | passed | passed | 0.265625 | 0.004053 | 0.261572 | 65.542169 |
| China | passed | passed | 0.078125 | 0.041577 | 0.036548 | 1.879037 |
| Russia | passed | passed | 0.078125 | 0.007749 | 0.070376 | 10.082213 |
| Sweden | passed | passed | 0.375000 | 0.242187 | 0.132813 | 1.548387 |
| Hungary | passed | passed | 0.437500 | 0.031250 | 0.406250 | 14.000000 |
| Austria | passed | passed | 0.187500 | 0.036719 | 0.150781 | 5.106383 |
| Netherlands | passed | passed | 0.296875 | 0.267969 | 0.028906 | 1.107872 |
| Ireland | passed | passed | 0.093750 | 0.029687 | 0.064063 | 3.157895 |
| Israel | passed | passed | 0.312500 | 0.260449 | 0.052051 | 1.199850 |
| Switzerland | passed | passed | 0.218750 | 0.045312 | 0.173438 | 4.827586 |
| Chile | passed | passed | 0.125000 | 0.015625 | 0.109375 | 8.000000 |
| nearest star | passed | passed | 0.117188 | 0.104687 | 0.012500 | 1.119403 |
| grass color | passed | passed | 0.093750 | 0.000004 | 0.093746 | 21908.090885 |
| opposite of hot | passed | passed | 0.703125 | 0.060352 | 0.642773 | 11.650485 |
| number after nine | passed | passed | 0.906250 | 0.038281 | 0.867969 | 23.673469 |
| month after January | passed | passed | 0.250000 | 0.020172 | 0.229828 | 12.393226 |
