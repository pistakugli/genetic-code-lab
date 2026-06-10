# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `20`
- Passed cases: `18`
- Pass rate: `0.9`
- Combined graph pass rate: `0.9`
- Median selected path activation delta: `0.421875`
- Median control path p95 activation delta: `0.045970916748046875`
- Median ordered path lift over control p95: `9.395348837209303`
- Median combined target-logit drop: `2.0`

## Claim Boundary

Benchmark-scoped evidence that the real gpt-neox pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 1.062500 | 0.066162 | 0.996338 | 16.059041 |
| Germany | passed | passed | 0.359375 | 0.117188 | 0.242188 | 3.066667 |
| Italy | passed | passed | 0.417969 | 0.017822 | 0.400146 | 23.452055 |
| Spain | passed | passed | 0.953125 | 0.044559 | 0.908566 | 21.389949 |
| Egypt | passed | passed | 0.648438 | 0.015625 | 0.632812 | 41.500000 |
| Japan | passed | passed | 0.421875 | 0.079590 | 0.342285 | 5.300613 |
| China | passed | passed | 0.328125 | 0.000000 | 0.328125 | 328125000.000000 |
| Russia | passed | passed | 0.476562 | 0.217285 | 0.259277 | 2.193258 |
| Sweden | passed | passed | 0.394531 | 0.041992 | 0.352539 | 9.395349 |
| Norway | passed | passed | 0.507812 | 0.125000 | 0.382812 | 4.062500 |
| Hungary | passed | passed | 0.429688 | 0.082855 | 0.346832 | 5.186004 |
| Austria | passed | passed | 0.234375 | 0.159515 | 0.074860 | 1.469294 |
| Belgium | passed | passed | 1.148438 | 0.028320 | 1.120117 | 40.551724 |
| Netherlands | passed | passed | 0.076172 | 0.045971 | 0.030201 | 1.656958 |
| Ireland | failed | failed | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Iran | passed | passed | 0.625000 | 0.089844 | 0.535156 | 6.956522 |
| Iraq | passed | passed | 0.359375 | 0.000000 | 0.359375 | 359375000.000000 |
| Syria | passed | passed | 0.843750 | 0.056641 | 0.787109 | 14.896552 |
| Israel | failed | failed | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Thailand | passed | passed | 0.189453 | 0.000000 | 0.189453 | 189453125.000000 |
