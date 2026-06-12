# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `20`
- Passed cases: `20`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.51171875`
- Median control path p95 activation delta: `0.06836662292480464`
- Median ordered path lift over control p95: `10.015731969055064`
- Median combined target-logit drop: `2.0`

## Claim Boundary

Benchmark-scoped evidence that the real gpt-neox pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.371704 | 0.046970 | 0.324734 | 7.913716 |
| Germany | passed | passed | 0.543457 | 0.136890 | 0.406567 | 3.970037 |
| Italy | passed | passed | 0.357178 | 0.008020 | 0.349158 | 44.535769 |
| Spain | passed | passed | 1.122559 | 0.893530 | 0.229028 | 1.256318 |
| Egypt | passed | passed | 0.648438 | 0.044885 | 0.603552 | 14.446560 |
| Japan | passed | passed | 0.882324 | 0.688794 | 0.193530 | 1.280970 |
| China | passed | passed | 0.047729 | 0.000000 | 0.047729 | 47729492.187500 |
| Russia | passed | passed | 1.968750 | 1.818500 | 0.150250 | 1.082623 |
| Sweden | passed | passed | 0.408447 | 0.079563 | 0.328884 | 5.133624 |
| Norway | passed | passed | 0.377808 | 0.078381 | 0.299426 | 4.820121 |
| Hungary | passed | passed | 0.322266 | 0.092152 | 0.230113 | 3.497094 |
| Austria | passed | passed | 0.392914 | 0.000110 | 0.392804 | 3576.388889 |
| Belgium | passed | passed | 0.613281 | 0.154825 | 0.458456 | 3.961130 |
| Netherlands | passed | passed | 0.867188 | 0.058352 | 0.808836 | 14.861342 |
| Ireland | passed | passed | 0.392090 | 0.000000 | 0.392090 | 392089843.750000 |
| Iran | passed | passed | 1.531006 | 0.108752 | 1.422253 | 14.077899 |
| Iraq | passed | passed | 1.505859 | 0.000000 | 1.505859 | 1505859375.000000 |
| Syria | passed | passed | 0.843750 | 0.089636 | 0.754114 | 9.413046 |
| Israel | passed | passed | 0.210938 | 0.000000 | 0.210938 | 210937500.000000 |
| Thailand | passed | passed | 0.479980 | 0.045203 | 0.434778 | 10.618417 |
