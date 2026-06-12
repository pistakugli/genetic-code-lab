# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `10`
- Passed cases: `10`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.203125`
- Median control path p95 activation delta: `0.04531249999999998`
- Median ordered path lift over control p95: `5.104890772220551`
- Median combined target-logit drop: `0.34375`

## Claim Boundary

Benchmark-scoped evidence that the real mistral-swiglu pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 5.750000 | 1.185156 | 4.564844 | 4.851681 |
| Germany | passed | passed | 0.125000 | 0.046875 | 0.078125 | 2.666667 |
| Italy | passed | passed | 0.218750 | 0.053906 | 0.164844 | 4.057971 |
| Spain | passed | passed | 0.468750 | 0.000000 | 0.468750 | 468750000.000000 |
| Japan | passed | passed | 0.093750 | 0.006201 | 0.087549 | 15.118110 |
| China | passed | passed | 0.375000 | 0.236816 | 0.138184 | 1.583505 |
| Russia | passed | passed | 0.187500 | 0.030762 | 0.156738 | 6.095238 |
| Sweden | passed | passed | 0.312500 | 0.045312 | 0.267188 | 6.896552 |
| Hungary | passed | passed | 0.078125 | 0.014581 | 0.063544 | 5.358101 |
| Austria | passed | passed | 0.187500 | 0.045312 | 0.142188 | 4.137931 |
