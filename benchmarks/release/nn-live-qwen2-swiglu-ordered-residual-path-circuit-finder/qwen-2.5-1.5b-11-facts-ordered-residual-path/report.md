# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `False`
- Valid cases: `11`
- Passed cases: `9`
- Pass rate: `0.8181818181818182`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.14971387386322021`
- Median control path p95 activation delta: `0.06075114645063869`
- Median ordered path lift over control p95: `6.8772905613795015`
- Median combined target-logit drop: `1.02490234375`

## Claim Boundary

Benchmark-scoped evidence that the real qwen2-swiglu pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 1.060050 | 0.076399 | 0.983652 | 13.875265 |
| Germany | passed | passed | 0.458282 | 0.051135 | 0.407148 | 8.962234 |
| Italy | passed | passed | 8.776144 | 0.068315 | 8.707829 | 128.466627 |
| Spain | passed | passed | 0.083183 | 0.046782 | 0.036401 | 1.778105 |
| Portugal | passed | passed | 0.149714 | 0.021769 | 0.127945 | 6.877291 |
| Greece | failed | passed | 0.000000 | 0.675862 | -0.675862 | 0.000000 |
| Japan | passed | passed | 2.730762 | 0.206233 | 2.524530 | 13.241181 |
| Russia | passed | passed | 0.145366 | 0.039855 | 0.105510 | 3.647350 |
| Norway | passed | passed | 0.114490 | 0.060751 | 0.053739 | 1.884568 |
| Austria | passed | passed | 0.775219 | 0.056773 | 0.718446 | 13.654663 |
| Netherlands | failed | passed | 0.000000 | 0.890996 | -0.890996 | 0.000000 |
