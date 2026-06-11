# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `20`
- Passed cases: `18`
- Pass rate: `0.9`
- Combined graph pass rate: `0.9`
- Median selected path activation delta: `0.51171875`
- Median control path p95 activation delta: `0.09458007812499991`
- Median ordered path lift over control p95: `6.051611430866259`
- Median combined target-logit drop: `2.0`

## Claim Boundary

Benchmark-scoped evidence that the real gpt-neox pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.371704 | 0.071971 | 0.299733 | 5.164628 |
| Germany | passed | passed | 0.543457 | 0.185797 | 0.357660 | 2.925002 |
| Italy | passed | passed | 0.357178 | 0.052869 | 0.304309 | 6.755946 |
| Spain | passed | passed | 1.122559 | 0.838814 | 0.283744 | 1.338268 |
| Egypt | passed | passed | 0.648438 | 0.090771 | 0.557666 | 7.143626 |
| Japan | failed | failed | 1.527344 | 1.242603 | 0.284741 | 1.229149 |
| China | passed | passed | 0.303955 | 0.000000 | 0.303955 | 303955078.125000 |
| Russia | passed | passed | 1.968750 | 1.876440 | 0.092310 | 1.049194 |
| Sweden | passed | passed | 0.408447 | 0.055114 | 0.353333 | 7.410955 |
| Norway | passed | passed | 0.377808 | 0.125110 | 0.252698 | 3.019807 |
| Hungary | passed | passed | 0.322266 | 0.105995 | 0.216270 | 3.040380 |
| Austria | passed | passed | 0.392914 | 0.178920 | 0.213994 | 2.196031 |
| Belgium | passed | passed | 0.076782 | 0.054880 | 0.021902 | 1.399099 |
| Netherlands | passed | passed | 0.867188 | 0.162174 | 0.705014 | 5.347277 |
| Ireland | passed | passed | 0.392090 | 0.000000 | 0.392090 | 392089843.750000 |
| Iran | passed | passed | 1.531006 | 0.109290 | 1.421716 | 14.008712 |
| Iraq | passed | passed | 1.505859 | 0.000000 | 1.505859 | 1505859375.000000 |
| Syria | passed | passed | 0.843750 | 0.098389 | 0.745361 | 8.575682 |
| Israel | failed | failed | 1.822266 | 0.000000 | 1.822266 | 1822265625.000000 |
| Thailand | passed | passed | 0.479980 | 0.000000 | 0.479980 | 479980468.750000 |
