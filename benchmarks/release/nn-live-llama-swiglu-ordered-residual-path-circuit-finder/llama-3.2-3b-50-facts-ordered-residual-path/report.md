# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.07421875`
- Median control path p95 activation delta: `0.001815529260784385`
- Median ordered path lift over control p95: `50.018348623853214`
- Median combined target-logit drop: `1.0`

## Claim Boundary

Benchmark-scoped evidence that the real llama-swiglu pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.085938 | 0.002832 | 0.083105 | 30.344828 |
| Germany | passed | passed | 0.039062 | 0.000000 | 0.039062 | 39062500.000000 |
| Italy | passed | passed | 0.109375 | 0.006738 | 0.102637 | 16.231884 |
| Spain | passed | passed | 0.007812 | 0.005664 | 0.002148 | 1.379310 |
| Portugal | passed | passed | 0.242188 | 0.000000 | 0.242188 | 242187500.000000 |
| Greece | passed | passed | 0.003906 | 0.000887 | 0.003019 | 4.404604 |
| Egypt | passed | passed | 0.359375 | 0.002563 | 0.356812 | 140.190476 |
| Japan | passed | passed | 0.152344 | 0.000953 | 0.151390 | 159.779139 |
| China | passed | passed | 0.039062 | 0.000000 | 0.039062 | 39062500.000000 |
| Russia | passed | passed | 0.132812 | 0.000901 | 0.131911 | 147.362490 |
| Canada | passed | passed | 0.015625 | 0.001953 | 0.013672 | 8.000000 |
| Australia | passed | passed | 0.148438 | 0.000014 | 0.148424 | 10632.374853 |
| Sweden | passed | passed | 0.203125 | 0.005469 | 0.197656 | 37.142857 |
| Norway | passed | passed | 0.113281 | 0.001828 | 0.111453 | 61.957898 |
| Denmark | passed | passed | 0.023438 | 0.001758 | 0.021680 | 13.332370 |
| Finland | passed | passed | 0.039062 | 0.000048 | 0.039015 | 818.737062 |
| Poland | passed | passed | 0.023438 | 0.003711 | 0.019727 | 6.315789 |
| Hungary | passed | passed | 0.289062 | 0.000000 | 0.289062 | 289062500.000000 |
| Austria | passed | passed | 0.007812 | 0.000000 | 0.007812 | 7812500.000000 |
| Belgium | passed | passed | 0.097656 | 0.007034 | 0.090622 | 13.883403 |
| Netherlands | passed | passed | 0.054688 | 0.009985 | 0.044702 | 5.476773 |
| Ireland | passed | passed | 0.097656 | 0.007067 | 0.090589 | 13.818213 |
| Turkey | passed | passed | 0.015625 | 0.000000 | 0.015625 | 15625000.000000 |
| Iran | passed | passed | 0.046875 | 0.000000 | 0.046875 | 46875000.000000 |
| Iraq | passed | passed | 0.062500 | 0.000056 | 0.062444 | 1117.264180 |
| Syria | passed | passed | 0.171875 | 0.096094 | 0.075781 | 1.788618 |
| Israel | passed | passed | 0.062500 | 0.001803 | 0.060697 | 34.670217 |
| Saudi Arabia | passed | passed | 0.437500 | 0.007812 | 0.429688 | 56.000000 |
| Pakistan | passed | passed | 0.015625 | 0.000014 | 0.015611 | 1122.432027 |
| Thailand | passed | passed | 0.078125 | 0.002410 | 0.075715 | 32.418268 |
| Indonesia | passed | passed | 0.046875 | 0.000000 | 0.046875 | 46875000.000000 |
| Philippines | passed | passed | 0.015625 | 0.000000 | 0.015625 | 15625000.000000 |
| South Korea | passed | passed | 0.046875 | 0.001759 | 0.045116 | 26.646013 |
| Bulgaria | passed | passed | 0.140625 | 0.026953 | 0.113672 | 5.217391 |
| Czech Republic | passed | passed | 0.070312 | 0.055078 | 0.015234 | 1.276596 |
| Switzerland | passed | passed | 0.078125 | 0.000000 | 0.078125 | 78125000.000000 |
| Tunisia | passed | passed | 0.078125 | 0.033203 | 0.044922 | 2.352941 |
| Kenya | passed | passed | 0.031250 | 0.000891 | 0.030359 | 35.058282 |
| Chile | passed | passed | 0.062500 | 0.006348 | 0.056152 | 9.846154 |
| Peru | passed | passed | 0.031250 | 0.011133 | 0.020117 | 2.807018 |
| Cuba | passed | passed | 0.187500 | 0.062500 | 0.125000 | 3.000000 |
| Jamaica | passed | passed | 0.031250 | 0.000000 | 0.031250 | 31250000.000000 |
| largest planet | passed | passed | 0.210938 | 0.001761 | 0.209177 | 119.803940 |
| smallest planet | passed | passed | 1.050781 | 0.002118 | 1.048664 | 496.193837 |
| nearest star | passed | passed | 0.023438 | 0.006079 | 0.017358 | 3.855422 |
| grass color | passed | passed | 0.246094 | 0.000000 | 0.246094 | 246093750.000000 |
| opposite of hot | passed | passed | 1.281250 | 0.002026 | 1.279224 | 632.289157 |
| number after nine | passed | passed | 0.234375 | 0.005322 | 0.229053 | 44.036697 |
| month after January | passed | passed | 0.125000 | 0.026465 | 0.098535 | 4.723247 |
| frozen water | passed | passed | 0.027344 | 0.000000 | 0.027344 | 27343750.000000 |
