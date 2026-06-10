# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.1944112479686737`
- Median control path p95 activation delta: `0.023056864738464355`
- Median ordered path lift over control p95: `6.872610381015411`
- Median combined target-logit drop: `1.9851608276367188`

## Claim Boundary

Benchmark-scoped evidence that the real gpt2 pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.163164 | 0.020911 | 0.142252 | 7.802608 |
| Germany | passed | passed | 0.228542 | 0.038098 | 0.190444 | 5.998799 |
| Italy | passed | passed | 0.072791 | 0.041088 | 0.031703 | 1.771583 |
| Spain | passed | passed | 0.443959 | 0.027669 | 0.416290 | 16.045132 |
| Portugal | passed | passed | 0.190367 | 0.023057 | 0.167310 | 8.256394 |
| Greece | passed | passed | 0.126589 | 0.047023 | 0.079567 | 2.692089 |
| Egypt | passed | passed | 0.200507 | 0.035177 | 0.165330 | 5.699918 |
| Japan | passed | passed | 0.105458 | 0.034708 | 0.070751 | 3.038475 |
| China | passed | passed | 0.188483 | 0.069547 | 0.118936 | 2.710141 |
| Russia | passed | passed | 0.637345 | 0.024587 | 0.612757 | 25.921533 |
| Canada | passed | passed | 0.167997 | 0.028324 | 0.139672 | 5.931210 |
| Australia | passed | passed | 0.194316 | 0.038166 | 0.156151 | 5.091412 |
| Sweden | passed | passed | 0.065967 | 0.017495 | 0.048472 | 3.770661 |
| Norway | passed | passed | 0.211211 | 0.008299 | 0.202912 | 25.451219 |
| Denmark | passed | passed | 0.174739 | 0.007005 | 0.167734 | 24.943424 |
| Finland | passed | passed | 0.250630 | 0.056270 | 0.194360 | 4.454030 |
| Poland | passed | passed | 0.140418 | 0.002005 | 0.138413 | 70.024775 |
| Hungary | passed | passed | 0.132298 | 0.001797 | 0.130501 | 73.620954 |
| Austria | passed | passed | 0.129983 | 0.018913 | 0.111070 | 6.872610 |
| Belgium | passed | passed | 0.190701 | 0.024104 | 0.166597 | 7.911634 |
| Netherlands | passed | passed | 0.212615 | 0.012355 | 0.200260 | 17.208315 |
| Ireland | passed | passed | 0.044628 | 0.011131 | 0.033497 | 4.009486 |
| Turkey | passed | passed | 0.197924 | 0.019740 | 0.178184 | 10.026447 |
| Iran | passed | passed | 0.194411 | 0.032178 | 0.162233 | 6.041791 |
| Iraq | passed | passed | 0.286964 | 0.005557 | 0.281406 | 51.635443 |
| Syria | passed | passed | 0.245497 | 0.009175 | 0.236322 | 26.757866 |
| Israel | passed | passed | 0.357032 | 0.035871 | 0.321161 | 9.953179 |
| Saudi Arabia | passed | passed | 0.228038 | 0.038278 | 0.189760 | 5.957471 |
| Pakistan | passed | passed | 0.331118 | 0.021525 | 0.309593 | 15.383128 |
| Thailand | passed | passed | 0.078623 | 0.032340 | 0.046282 | 2.431101 |
| Indonesia | passed | passed | 0.239742 | 0.012790 | 0.226951 | 18.744302 |
| Philippines | passed | passed | 0.210300 | 0.037195 | 0.173105 | 5.654045 |
| South Korea | passed | passed | 0.067461 | 0.051618 | 0.015843 | 1.306933 |
| Czech Republic | passed | passed | 0.126951 | 0.026269 | 0.100683 | 4.832798 |
| Tunisia | passed | passed | 0.435673 | 0.013854 | 0.421819 | 31.448439 |
| Chile | passed | passed | 0.193968 | 0.002326 | 0.191643 | 83.401689 |
| Peru | passed | passed | 0.171942 | 0.025193 | 0.146749 | 6.825063 |
| Cuba | passed | passed | 0.211102 | 0.041408 | 0.169694 | 5.098073 |
| Jamaica | passed | passed | 0.148421 | 0.036107 | 0.112314 | 4.110618 |
| largest planet | passed | passed | 0.273457 | 0.024183 | 0.249274 | 11.307935 |
| smallest planet | passed | passed | 0.346902 | 0.020878 | 0.326024 | 16.615778 |
| nearest star | passed | passed | 0.368653 | 0.097619 | 0.271033 | 3.776436 |
| grass color | passed | passed | 0.041309 | 0.027599 | 0.013711 | 1.496791 |
| opposite of hot | passed | passed | 0.239127 | 0.000000 | 0.239127 | 239127159.118652 |
| frozen water | passed | passed | 0.111161 | 0.020563 | 0.090598 | 5.405904 |
| baby dog | passed | passed | 0.411957 | 0.011079 | 0.400877 | 37.181966 |
| female parent | passed | passed | 0.215482 | 0.007282 | 0.208200 | 29.592666 |
| male parent | passed | passed | 0.250469 | 0.000000 | 0.250469 | 250468730.926514 |
| medical profession | passed | passed | 0.021875 | 0.004505 | 0.017371 | 4.856311 |
| school profession | passed | passed | 0.488796 | 0.012399 | 0.476397 | 39.423064 |
