# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.18021255731582642`
- Median control path p95 activation delta: `0.0318724343786016`
- Median ordered path lift over control p95: `5.259007093850701`
- Median combined target-logit drop: `1.9063301086425781`

## Claim Boundary

Benchmark-scoped evidence that the real gpt2 pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.163164 | 0.016642 | 0.146521 | 9.804076 |
| Germany | passed | passed | 0.228542 | 0.058329 | 0.170213 | 3.918146 |
| Italy | passed | passed | 0.072791 | 0.027875 | 0.044916 | 2.611371 |
| Spain | passed | passed | 0.144526 | 0.024808 | 0.119718 | 5.825817 |
| Portugal | passed | passed | 0.162924 | 0.018194 | 0.144730 | 8.954805 |
| Greece | passed | passed | 0.095467 | 0.035084 | 0.060383 | 2.721112 |
| Egypt | passed | passed | 0.200507 | 0.031315 | 0.169192 | 6.402820 |
| Japan | passed | passed | 0.070986 | 0.048066 | 0.022920 | 1.476851 |
| China | passed | passed | 0.188483 | 0.067225 | 0.121259 | 2.803785 |
| Russia | passed | passed | 0.637345 | 0.015205 | 0.622140 | 41.917353 |
| Canada | passed | passed | 0.190377 | 0.063476 | 0.126901 | 2.999185 |
| Australia | passed | passed | 0.194316 | 0.036984 | 0.157333 | 5.254102 |
| Sweden | passed | passed | 0.141824 | 0.032429 | 0.109395 | 4.373313 |
| Norway | passed | passed | 0.095301 | 0.054200 | 0.041101 | 1.758314 |
| Denmark | passed | passed | 0.145940 | 0.025832 | 0.120108 | 5.649572 |
| Finland | passed | passed | 0.141865 | 0.041432 | 0.100433 | 3.424084 |
| Poland | passed | passed | 0.105328 | 0.021564 | 0.083764 | 4.884509 |
| Hungary | passed | passed | 0.141796 | 0.017536 | 0.124260 | 8.086039 |
| Austria | passed | passed | 0.129983 | 0.014965 | 0.115018 | 8.685857 |
| Belgium | passed | passed | 0.021497 | 0.000000 | 0.021497 | 21496593.952179 |
| Netherlands | passed | passed | 0.212615 | 0.020180 | 0.192435 | 10.535726 |
| Ireland | passed | passed | 0.204885 | 0.038572 | 0.166313 | 5.311808 |
| Turkey | passed | passed | 0.197924 | 0.024798 | 0.173125 | 7.981347 |
| Iran | passed | passed | 0.194411 | 0.045857 | 0.148555 | 4.239537 |
| Iraq | passed | passed | 0.231618 | 0.021436 | 0.210183 | 10.805252 |
| Syria | passed | passed | 0.094395 | 0.040917 | 0.053478 | 2.306976 |
| Israel | passed | passed | 0.072171 | 0.010665 | 0.061506 | 6.767307 |
| Saudi Arabia | passed | passed | 0.228038 | 0.044967 | 0.183071 | 5.071268 |
| Pakistan | passed | passed | 0.165917 | 0.048685 | 0.117231 | 3.407936 |
| Thailand | passed | passed | 0.208013 | 0.034937 | 0.173077 | 5.954009 |
| Indonesia | passed | passed | 0.188590 | 0.020051 | 0.168539 | 9.405689 |
| Philippines | passed | passed | 0.139742 | 0.008327 | 0.131415 | 16.781087 |
| South Korea | passed | passed | 0.166535 | 0.058596 | 0.107939 | 2.842078 |
| Czech Republic | passed | passed | 0.112674 | 0.046090 | 0.066584 | 2.444653 |
| Tunisia | passed | passed | 0.065096 | 0.023571 | 0.041525 | 2.761701 |
| Chile | passed | passed | 0.157887 | 0.051364 | 0.106523 | 3.073884 |
| Peru | passed | passed | 0.171942 | 0.017716 | 0.154226 | 9.705392 |
| Cuba | passed | passed | 0.211102 | 0.062463 | 0.148639 | 3.379614 |
| Jamaica | passed | passed | 0.148421 | 0.056034 | 0.092387 | 2.648770 |
| largest planet | passed | passed | 0.335151 | 0.069744 | 0.265407 | 4.805418 |
| smallest planet | passed | passed | 0.197301 | 0.070605 | 0.126696 | 2.794431 |
| nearest star | passed | passed | 0.380190 | 0.135693 | 0.244497 | 2.801835 |
| grass color | passed | passed | 0.222204 | 0.048035 | 0.174169 | 4.625861 |
| opposite of hot | passed | passed | 0.225018 | 0.042026 | 0.182992 | 5.354256 |
| frozen water | passed | passed | 0.111161 | 0.021118 | 0.090043 | 5.263912 |
| baby dog | passed | passed | 0.411957 | 0.022554 | 0.389403 | 18.265404 |
| female parent | passed | passed | 0.215482 | 0.006020 | 0.209462 | 35.796998 |
| male parent | passed | passed | 0.222198 | 0.000000 | 0.222198 | 222197532.653809 |
| medical profession | passed | passed | 0.414556 | 0.019135 | 0.395421 | 21.665209 |
| school profession | passed | passed | 0.488796 | 0.014725 | 0.474071 | 33.195620 |
