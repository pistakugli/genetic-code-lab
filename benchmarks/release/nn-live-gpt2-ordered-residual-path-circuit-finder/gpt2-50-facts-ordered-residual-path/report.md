# Live LLM Ordered Residual Path Circuit Finder

This benchmark uses a real causal-LM model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.19325748085975647`
- Median control path p95 activation delta: `0.03085190749261527`
- Median ordered path lift over control p95: `5.357243298305573`
- Median combined target-logit drop: `1.9847869873046875`

## Claim Boundary

Benchmark-scoped evidence that the real gpt2 pipeline can discover selected attention heads and MLP units for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full-model circuit recovery, not exhaustive residual-stream patching for every edge, not full FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.163164 | 0.024565 | 0.138598 | 6.642033 |
| Germany | passed | passed | 0.228542 | 0.042318 | 0.186224 | 5.400621 |
| Italy | passed | passed | 0.072791 | 0.047551 | 0.025239 | 1.530779 |
| Spain | passed | passed | 0.443959 | 0.042211 | 0.401749 | 10.517716 |
| Portugal | passed | passed | 0.162924 | 0.027424 | 0.135500 | 5.940989 |
| Greece | passed | passed | 0.126589 | 0.050127 | 0.076462 | 2.525358 |
| Egypt | passed | passed | 0.200507 | 0.042376 | 0.158131 | 4.731574 |
| Japan | passed | passed | 0.105458 | 0.044501 | 0.060958 | 2.369815 |
| China | passed | passed | 0.188483 | 0.069547 | 0.118936 | 2.710141 |
| Russia | passed | passed | 0.637345 | 0.037275 | 0.600070 | 17.098567 |
| Canada | passed | passed | 0.167997 | 0.041900 | 0.126097 | 4.009498 |
| Australia | passed | passed | 0.194316 | 0.044361 | 0.149955 | 4.380328 |
| Sweden | passed | passed | 0.065967 | 0.030815 | 0.035152 | 2.140725 |
| Norway | passed | passed | 0.192199 | 0.027171 | 0.165028 | 7.073656 |
| Denmark | passed | passed | 0.145940 | 0.025135 | 0.120805 | 5.806276 |
| Finland | passed | passed | 0.250630 | 0.056442 | 0.194188 | 4.440487 |
| Poland | passed | passed | 0.140418 | 0.005451 | 0.134968 | 25.762306 |
| Hungary | passed | passed | 0.132298 | 0.014211 | 0.118087 | 9.309487 |
| Austria | passed | passed | 0.129983 | 0.022505 | 0.107478 | 5.775725 |
| Belgium | passed | passed | 0.190701 | 0.028634 | 0.162067 | 6.659902 |
| Netherlands | passed | passed | 0.212615 | 0.019993 | 0.192622 | 10.634304 |
| Ireland | passed | passed | 0.044628 | 0.030685 | 0.013943 | 1.454373 |
| Turkey | passed | passed | 0.197924 | 0.020475 | 0.177448 | 9.666437 |
| Iran | passed | passed | 0.194411 | 0.037001 | 0.157411 | 5.254269 |
| Iraq | passed | passed | 0.231618 | 0.016492 | 0.215126 | 14.044108 |
| Syria | passed | passed | 0.245497 | 0.046199 | 0.199298 | 5.313866 |
| Israel | passed | passed | 0.357032 | 0.037624 | 0.319407 | 9.489355 |
| Saudi Arabia | passed | passed | 0.228038 | 0.052570 | 0.175468 | 4.337817 |
| Pakistan | passed | passed | 0.101133 | 0.045441 | 0.055692 | 2.225571 |
| Thailand | passed | passed | 0.078623 | 0.075937 | 0.002685 | 1.035363 |
| Indonesia | passed | passed | 0.239742 | 0.045832 | 0.193909 | 5.230854 |
| Philippines | passed | passed | 0.190537 | 0.028378 | 0.162159 | 6.714170 |
| South Korea | passed | passed | 0.067461 | 0.024804 | 0.042657 | 2.719732 |
| Czech Republic | passed | passed | 0.112674 | 0.027196 | 0.085478 | 4.143045 |
| Tunisia | passed | passed | 0.290091 | 0.019932 | 0.270158 | 14.553800 |
| Chile | passed | passed | 0.157887 | 0.028355 | 0.129532 | 5.568295 |
| Peru | passed | passed | 0.171942 | 0.017721 | 0.154221 | 9.702914 |
| Cuba | passed | passed | 0.211102 | 0.048364 | 0.162738 | 4.364849 |
| Jamaica | passed | passed | 0.148421 | 0.059555 | 0.088865 | 2.492146 |
| largest planet | passed | passed | 0.335151 | 0.074764 | 0.260387 | 4.482774 |
| smallest planet | passed | passed | 0.416453 | 0.120764 | 0.295689 | 3.448484 |
| nearest star | passed | passed | 0.214884 | 0.115262 | 0.099622 | 1.864308 |
| grass color | passed | passed | 0.222204 | 0.024579 | 0.197626 | 9.040557 |
| opposite of hot | passed | passed | 0.239127 | 0.000000 | 0.239127 | 239127159.118652 |
| frozen water | passed | passed | 0.111161 | 0.030889 | 0.080272 | 3.598762 |
| baby dog | passed | passed | 0.411957 | 0.033438 | 0.378519 | 12.320193 |
| female parent | passed | passed | 0.215482 | 0.017495 | 0.197987 | 12.316514 |
| male parent | passed | passed | 0.222198 | 0.011724 | 0.210474 | 18.952346 |
| medical profession | passed | passed | 0.021875 | 0.018661 | 0.003214 | 1.172243 |
| school profession | passed | passed | 0.488796 | 0.016667 | 0.472128 | 29.326583 |
