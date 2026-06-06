# Live GPT-2 Ordered Residual Path Circuit Finder

This benchmark uses a real GPT-2 model, real attention-path hooks, real MLP activation hooks, and ordered attention->residual->MLP->target path evidence. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Combined graph pass rate: `1.0`
- Median selected path activation delta: `0.16316372156143188`
- Median control path p95 activation delta: `0.02243638038635254`
- Median ordered path lift over control p95: `6.071965452028251`
- Median combined target-logit drop: `1.9691314697265625`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can discover selected attention heads and MLP nodes for factual prompts, then validate an ordered attention->residual->MLP->target path by showing selected attention interventions change selected MLP activations above full-control path interventions while the combined circuit remains causally effective on the target logit.

This is not full GPT-2 circuit recovery, not exhaustive residual-stream patching for every edge, not full GPT-2 FFN extraction, not arbitrary LLM understanding, not output-only original-neuron identity, and not a broad SOTA claim.

## Cases

| subject | status | combined | path delta | control p95 | margin | lift |
|---|---:|---:|---:|---:|---:|---:|
| France | passed | passed | 0.163164 | 0.020911 | 0.142252 | 7.802608 |
| Germany | passed | passed | 0.112686 | 0.038098 | 0.074588 | 2.957802 |
| Italy | passed | passed | 0.072791 | 0.041088 | 0.031703 | 1.771583 |
| Spain | passed | passed | 0.443959 | 0.027669 | 0.416290 | 16.045132 |
| Portugal | passed | passed | 0.138547 | 0.023057 | 0.115490 | 6.008939 |
| Greece | passed | passed | 0.126589 | 0.053922 | 0.072668 | 2.347649 |
| Egypt | passed | passed | 0.148936 | 0.000000 | 0.148936 | 148936390.876770 |
| Japan | passed | passed | 0.070986 | 0.016988 | 0.053999 | 4.178728 |
| China | passed | passed | 0.188483 | 0.029790 | 0.158693 | 6.327036 |
| Russia | passed | passed | 0.132630 | 0.020090 | 0.112539 | 6.601701 |
| Canada | passed | passed | 0.167997 | 0.046660 | 0.121337 | 3.600467 |
| Australia | passed | passed | 0.194316 | 0.051934 | 0.142383 | 3.741632 |
| Sweden | passed | passed | 0.066994 | 0.014140 | 0.052854 | 4.737906 |
| Norway | passed | passed | 0.211211 | 0.024650 | 0.186561 | 8.568401 |
| Denmark | passed | passed | 0.174739 | 0.038891 | 0.135848 | 4.493073 |
| Finland | passed | passed | 0.250630 | 0.036135 | 0.214495 | 6.935942 |
| Poland | passed | passed | 0.140418 | 0.009661 | 0.130757 | 14.533927 |
| Hungary | passed | passed | 0.141796 | 0.029384 | 0.112413 | 4.825669 |
| Austria | passed | passed | 0.080308 | 0.026895 | 0.053413 | 2.985994 |
| Belgium | passed | passed | 0.190701 | 0.031407 | 0.159294 | 6.071965 |
| Netherlands | passed | passed | 0.035767 | 0.013394 | 0.022374 | 2.670460 |
| Ireland | passed | passed | 0.295054 | 0.000000 | 0.295054 | 295053541.660309 |
| Turkey | passed | passed | 0.197924 | 0.021374 | 0.176550 | 9.260057 |
| Iran | passed | passed | 0.194411 | 0.029051 | 0.165360 | 6.692125 |
| Iraq | passed | passed | 0.231618 | 0.022764 | 0.208855 | 10.174907 |
| Syria | passed | passed | 0.094395 | 0.037786 | 0.056609 | 2.498167 |
| Israel | passed | passed | 0.357032 | 0.053927 | 0.303105 | 6.620642 |
| Saudi Arabia | passed | passed | 0.228038 | 0.027065 | 0.200973 | 8.425523 |
| Pakistan | passed | passed | 0.331118 | 0.006921 | 0.324197 | 47.843960 |
| Thailand | passed | passed | 0.078623 | 0.015477 | 0.063145 | 5.079880 |
| Indonesia | passed | passed | 0.060676 | 0.017122 | 0.043554 | 3.543732 |
| Philippines | passed | passed | 0.139742 | 0.000000 | 0.139742 | 139741897.583008 |
| South Korea | passed | passed | 0.116099 | 0.019194 | 0.096905 | 6.048670 |
| Czech Republic | passed | passed | 0.112674 | 0.015611 | 0.097063 | 7.217430 |
| Tunisia | passed | passed | 0.304403 | 0.022436 | 0.281966 | 13.567377 |
| Chile | passed | passed | 0.193968 | 0.037094 | 0.156874 | 5.229046 |
| Peru | passed | passed | 0.171942 | 0.020342 | 0.151600 | 8.452511 |
| Cuba | passed | passed | 0.211102 | 0.033667 | 0.177435 | 6.270232 |
| Jamaica | passed | passed | 0.148421 | 0.050629 | 0.097792 | 2.931555 |
| largest planet | passed | passed | 0.335151 | 0.129168 | 0.205983 | 2.594689 |
| smallest planet | passed | passed | 0.077486 | 0.017555 | 0.059932 | 4.414012 |
| nearest star | passed | passed | 0.124466 | 0.097619 | 0.026846 | 1.275011 |
| grass color | passed | passed | 0.041309 | 0.027599 | 0.013711 | 1.496791 |
| opposite of hot | passed | passed | 0.239127 | 0.000000 | 0.239127 | 239127159.118652 |
| frozen water | passed | passed | 0.111161 | 0.020563 | 0.090598 | 5.405904 |
| baby dog | passed | passed | 0.411957 | 0.011079 | 0.400877 | 37.181966 |
| female parent | passed | passed | 0.215482 | 0.007282 | 0.208200 | 29.592666 |
| male parent | passed | passed | 0.222198 | 0.000000 | 0.222198 | 222197532.653809 |
| medical profession | passed | passed | 0.021875 | 0.004505 | 0.017371 | 4.856311 |
| school profession | passed | passed | 0.488796 | 0.012399 | 0.476397 | 39.423064 |
