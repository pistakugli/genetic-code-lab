# Live GPT-2 Fact Circuit Finder

This benchmark uses a real GPT-2 model, real forward-pass hooks, automatic layer/neuron scanning, graph construction, joint graph ablations, and graph-level negative controls. It is not synthetic and does not fit a surrogate.

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Median graph joint target-logit drop: `0.2544403076171875`
- Median graph control p95 target-logit drop: `0.06884765625`
- Median graph lift over control p95: `3.8083037124802526`
- Mean selected graph nodes: `1.6`
- Paraphrase validation pass rate: `1.0`
- Wrong-answer specificity pass rate: `1.0`
- Negative-fact specificity pass rate: `1.0`
- Unrelated-logit preservation pass rate: `0.955`
- Primary layers scanned: `[6, 9, 11]`
- Fallback layers scanned on strict failures: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`
- Top neurons per layer: `50`
- Controls per family: `8`
- Max graph nodes: `5`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can automatically scan layers/top activation-contribution neurons, build a token/neuron/target graph, validate the graph through joint causal ablation above graph-level controls, and harden it against paraphrase, wrong-answer, negative-fact, and unrelated-logit checks across a factual prompt set.

This is not attention-head/path recovery yet, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.

## Cases

| subject | answer | status | target rank | graph nodes | graph drop | graph control p95 | graph margin | hardening | best layer | best neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| France | Paris | passed | 5 | 1 | 0.106339 | 0.046806 | 0.059532 | True | 9 | 1828 |
| Germany | Berlin | passed | 2 | 1 | 0.126572 | 0.027695 | 0.098877 | True | 9 | 1409 |
| Italy | Rome | passed | 1 | 1 | 0.061691 | 0.023575 | 0.038116 | True | 9 | 1828 |
| Spain | Madrid | passed | 1 | 1 | 0.141266 | 0.018578 | 0.122688 | True | 9 | 2416 |
| Portugal | Lisbon | passed | 2 | 2 | 0.548134 | 0.454254 | 0.093880 | True | 9 | 1828 |
| Greece | Athens | passed | 1 | 1 | 0.194679 | 0.093758 | 0.100922 | True | 9 | 1828 |
| Egypt | Cairo | passed | 2 | 2 | 0.625488 | 0.309464 | 0.316025 | True | 9 | 1828 |
| Japan | Tokyo | passed | 2 | 1 | 0.058182 | 0.005188 | 0.052994 | True | 9 | 1828 |
| China | Beijing | passed | 10 | 4 | 0.483376 | 0.270584 | 0.212791 | True | 9 | 2416 |
| Russia | Moscow | passed | 7 | 2 | 0.820930 | 0.046898 | 0.774033 | True | 9 | 2416 |
| Canada | Ottawa | passed | 23 | 2 | 0.266602 | 0.016441 | 0.250160 | True | 9 | 2746 |
| Australia | Canberra | passed | 17 | 2 | 0.346642 | 0.171021 | 0.175621 | True | 9 | 1828 |
| Brazil | Brasilia | skipped | - | - | - | - | - | - | - | - |
| Sweden | Stockholm | passed | 13 | 2 | 0.809036 | 0.169815 | 0.639221 | True | 9 | 1828 |
| Norway | Oslo | passed | 19 | 1 | 0.448746 | 0.067116 | 0.381630 | True | 9 | 1828 |
| Denmark | Copenhagen | passed | 11 | 2 | 0.614365 | 0.169182 | 0.445183 | True | 9 | 1828 |
| Finland | Helsinki | passed | 7 | 2 | 0.391907 | 0.237572 | 0.154335 | True | 9 | 1828 |
| Poland | Warsaw | passed | 1 | 2 | 0.723038 | 0.069427 | 0.653610 | True | 9 | 2416 |
| Hungary | Budapest | passed | 2 | 1 | 0.220749 | 0.136971 | 0.083778 | True | 9 | 2416 |
| Austria | Vienna | passed | 13 | 2 | 0.611626 | 0.178505 | 0.433121 | True | 9 | 1828 |
| Belgium | Brussels | passed | 11 | 1 | 0.182137 | 0.025505 | 0.156631 | True | 9 | 1828 |
| Netherlands | Amsterdam | passed | 3 | 2 | 0.254440 | 0.180557 | 0.073883 | True | 9 | 1828 |
| Ireland | Dublin | passed | 3 | 2 | 0.431648 | 0.147163 | 0.284485 | True | 9 | 1828 |
| Turkey | Ankara | passed | 12 | 1 | 1.704414 | 0.087135 | 1.617279 | True | 9 | 2416 |
| Iran | Tehran | passed | 4 | 1 | 0.102852 | 0.060219 | 0.042633 | True | 9 | 2416 |
| Iraq | Baghdad | passed | 10 | 1 | 0.188354 | 0.031128 | 0.157227 | True | 9 | 1828 |
| Syria | Damascus | passed | 18 | 1 | 0.152763 | 0.025475 | 0.127289 | True | 9 | 2416 |
| Jordan | Amman | skipped | - | - | - | - | - | - | - | - |
| Israel | Jerusalem | passed | 8 | 2 | 0.211655 | 0.175865 | 0.035789 | True | 9 | 1828 |
| Saudi Arabia | Riyadh | passed | 10 | 1 | 0.189873 | 0.029236 | 0.160637 | True | 9 | 1828 |
| Pakistan | Islamabad | passed | 3 | 1 | 0.427414 | 0.068848 | 0.358566 | True | 9 | 1828 |
| Bangladesh | Dhaka | skipped | - | - | - | - | - | - | - | - |
| Thailand | Bangkok | passed | 8 | 2 | 0.421753 | 0.175323 | 0.246429 | True | 9 | 1828 |
| Vietnam | Hanoi | skipped | - | - | - | - | - | - | - | - |
| Indonesia | Jakarta | passed | 10 | 2 | 0.298645 | 0.068420 | 0.230225 | True | 9 | 2416 |
| Philippines | Manila | passed | 3 | 1 | 0.139435 | 0.023476 | 0.115959 | True | 9 | 1828 |
| South Korea | Seoul | passed | 6 | 1 | 0.105309 | 0.005745 | 0.099564 | True | 9 | 1828 |
| Belarus | Minsk | skipped | - | - | - | - | - | - | - | - |
| Estonia | Tallinn | skipped | - | - | - | - | - | - | - | - |
| Latvia | Riga | skipped | - | - | - | - | - | - | - | - |
| Lithuania | Vilnius | skipped | - | - | - | - | - | - | - | - |
| Romania | Bucharest | skipped | - | - | - | - | - | - | - | - |
| Bulgaria | Sofia | skipped | - | - | - | - | - | - | - | - |
| Serbia | Belgrade | skipped | - | - | - | - | - | - | - | - |
| Croatia | Zagreb | skipped | - | - | - | - | - | - | - | - |
| Slovakia | Bratislava | skipped | - | - | - | - | - | - | - | - |
| Czech Republic | Prague | passed | 5 | 2 | 0.932259 | 0.703743 | 0.228516 | True | 9 | 1828 |
| Switzerland | Bern | passed | 299 | 2 | 0.551476 | 0.165527 | 0.385948 | True | 9 | 1828 |
| Morocco | Rabat | skipped | - | - | - | - | - | - | - | - |
| Algeria | Algiers | skipped | - | - | - | - | - | - | - | - |
| Tunisia | Tunis | passed | 23 | 2 | 0.453629 | 0.000526 | 0.453102 | True | 9 | 2416 |
| Kenya | Nairobi | skipped | - | - | - | - | - | - | - | - |
| Nigeria | Abuja | skipped | - | - | - | - | - | - | - | - |
| Ghana | Accra | skipped | - | - | - | - | - | - | - | - |
| Senegal | Dakar | skipped | - | - | - | - | - | - | - | - |
| Mali | Bamako | skipped | - | - | - | - | - | - | - | - |
| Niger | Niamey | skipped | - | - | - | - | - | - | - | - |
| Sudan | Khartoum | skipped | - | - | - | - | - | - | - | - |
| Uganda | Kampala | skipped | - | - | - | - | - | - | - | - |
| Tanzania | Dodoma | skipped | - | - | - | - | - | - | - | - |
| Rwanda | Kigali | skipped | - | - | - | - | - | - | - | - |
| Chile | Santiago | passed | 11 | 1 | 0.220177 | 0.010612 | 0.209564 | True | 9 | 1828 |
| Peru | Lima | passed | 63 | 2 | 0.438194 | 0.170837 | 0.267357 | True | 9 | 1828 |
| Colombia | Bogota | skipped | - | - | - | - | - | - | - | - |
| Venezuela | Caracas | skipped | - | - | - | - | - | - | - | - |
| Cuba | Havana | passed | 16 | 1 | 0.104118 | 0.006363 | 0.097755 | True | 9 | 1828 |
| Jamaica | Kingston | passed | 43 | 2 | 0.658508 | 0.162041 | 0.496468 | True | 9 | 1828 |
| Uruguay | Montevideo | skipped | - | - | - | - | - | - | - | - |
| Paraguay | Asuncion | skipped | - | - | - | - | - | - | - | - |
| Ecuador | Quito | skipped | - | - | - | - | - | - | - | - |
| largest planet | Jupiter | passed | 7 | 2 | 0.874321 | 0.371292 | 0.503029 | True | 9 | 662 |
| smallest planet | Mercury | passed | 43 | 1 | 0.060257 | 0.000610 | 0.059647 | True | 9 | 662 |
| nearest star | Sun | passed | 1 | 1 | 0.171555 | 0.000000 | 0.171555 | True | 11 | 2527 |
| grass color | green | passed | 7 | 2 | 0.185150 | 0.000000 | 0.185150 | True | 11 | 2471 |
| opposite of hot | cold | passed | 1 | 1 | 0.252220 | 0.074814 | 0.177406 | True | 6 | 219 |
| number after nine | ten | passed | 119 | 1 | 0.119766 | 0.068764 | 0.051003 | True | 9 | 2798 |
| month after January | February | passed | 205 | 5 | 0.914276 | 0.405907 | 0.508369 | True | 9 | 1873 |
| frozen water | ice | passed | 2 | 1 | 0.209557 | 0.008919 | 0.200638 | True | 11 | 965 |
| baby dog | puppy | passed | 5 | 1 | 0.139107 | 0.074493 | 0.064613 | True | 11 | 165 |
| female parent | mother | passed | 4 | 2 | 0.588539 | 0.154541 | 0.433998 | True | 11 | 2627 |
