# Live GPT-2 Fact Circuit Finder

This benchmark uses a real GPT-2 model, real forward-pass hooks, automatic layer/neuron scanning, graph construction, joint graph ablations, and graph-level negative controls. It is not synthetic and does not fit a surrogate.

- Pass: `True`
- Valid cases: `50`
- Passed cases: `45`
- Pass rate: `0.9`
- Median graph joint target-logit drop: `0.7129135131835938`
- Median graph control p95 target-logit drop: `0.3569793701171875`
- Median graph lift over control p95: `1.845643203157706`
- Mean selected graph nodes: `4.96`
- Paraphrase validation pass rate: `0.75`
- Wrong-answer specificity pass rate: `0.9666666666666667`
- Negative-fact specificity pass rate: `1.0`
- Unrelated-logit preservation pass rate: `0.8725`
- Layers scanned: `[6, 9, 11]`
- Top neurons per layer: `50`
- Controls per family: `8`
- Max graph nodes: `5`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can automatically scan layers/top activation-contribution neurons, build a token/neuron/target graph, validate the graph through joint causal ablation above graph-level controls, and harden it against paraphrase, wrong-answer, negative-fact, and unrelated-logit checks across a factual prompt set.

This is not attention-head/path recovery yet, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.

## Cases

| subject | answer | status | target rank | graph nodes | graph drop | graph control p95 | graph margin | hardening | best layer | best neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| France | Paris | passed | 5 | 5 | 0.541969 | 0.356186 | 0.185783 | True | 9 | 1828 |
| Germany | Berlin | passed | 2 | 5 | 0.593224 | 0.000000 | 0.593224 | False | 9 | 1409 |
| Italy | Rome | passed | 1 | 5 | 0.479263 | 0.377739 | 0.101524 | False | 9 | 1828 |
| Spain | Madrid | passed | 1 | 5 | 0.568916 | 0.000000 | 0.568916 | True | 9 | 2416 |
| Portugal | Lisbon | failed | 2 | 5 | 0.869118 | 0.983238 | -0.114120 | True | 9 | 1828 |
| Greece | Athens | passed | 1 | 5 | 0.739990 | 0.608505 | 0.131485 | True | 9 | 1828 |
| Egypt | Cairo | passed | 2 | 5 | 0.799164 | 0.554832 | 0.244331 | False | 9 | 1828 |
| Japan | Tokyo | passed | 2 | 5 | 0.542244 | 0.302277 | 0.239967 | False | 9 | 1828 |
| China | Beijing | passed | 10 | 5 | 0.529465 | 0.481621 | 0.047844 | True | 9 | 2416 |
| Russia | Moscow | passed | 7 | 5 | 1.534798 | 0.188492 | 1.346306 | False | 9 | 2416 |
| Canada | Ottawa | passed | 23 | 5 | 0.539627 | 0.000000 | 0.539627 | True | 9 | 2746 |
| Australia | Canberra | passed | 17 | 5 | 0.600212 | 0.306099 | 0.294113 | True | 9 | 1828 |
| Brazil | Brasilia | skipped | - | - | - | - | - | - | - | - |
| Sweden | Stockholm | passed | 13 | 5 | 0.963264 | 0.340660 | 0.622604 | False | 9 | 1828 |
| Norway | Oslo | passed | 19 | 5 | 1.150993 | 0.476501 | 0.674492 | False | 9 | 1828 |
| Denmark | Copenhagen | passed | 11 | 5 | 0.941460 | 0.431267 | 0.510193 | True | 9 | 1828 |
| Finland | Helsinki | passed | 7 | 5 | 1.223145 | 0.358612 | 0.864532 | True | 9 | 1828 |
| Poland | Warsaw | passed | 1 | 5 | 0.823303 | 0.544434 | 0.278870 | True | 9 | 2416 |
| Hungary | Budapest | passed | 2 | 5 | 1.229851 | 0.190628 | 1.039223 | True | 9 | 2416 |
| Austria | Vienna | passed | 13 | 5 | 0.866959 | 0.377350 | 0.489609 | False | 9 | 1828 |
| Belgium | Brussels | passed | 11 | 3 | 0.688850 | 0.335548 | 0.353302 | False | 9 | 1828 |
| Netherlands | Amsterdam | failed | 3 | 5 | 0.310921 | 0.612320 | -0.301399 | False | 9 | 1828 |
| Ireland | Dublin | passed | 3 | 5 | 0.608452 | 0.428818 | 0.179634 | False | 9 | 1828 |
| Turkey | Ankara | passed | 12 | 5 | 2.489037 | 0.223511 | 2.265526 | True | 9 | 2416 |
| Iran | Tehran | passed | 4 | 5 | 0.887604 | 0.650276 | 0.237328 | True | 9 | 2416 |
| Iraq | Baghdad | passed | 10 | 5 | 0.763351 | 0.336266 | 0.427086 | False | 9 | 1828 |
| Syria | Damascus | passed | 18 | 5 | 0.995361 | 0.658424 | 0.336937 | True | 9 | 2416 |
| Jordan | Amman | skipped | - | - | - | - | - | - | - | - |
| Israel | Jerusalem | passed | 8 | 5 | 0.607231 | 0.412682 | 0.194550 | False | 9 | 1828 |
| Saudi Arabia | Riyadh | passed | 10 | 5 | 1.072502 | 0.271652 | 0.800850 | False | 9 | 1828 |
| Pakistan | Islamabad | passed | 3 | 5 | 0.792046 | 0.325798 | 0.466248 | False | 9 | 1828 |
| Bangladesh | Dhaka | skipped | - | - | - | - | - | - | - | - |
| Thailand | Bangkok | passed | 8 | 5 | 0.824661 | 0.577202 | 0.247459 | False | 9 | 1828 |
| Vietnam | Hanoi | skipped | - | - | - | - | - | - | - | - |
| Indonesia | Jakarta | passed | 10 | 5 | 1.071083 | 1.047318 | 0.023766 | False | 9 | 2416 |
| Philippines | Manila | passed | 3 | 5 | 0.706352 | 0.382713 | 0.323639 | False | 9 | 1828 |
| South Korea | Seoul | failed | 6 | 5 | 0.309166 | 0.294899 | 0.014267 | False | 9 | 1828 |
| Belarus | Minsk | skipped | - | - | - | - | - | - | - | - |
| Estonia | Tallinn | skipped | - | - | - | - | - | - | - | - |
| Latvia | Riga | skipped | - | - | - | - | - | - | - | - |
| Lithuania | Vilnius | skipped | - | - | - | - | - | - | - | - |
| Romania | Bucharest | skipped | - | - | - | - | - | - | - | - |
| Bulgaria | Sofia | skipped | - | - | - | - | - | - | - | - |
| Serbia | Belgrade | skipped | - | - | - | - | - | - | - | - |
| Croatia | Zagreb | skipped | - | - | - | - | - | - | - | - |
| Slovakia | Bratislava | skipped | - | - | - | - | - | - | - | - |
| Czech Republic | Prague | passed | 5 | 5 | 1.340836 | 0.356979 | 0.983856 | True | 9 | 1828 |
| Switzerland | Bern | passed | 299 | 5 | 0.642700 | 0.364204 | 0.278496 | True | 9 | 1828 |
| Morocco | Rabat | skipped | - | - | - | - | - | - | - | - |
| Algeria | Algiers | skipped | - | - | - | - | - | - | - | - |
| Tunisia | Tunis | failed | 23 | 5 | 0.453911 | 0.837578 | -0.383667 | True | 9 | 2416 |
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
| Chile | Santiago | passed | 11 | 5 | 0.712914 | 0.496262 | 0.216652 | False | 9 | 1828 |
| Peru | Lima | passed | 63 | 5 | 0.383720 | 0.334801 | 0.048920 | False | 9 | 1828 |
| Colombia | Bogota | skipped | - | - | - | - | - | - | - | - |
| Venezuela | Caracas | skipped | - | - | - | - | - | - | - | - |
| Cuba | Havana | passed | 16 | 5 | 0.912186 | 0.368591 | 0.543594 | True | 9 | 1828 |
| Jamaica | Kingston | passed | 43 | 5 | 0.489510 | 0.318504 | 0.171005 | False | 9 | 1828 |
| Uruguay | Montevideo | skipped | - | - | - | - | - | - | - | - |
| Paraguay | Asuncion | skipped | - | - | - | - | - | - | - | - |
| Ecuador | Quito | skipped | - | - | - | - | - | - | - | - |
| largest planet | Jupiter | passed | 7 | 5 | 1.268684 | 0.832397 | 0.436287 | True | 9 | 662 |
| smallest planet | Mercury | passed | 43 | 5 | 0.425140 | 0.293915 | 0.131226 | True | 9 | 662 |
| nearest star | Sun | passed | 1 | 5 | 0.519554 | 0.015656 | 0.503899 | True | 11 | 2527 |
| grass color | green | passed | 7 | 5 | 0.666512 | 0.224762 | 0.441750 | True | 11 | 2471 |
| opposite of hot | cold | passed | 1 | 5 | 0.547638 | 0.184494 | 0.363144 | True | 6 | 219 |
| number after nine | ten | failed | 119 | 5 | 0.233307 | 0.584312 | -0.351006 | False | 9 | 2798 |
| month after January | February | passed | 205 | 5 | 0.914276 | 0.152924 | 0.761353 | True | 9 | 1873 |
| frozen water | ice | passed | 2 | 5 | 0.510277 | 0.013748 | 0.496529 | True | 11 | 965 |
| baby dog | puppy | passed | 5 | 5 | 2.533447 | 0.452110 | 2.081337 | False | 11 | 165 |
| female parent | mother | passed | 4 | 5 | 0.883804 | 0.135529 | 0.748276 | True | 11 | 2627 |
