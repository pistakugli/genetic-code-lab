# Live GPT-2 Fact Circuit Finder

This benchmark uses a real GPT-2 model, real forward-pass hooks, automatic layer/neuron scanning, graph construction, joint graph ablations, and graph-level negative controls. It is not synthetic and does not fit a surrogate.

- Pass: `True`
- Valid cases: `50`
- Strict case definition: `graphGate.pass && hardeningGate.pass`
- Passed cases: `50`
- Graph-passed cases: `50`
- Hardening-passed cases: `50`
- Pass rate: `1.0`
- Median graph joint target-logit drop: `0.25433349609375`
- Median graph control p95 target-logit drop: `0.06884765625`
- Median graph lift over control p95: `3.8081556082148498`
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
| France | Paris | passed | 5 | 1 | 0.106323 | 0.046822 | 0.059502 | True | 9 | 1828 |
| Germany | Berlin | passed | 2 | 1 | 0.126610 | 0.027687 | 0.098923 | True | 9 | 1409 |
| Italy | Rome | passed | 1 | 1 | 0.061676 | 0.023605 | 0.038071 | True | 9 | 1828 |
| Spain | Madrid | passed | 1 | 1 | 0.141235 | 0.018570 | 0.122665 | True | 9 | 2416 |
| Portugal | Lisbon | passed | 2 | 2 | 0.548134 | 0.454262 | 0.093872 | True | 9 | 1828 |
| Greece | Athens | passed | 1 | 1 | 0.194641 | 0.093781 | 0.100861 | True | 9 | 1828 |
| Egypt | Cairo | passed | 2 | 2 | 0.625496 | 0.309502 | 0.315994 | True | 9 | 1828 |
| Japan | Tokyo | passed | 2 | 1 | 0.058250 | 0.005241 | 0.053009 | True | 9 | 1828 |
| China | Beijing | passed | 10 | 4 | 0.483406 | 0.270653 | 0.212753 | True | 9 | 2416 |
| Russia | Moscow | passed | 7 | 2 | 0.820839 | 0.046822 | 0.774017 | True | 9 | 2416 |
| Canada | Ottawa | passed | 23 | 2 | 0.266693 | 0.016457 | 0.250237 | True | 9 | 2746 |
| Australia | Canberra | passed | 17 | 2 | 0.346680 | 0.170929 | 0.175751 | True | 9 | 1828 |
| Brazil | Brasilia | skipped | - | - | - | - | - | - | - | - |
| Sweden | Stockholm | passed | 13 | 2 | 0.808968 | 0.169769 | 0.639198 | True | 9 | 1828 |
| Norway | Oslo | passed | 19 | 1 | 0.448708 | 0.067078 | 0.381630 | True | 9 | 1828 |
| Denmark | Copenhagen | passed | 11 | 2 | 0.614380 | 0.169205 | 0.445175 | True | 9 | 1828 |
| Finland | Helsinki | passed | 7 | 2 | 0.391846 | 0.237419 | 0.154427 | True | 9 | 1828 |
| Poland | Warsaw | passed | 1 | 2 | 0.723053 | 0.069359 | 0.653694 | True | 9 | 2416 |
| Hungary | Budapest | passed | 2 | 1 | 0.220695 | 0.136948 | 0.083748 | True | 9 | 2416 |
| Austria | Vienna | passed | 13 | 2 | 0.611595 | 0.178543 | 0.433052 | True | 9 | 1828 |
| Belgium | Brussels | passed | 11 | 1 | 0.182159 | 0.025551 | 0.156609 | True | 9 | 1828 |
| Netherlands | Amsterdam | passed | 3 | 2 | 0.254333 | 0.180481 | 0.073853 | True | 9 | 1828 |
| Ireland | Dublin | passed | 3 | 2 | 0.431702 | 0.147163 | 0.284538 | True | 9 | 1828 |
| Turkey | Ankara | passed | 12 | 1 | 1.704391 | 0.087143 | 1.617249 | True | 9 | 2416 |
| Iran | Tehran | passed | 4 | 1 | 0.102859 | 0.060272 | 0.042587 | True | 9 | 2416 |
| Iraq | Baghdad | passed | 10 | 1 | 0.188354 | 0.031136 | 0.157219 | True | 9 | 1828 |
| Syria | Damascus | passed | 18 | 1 | 0.152733 | 0.025452 | 0.127281 | True | 9 | 2416 |
| Jordan | Amman | skipped | - | - | - | - | - | - | - | - |
| Israel | Jerusalem | passed | 8 | 2 | 0.211647 | 0.175850 | 0.035797 | True | 9 | 1828 |
| Saudi Arabia | Riyadh | passed | 10 | 1 | 0.189781 | 0.029198 | 0.160583 | True | 9 | 1828 |
| Pakistan | Islamabad | passed | 3 | 1 | 0.427422 | 0.068848 | 0.358574 | True | 9 | 1828 |
| Bangladesh | Dhaka | skipped | - | - | - | - | - | - | - | - |
| Thailand | Bangkok | passed | 8 | 2 | 0.421783 | 0.175369 | 0.246414 | True | 9 | 1828 |
| Vietnam | Hanoi | skipped | - | - | - | - | - | - | - | - |
| Indonesia | Jakarta | passed | 10 | 2 | 0.298599 | 0.068398 | 0.230202 | True | 9 | 2416 |
| Philippines | Manila | passed | 3 | 1 | 0.139473 | 0.023491 | 0.115982 | True | 9 | 1828 |
| South Korea | Seoul | passed | 6 | 1 | 0.105270 | 0.005699 | 0.099571 | True | 9 | 1828 |
| Belarus | Minsk | skipped | - | - | - | - | - | - | - | - |
| Estonia | Tallinn | skipped | - | - | - | - | - | - | - | - |
| Latvia | Riga | skipped | - | - | - | - | - | - | - | - |
| Lithuania | Vilnius | skipped | - | - | - | - | - | - | - | - |
| Romania | Bucharest | skipped | - | - | - | - | - | - | - | - |
| Bulgaria | Sofia | skipped | - | - | - | - | - | - | - | - |
| Serbia | Belgrade | skipped | - | - | - | - | - | - | - | - |
| Croatia | Zagreb | skipped | - | - | - | - | - | - | - | - |
| Slovakia | Bratislava | skipped | - | - | - | - | - | - | - | - |
| Czech Republic | Prague | passed | 5 | 2 | 0.932251 | 0.703766 | 0.228485 | True | 9 | 1828 |
| Switzerland | Bern | passed | 299 | 2 | 0.551514 | 0.165497 | 0.386017 | True | 9 | 1828 |
| Morocco | Rabat | skipped | - | - | - | - | - | - | - | - |
| Algeria | Algiers | skipped | - | - | - | - | - | - | - | - |
| Tunisia | Tunis | passed | 23 | 2 | 0.453575 | 0.000481 | 0.453094 | True | 9 | 2416 |
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
| Chile | Santiago | passed | 11 | 1 | 0.220192 | 0.010521 | 0.209671 | True | 9 | 1828 |
| Peru | Lima | passed | 63 | 2 | 0.438301 | 0.170929 | 0.267372 | True | 9 | 1828 |
| Colombia | Bogota | skipped | - | - | - | - | - | - | - | - |
| Venezuela | Caracas | skipped | - | - | - | - | - | - | - | - |
| Cuba | Havana | passed | 16 | 1 | 0.104141 | 0.006386 | 0.097755 | True | 9 | 1828 |
| Jamaica | Kingston | passed | 43 | 2 | 0.658447 | 0.162010 | 0.496437 | True | 9 | 1828 |
| Uruguay | Montevideo | skipped | - | - | - | - | - | - | - | - |
| Paraguay | Asuncion | skipped | - | - | - | - | - | - | - | - |
| Ecuador | Quito | skipped | - | - | - | - | - | - | - | - |
| largest planet | Jupiter | passed | 7 | 2 | 0.874313 | 0.371284 | 0.503029 | True | 9 | 662 |
| smallest planet | Mercury | passed | 43 | 1 | 0.060318 | 0.000618 | 0.059700 | True | 9 | 662 |
| nearest star | Sun | passed | 1 | 1 | 0.171562 | 0.000000 | 0.171562 | True | 11 | 2527 |
| grass color | green | passed | 7 | 2 | 0.185112 | 0.000000 | 0.185112 | True | 11 | 2471 |
| opposite of hot | cold | passed | 1 | 1 | 0.252182 | 0.074745 | 0.177437 | True | 6 | 219 |
| number after nine | ten | passed | 119 | 1 | 0.119781 | 0.068764 | 0.051018 | True | 9 | 2798 |
| month after January | February | passed | 205 | 5 | 0.914215 | 0.405937 | 0.508278 | True | 9 | 1873 |
| frozen water | ice | passed | 2 | 1 | 0.209557 | 0.008904 | 0.200653 | True | 11 | 965 |
| baby dog | puppy | passed | 5 | 1 | 0.139114 | 0.074509 | 0.064606 | True | 11 | 165 |
| female parent | mother | passed | 4 | 2 | 0.588516 | 0.154541 | 0.433975 | True | 11 | 2627 |
