# Live GPT-2 Fact Circuit Finder

This benchmark uses a real GPT-2 model, real forward-pass hooks, automatic layer/neuron scanning, graph construction, joint graph ablations, and graph-level negative controls. It is not synthetic and does not fit a surrogate.

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Median graph joint target-logit drop: `9.597953796386719`
- Median graph control p95 target-logit drop: `1.7921218872070312`
- Median graph lift over control p95: `4.993501607767975`
- Mean selected graph nodes: `3.74`
- Layers scanned: `[0, 3, 6, 9, 11]`
- Top neurons per layer: `6`
- Controls per family: `8`
- Max graph nodes: `4`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can automatically scan layers/top activation-contribution neurons, build a token/neuron/target graph, and validate the graph through joint causal ablation above graph-level negative controls across a factual prompt set.

This is not attention-head/path recovery yet, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.

## Cases

| subject | answer | status | target rank | graph nodes | graph drop | graph control p95 | graph margin | best layer | best neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| France | Paris | passed | 5 | 4 | 10.166962 | 3.105385 | 7.061577 | 0 | 2845 |
| Germany | Berlin | passed | 2 | 4 | 9.456749 | 1.702347 | 7.754402 | 0 | 2845 |
| Italy | Rome | passed | 1 | 4 | 9.516647 | 0.412781 | 9.103867 | 0 | 2845 |
| Spain | Madrid | passed | 1 | 4 | 9.555214 | 1.723885 | 7.831329 | 0 | 2845 |
| Portugal | Lisbon | passed | 2 | 4 | 9.319481 | 0.327751 | 8.991730 | 9 | 1828 |
| Greece | Athens | passed | 1 | 4 | 10.563339 | 1.459114 | 9.104225 | 0 | 2845 |
| Egypt | Cairo | passed | 2 | 4 | 9.767120 | 0.366425 | 9.400696 | 9 | 1828 |
| Japan | Tokyo | passed | 2 | 4 | 9.790222 | 1.599899 | 8.190323 | 0 | 2845 |
| China | Beijing | passed | 10 | 4 | 9.704384 | 2.474213 | 7.230171 | 0 | 2845 |
| Russia | Moscow | passed | 7 | 4 | 10.315933 | 2.593445 | 7.722488 | 0 | 2845 |
| Canada | Ottawa | passed | 23 | 4 | 10.940765 | 1.943192 | 8.997574 | 0 | 874 |
| Australia | Canberra | passed | 17 | 4 | 10.618103 | 1.777977 | 8.840126 | 0 | 874 |
| Brazil | Brasilia | skipped | - | - | - | - | - | - | - |
| Sweden | Stockholm | passed | 13 | 4 | 10.255722 | 2.531647 | 7.724075 | 0 | 2845 |
| Norway | Oslo | passed | 19 | 4 | 10.475067 | 1.080238 | 9.394829 | 0 | 2845 |
| Denmark | Copenhagen | passed | 11 | 4 | 10.139236 | 3.197227 | 6.942009 | 0 | 874 |
| Finland | Helsinki | passed | 7 | 4 | 8.937233 | 2.571976 | 6.365257 | 0 | 874 |
| Poland | Warsaw | passed | 1 | 4 | 10.261368 | 2.121391 | 8.139977 | 0 | 2845 |
| Hungary | Budapest | passed | 2 | 4 | 9.998451 | 3.274467 | 6.723984 | 0 | 2845 |
| Austria | Vienna | passed | 13 | 4 | 9.704758 | 1.983871 | 7.720886 | 0 | 2845 |
| Belgium | Brussels | passed | 11 | 4 | 9.527870 | 2.162788 | 7.365082 | 0 | 874 |
| Netherlands | Amsterdam | passed | 3 | 4 | 8.976707 | 2.124001 | 6.852707 | 0 | 874 |
| Ireland | Dublin | passed | 3 | 4 | 11.124382 | 2.500313 | 8.624069 | 0 | 874 |
| Turkey | Ankara | passed | 12 | 4 | 10.698349 | 2.431816 | 8.266533 | 0 | 874 |
| Iran | Tehran | passed | 4 | 4 | 9.620506 | 1.926605 | 7.693901 | 0 | 2845 |
| Iraq | Baghdad | passed | 10 | 4 | 9.597954 | 0.336266 | 9.261688 | 9 | 1828 |
| Syria | Damascus | passed | 18 | 4 | 10.072174 | 2.605751 | 7.466423 | 0 | 2845 |
| Jordan | Amman | skipped | - | - | - | - | - | - | - |
| Israel | Jerusalem | passed | 8 | 4 | 10.019386 | 2.052559 | 7.966827 | 0 | 2845 |
| Saudi Arabia | Riyadh | passed | 10 | 4 | 7.895279 | 0.272713 | 7.622566 | 9 | 1828 |
| Pakistan | Islamabad | passed | 3 | 4 | 8.546646 | 1.981613 | 6.565033 | 0 | 874 |
| Bangladesh | Dhaka | skipped | - | - | - | - | - | - | - |
| Thailand | Bangkok | passed | 8 | 4 | 9.315147 | 0.348244 | 8.966904 | 9 | 1828 |
| Vietnam | Hanoi | skipped | - | - | - | - | - | - | - |
| Indonesia | Jakarta | passed | 10 | 4 | 9.050201 | 0.835594 | 8.214607 | 9 | 1828 |
| Philippines | Manila | passed | 3 | 4 | 9.489128 | 2.437134 | 7.051994 | 0 | 874 |
| South Korea | Seoul | passed | 6 | 4 | 8.760201 | 0.352142 | 8.408058 | 0 | 2845 |
| Belarus | Minsk | skipped | - | - | - | - | - | - | - |
| Estonia | Tallinn | skipped | - | - | - | - | - | - | - |
| Latvia | Riga | skipped | - | - | - | - | - | - | - |
| Lithuania | Vilnius | skipped | - | - | - | - | - | - | - |
| Romania | Bucharest | skipped | - | - | - | - | - | - | - |
| Bulgaria | Sofia | skipped | - | - | - | - | - | - | - |
| Serbia | Belgrade | skipped | - | - | - | - | - | - | - |
| Croatia | Zagreb | skipped | - | - | - | - | - | - | - |
| Slovakia | Bratislava | skipped | - | - | - | - | - | - | - |
| Czech Republic | Prague | passed | 5 | 4 | 8.615654 | 1.112953 | 7.502701 | 0 | 874 |
| Switzerland | Bern | passed | 299 | 4 | 9.574471 | 1.847267 | 7.727203 | 0 | 2845 |
| Morocco | Rabat | skipped | - | - | - | - | - | - | - |
| Algeria | Algiers | skipped | - | - | - | - | - | - | - |
| Tunisia | Tunis | passed | 23 | 4 | 7.757156 | 2.060722 | 5.696434 | 0 | 2845 |
| Kenya | Nairobi | skipped | - | - | - | - | - | - | - |
| Nigeria | Abuja | skipped | - | - | - | - | - | - | - |
| Ghana | Accra | skipped | - | - | - | - | - | - | - |
| Senegal | Dakar | skipped | - | - | - | - | - | - | - |
| Mali | Bamako | skipped | - | - | - | - | - | - | - |
| Niger | Niamey | skipped | - | - | - | - | - | - | - |
| Sudan | Khartoum | skipped | - | - | - | - | - | - | - |
| Uganda | Kampala | skipped | - | - | - | - | - | - | - |
| Tanzania | Dodoma | skipped | - | - | - | - | - | - | - |
| Rwanda | Kigali | skipped | - | - | - | - | - | - | - |
| Chile | Santiago | passed | 11 | 4 | 8.605949 | 1.792122 | 6.813828 | 0 | 2845 |
| Peru | Lima | passed | 63 | 3 | 7.395676 | 2.287041 | 5.108635 | 0 | 2845 |
| Colombia | Bogota | skipped | - | - | - | - | - | - | - |
| Venezuela | Caracas | skipped | - | - | - | - | - | - | - |
| Cuba | Havana | passed | 16 | 4 | 9.912682 | 2.031479 | 7.881203 | 0 | 2845 |
| Jamaica | Kingston | passed | 43 | 4 | 9.877563 | 0.439796 | 9.437767 | 9 | 1828 |
| Uruguay | Montevideo | skipped | - | - | - | - | - | - | - |
| Paraguay | Asuncion | skipped | - | - | - | - | - | - | - |
| Ecuador | Quito | skipped | - | - | - | - | - | - | - |
| largest planet | Jupiter | passed | 7 | 4 | 10.907997 | 0.169624 | 10.738373 | 9 | 1828 |
| smallest planet | Mercury | passed | 43 | 4 | 12.829201 | 1.625832 | 11.203369 | 0 | 874 |
| nearest star | Sun | passed | 1 | 1 | 1.802338 | 0.512520 | 1.289818 | 11 | 2910 |
| grass color | green | passed | 7 | 3 | 8.413315 | 2.103096 | 6.310219 | 0 | 874 |
| opposite of hot | cold | passed | 1 | 2 | 5.497864 | 0.872467 | 4.625397 | 0 | 874 |
| number after nine | ten | passed | 119 | 4 | 12.493248 | 0.001923 | 12.491325 | 0 | 874 |
| month after January | February | passed | 205 | 3 | 10.506828 | 1.041466 | 9.465363 | 0 | 874 |
| frozen water | ice | passed | 2 | 1 | 8.468842 | 2.030708 | 6.438133 | 0 | 2754 |
| baby dog | puppy | passed | 5 | 4 | 4.166328 | 0.156548 | 4.009781 | 9 | 1312 |
| female parent | mother | passed | 4 | 2 | 5.352074 | 0.951622 | 4.400452 | 9 | 1312 |
