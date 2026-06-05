# Live GPT-2 Attention Path Circuit Finder

This benchmark uses a real GPT-2 model, real attention-path hooks, automatic layer/head scanning, joint attention-head ablations, and attention-head negative controls. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `48`
- Pass rate: `0.96`
- Median attention graph target-logit drop: `3.77752685546875`
- Median attention control p95 target-logit drop: `0.8765945434570312`
- Median attention graph lift over control p95: `4.1454169816074575`
- Mean selected attention heads: `3.0`
- Unique selected attention heads: `20`
- Median wrong-answer collateral ratio: `0.7703779014285085`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can scan attention heads, build token/head/target path graph evidence, and validate selected attention paths through causal head ablation above attention-head controls across a factual prompt set.

This is not answer-specific factual identity. Attention heads can carry broad task or token-class effects, so wrong-answer collateral is reported but not gated in this Stage-K path graph. This is also not full attention-circuit recovery across all model components, not attention-to-MLP composition recovery, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.

## Cases

| subject | status | target rank | heads | graph drop | control p95 | margin | wrong ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| France | passed | 5 | 3 | 3.152863 | 2.209396 | 0.943466 | 0.725828 |
| Germany | passed | 2 | 3 | 3.531776 | 1.192802 | 2.338974 | 0.826971 |
| Italy | passed | 1 | 3 | 4.717697 | 2.657845 | 2.059853 | 0.814425 |
| Spain | passed | 1 | 3 | 5.474655 | 1.073280 | 4.401375 | 0.804015 |
| Portugal | passed | 2 | 3 | 6.011963 | 0.533745 | 5.478218 | 0.804020 |
| Greece | passed | 1 | 3 | 4.866730 | 1.276932 | 3.589798 | 0.873483 |
| Egypt | passed | 2 | 3 | 5.644661 | 0.980911 | 4.663750 | 0.828830 |
| Japan | passed | 2 | 3 | 3.777527 | 0.867996 | 2.909531 | 0.811120 |
| China | passed | 10 | 3 | 2.946770 | 1.123528 | 1.823242 | 0.590313 |
| Russia | passed | 7 | 3 | 2.078644 | 0.925980 | 1.152664 | 0.770378 |
| Canada | passed | 23 | 3 | 2.032089 | 1.059700 | 0.972389 | 0.112600 |
| Australia | passed | 17 | 3 | 3.461090 | 0.890854 | 2.570236 | 0.330132 |
| Sweden | passed | 13 | 3 | 3.234329 | 1.363617 | 1.870712 | 0.683195 |
| Norway | passed | 19 | 3 | 3.348740 | 0.964104 | 2.384636 | 0.502768 |
| Denmark | passed | 11 | 3 | 3.917587 | 1.379059 | 2.538528 | 0.635930 |
| Finland | passed | 7 | 3 | 3.866821 | 0.857628 | 3.009193 | 0.695538 |
| Poland | passed | 1 | 3 | 4.750610 | 0.717224 | 4.033386 | 0.741784 |
| Hungary | passed | 2 | 3 | 3.933868 | 1.279472 | 2.654396 | 0.759230 |
| Austria | passed | 13 | 3 | 2.606667 | 2.324936 | 0.281731 | 0.680016 |
| Belgium | failed | 11 | 3 | 2.120430 | 2.242706 | -0.122276 | 0.798762 |
| Netherlands | passed | 3 | 3 | 3.275795 | 2.943375 | 0.332420 | 0.756368 |
| Ireland | passed | 3 | 3 | 3.138191 | 0.757027 | 2.381165 | 0.508457 |
| Turkey | passed | 12 | 3 | 6.852661 | 0.883202 | 5.969460 | 0.782751 |
| Iran | passed | 4 | 3 | 6.840469 | 0.307297 | 6.533173 | 0.873462 |
| Iraq | passed | 10 | 3 | 8.229561 | 0.221779 | 8.007782 | 0.854224 |
| Syria | passed | 18 | 3 | 5.724243 | 0.854179 | 4.870064 | 0.849282 |
| Israel | passed | 8 | 3 | 4.206566 | 1.285622 | 2.920944 | 0.748728 |
| Saudi Arabia | passed | 10 | 3 | 5.684113 | 0.669350 | 5.014763 | 0.764922 |
| Pakistan | passed | 3 | 3 | 5.772362 | 0.796432 | 4.975929 | 0.691480 |
| Thailand | passed | 8 | 3 | 5.715866 | 0.360359 | 5.355507 | 0.712648 |
| Indonesia | passed | 10 | 3 | 7.067856 | 0.527428 | 6.540428 | 0.842940 |
| Philippines | passed | 3 | 3 | 5.842171 | 0.852280 | 4.989891 | 0.899700 |
| South Korea | passed | 6 | 3 | 3.598770 | 0.765404 | 2.833366 | 0.718568 |
| Czech Republic | passed | 5 | 3 | 4.090225 | 0.876595 | 3.213631 | 0.843352 |
| Switzerland | passed | 299 | 3 | 3.009659 | 0.938881 | 2.070778 | 0.868960 |
| Tunisia | passed | 23 | 3 | 4.921890 | 1.308517 | 3.613373 | 0.793362 |
| Chile | passed | 11 | 3 | 7.516266 | 0.849388 | 6.666878 | 0.841554 |
| Peru | passed | 63 | 3 | 3.746651 | 1.816437 | 1.930214 | 0.782598 |
| Cuba | passed | 16 | 3 | 6.765022 | 0.645592 | 6.119431 | 0.822312 |
| Jamaica | passed | 43 | 3 | 7.332512 | 0.255135 | 7.077377 | 0.818587 |
| largest planet | passed | 7 | 3 | 2.908974 | 0.987839 | 1.921135 | 0.750911 |
| smallest planet | passed | 43 | 3 | 3.049950 | 0.598846 | 2.451103 | 0.857418 |
| nearest star | passed | 1 | 3 | 3.254044 | 0.000000 | 3.254044 | 0.876811 |
| grass color | passed | 7 | 3 | 1.569336 | 1.248825 | 0.320511 | 0.690660 |
| opposite of hot | passed | 1 | 3 | 4.899635 | 0.159004 | 4.740631 | 0.718515 |
| number after nine | passed | 119 | 3 | 1.384033 | 0.115685 | 1.268349 | 0.898500 |
| month after January | failed | 205 | 3 | 0.808609 | 2.494255 | -1.685646 | 0.792029 |
| frozen water | passed | 2 | 3 | 2.452377 | 0.427292 | 2.025085 | 0.700371 |
| baby dog | passed | 5 | 3 | 1.568787 | 0.421387 | 1.147400 | 0.000000 |
| female parent | passed | 4 | 3 | 1.605240 | 0.417450 | 1.187790 | 0.000000 |
