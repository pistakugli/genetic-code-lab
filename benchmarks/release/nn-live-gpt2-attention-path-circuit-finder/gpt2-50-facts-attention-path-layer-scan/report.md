# Live GPT-2 Attention Path Circuit Finder

This benchmark uses a real GPT-2 model, real attention-path hooks, automatic layer/head scanning, joint attention-head ablations, and attention-head negative controls. It is not synthetic and does not fit a surrogate.

## Summary

- Pass: `True`
- Valid cases: `50`
- Passed cases: `50`
- Pass rate: `1.0`
- Median attention graph target-logit drop: `1.7271881103515625`
- Median attention control p95 target-logit drop: `0.448638916015625`
- Median attention graph lift over control p95: `3.7378692395502666`
- Mean selected attention heads: `1.04`
- Unique selected attention heads: `6`
- Median wrong-answer collateral ratio: `0.7162562846461261`

## Claim Boundary

Benchmark-scoped evidence that the real GPT-2 live pipeline can scan attention heads, build token/head/target path graph evidence, and validate selected attention paths through causal head ablation above attention-head controls across a factual prompt set.

This is not answer-specific factual identity. Attention heads can carry broad task or token-class effects, so wrong-answer collateral is reported but not gated in this Stage-K path graph. This is also not full attention-circuit recovery across all model components, not attention-to-MLP composition recovery, not full GPT-2 FFN extraction, not arbitrary LLM understanding, and not output-only original-neuron identity.

## Cases

| subject | status | target rank | heads | graph drop | control p95 | margin | wrong ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| France | passed | 5 | 1 | 1.727188 | 1.519371 | 0.207817 | 0.756036 |
| Germany | passed | 2 | 1 | 1.214882 | 0.609848 | 0.605034 | 0.726282 |
| Italy | passed | 1 | 1 | 3.036995 | 0.894081 | 2.142914 | 0.814357 |
| Spain | passed | 1 | 1 | 3.170837 | 0.407089 | 2.763748 | 0.759393 |
| Portugal | passed | 2 | 1 | 1.790855 | 0.449188 | 1.341667 | 0.711926 |
| Greece | passed | 1 | 1 | 2.116165 | 1.041260 | 1.074905 | 0.855749 |
| Egypt | passed | 2 | 1 | 2.155434 | 0.518204 | 1.637230 | 0.750659 |
| Japan | passed | 2 | 1 | 0.607285 | 0.413971 | 0.193314 | 0.523531 |
| China | passed | 10 | 1 | 2.552483 | 0.590065 | 1.962418 | 0.839553 |
| Russia | passed | 7 | 1 | 1.690453 | 0.690620 | 0.999832 | 0.835881 |
| Canada | passed | 23 | 2 | 1.641754 | 0.808670 | 0.833084 | 0.195517 |
| Australia | passed | 17 | 1 | 1.145401 | 0.421555 | 0.723846 | 0.000000 |
| Sweden | passed | 13 | 1 | 1.823677 | 0.487892 | 1.335785 | 0.728732 |
| Norway | passed | 19 | 2 | 3.163925 | 1.675468 | 1.488457 | 0.542252 |
| Denmark | passed | 11 | 1 | 1.324722 | 0.345230 | 0.979492 | 0.396368 |
| Finland | passed | 7 | 1 | 1.284286 | 0.294991 | 0.989296 | 0.461921 |
| Poland | passed | 1 | 1 | 1.320648 | 0.368179 | 0.952469 | 0.460485 |
| Hungary | passed | 2 | 1 | 1.835587 | 0.827789 | 1.007797 | 0.772550 |
| Austria | passed | 13 | 1 | 1.741425 | 0.556595 | 1.184830 | 0.704826 |
| Belgium | passed | 11 | 1 | 1.243149 | 0.527481 | 0.715668 | 0.829172 |
| Netherlands | passed | 3 | 1 | 1.497498 | 0.344887 | 1.152611 | 0.762615 |
| Ireland | passed | 3 | 1 | 1.251823 | 0.351135 | 0.900688 | 0.133978 |
| Turkey | passed | 12 | 1 | 3.142685 | 0.522316 | 2.620369 | 0.778633 |
| Iran | passed | 4 | 1 | 1.408066 | 0.257103 | 1.150963 | 0.747142 |
| Iraq | passed | 10 | 1 | 2.111160 | 0.007790 | 2.103371 | 0.714467 |
| Syria | passed | 18 | 1 | 1.384361 | 0.470451 | 0.913910 | 0.731233 |
| Israel | passed | 8 | 1 | 1.484314 | 0.402443 | 1.081871 | 0.660137 |
| Saudi Arabia | passed | 10 | 1 | 2.547142 | 0.448639 | 2.098503 | 0.606897 |
| Pakistan | passed | 3 | 1 | 1.680931 | 0.394897 | 1.286034 | 0.358982 |
| Thailand | passed | 8 | 1 | 1.394485 | 0.267975 | 1.126511 | 0.307701 |
| Indonesia | passed | 10 | 1 | 2.070213 | 0.316200 | 1.754013 | 0.669062 |
| Philippines | passed | 3 | 1 | 1.450729 | 0.370132 | 1.080597 | 0.860121 |
| South Korea | passed | 6 | 1 | 0.703369 | 0.285889 | 0.417480 | 0.376963 |
| Czech Republic | passed | 5 | 1 | 1.949554 | 0.509163 | 1.440392 | 0.772709 |
| Tunisia | passed | 23 | 1 | 2.308067 | 0.915092 | 1.392975 | 0.716256 |
| Chile | passed | 11 | 1 | 3.257645 | 0.749008 | 2.508636 | 0.789684 |
| Peru | passed | 63 | 1 | 2.072281 | 1.846596 | 0.225685 | 0.781734 |
| Cuba | passed | 16 | 1 | 2.322624 | 0.156609 | 2.166016 | 0.717043 |
| Jamaica | passed | 43 | 1 | 2.963318 | 0.288818 | 2.674500 | 0.704864 |
| largest planet | passed | 7 | 1 | 2.062889 | 0.149666 | 1.913223 | 0.784154 |
| smallest planet | passed | 43 | 1 | 2.228348 | 0.643181 | 1.585167 | 0.916377 |
| nearest star | passed | 1 | 1 | 1.904694 | 0.032768 | 1.871925 | 0.851449 |
| grass color | passed | 7 | 1 | 1.141380 | 0.575142 | 0.566238 | 0.790459 |
| opposite of hot | passed | 1 | 1 | 1.912231 | 0.000000 | 1.912231 | 0.437811 |
| frozen water | passed | 2 | 1 | 2.266708 | 0.555344 | 1.711365 | 0.831448 |
| baby dog | passed | 5 | 1 | 1.423622 | 0.561401 | 0.862221 | 0.189462 |
| female parent | passed | 4 | 1 | 0.618553 | 0.000000 | 0.618553 | 0.000000 |
| male parent | passed | 4 | 1 | 0.499039 | 0.198387 | 0.300652 | 0.000000 |
| medical profession | passed | 16 | 1 | 1.176933 | 0.210464 | 0.966469 | 0.261962 |
| school profession | passed | 1 | 1 | 1.153694 | 0.543480 | 0.610214 | 0.418419 |
