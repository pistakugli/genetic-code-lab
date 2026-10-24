# Live GPT-2 Fact Circuit Finder

This benchmark uses a real GPT-2 model, real forward-pass hooks, automatic layer/neuron scanning, graph construction, joint graph ablations, and graph-level negative controls. It is not synthetic and does not fit a surrogate.

- Pass: `True`
- Valid cases: `5`
- Strict case definition: `graphGate.pass && hardeningGate.pass`
- Passed cases: `5`
- Graph-passed cases: `5`
- Hardening-passed cases: `5`
- Pass rate: `1.0`
- Median graph joint target-logit drop: `0.28775787353515625`
- Median graph control p95 target-logit drop: `0.04071731567382811`
- Median graph lift over control p95: `2.6112537240720277`
- Mean selected graph nodes: `1.6`
- Paraphrase validation pass rate: `1.0`
- Wrong-answer specificity pass rate: `1.0`
- Negative-fact specificity pass rate: `1.0`
- Unrelated-logit preservation pass rate: `0.975`
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
| France | Paris | passed | 5 | 1 | 0.106323 | 0.040717 | 0.065606 | True | 9 | 1828 |
| Germany | Berlin | passed | 2 | 2 | 0.287758 | 0.014363 | 0.273394 | True | 9 | 1409 |
| Italy | Rome | passed | 1 | 2 | 0.322838 | 0.163510 | 0.159328 | True | 9 | 1828 |
| Spain | Madrid | passed | 1 | 1 | 0.145294 | 0.016478 | 0.128816 | True | 9 | 2416 |
| Portugal | Lisbon | passed | 2 | 2 | 0.593819 | 0.409357 | 0.184461 | True | 9 | 1828 |
