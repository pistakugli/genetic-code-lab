# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9999922485589756`
- max MAE: `0.0006021497608951118`
- max worst error: `0.003997934982926221`
- min subspace principal cosine: `0.9999976562319123`
- min post-hoc paired atom joint cosine: `0.9999711941934621`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6728898818172195`
- minimum source pair correlation: `-0.047097926804822`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `109808`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer9-units958-1532-1647-1765-2755-3182-3856-4191-4371-4373-5442-7098-7145-7305-7679-7685: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-units958-1532-1647-1765-2755-3182-3856-4191-4371-4373-5442-7098-7145-7305-7679-7685-16atom-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
