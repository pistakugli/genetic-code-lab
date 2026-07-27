# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9991602047715517`
- max MAE: `0.006134974735926905`
- max worst error: `0.05285810130612889`
- min subspace principal cosine: `0.9999976562319123`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `16384`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer9-units958-1532-1647-1765-2755-3182-3856-4191-4371-4373-5442-7098-7145-7305-7679-7685: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-units958-1532-1647-1765-2755-3182-3856-4191-4371-4373-5442-7098-7145-7305-7679-7685-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
