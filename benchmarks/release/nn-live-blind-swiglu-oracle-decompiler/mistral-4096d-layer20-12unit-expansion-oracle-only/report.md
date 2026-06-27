# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9952350522320941`
- max MAE: `0.0013654271559954592`
- max worst error: `0.013953604091311328`
- min subspace principal cosine: `0.9999973679313839`
- max unit count: `12`
- max atom count: `12`
- total raw oracle calls: `128697`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units0-17-123-456-789-1024-2048-4096-6144-8191-10000-12288: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-2048-4096-6144-8191-10000-12288-12atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
