# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9950616546567232`
- max MAE: `0.0012622919284927562`
- max worst error: `0.019690672359748522`
- min subspace principal cosine: `0.9999950507602267`
- max unit count: `10`
- max atom count: `10`
- total raw oracle calls: `108885`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units0-17-123-456-789-1024-2048-4096-6144-8191: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-2048-4096-6144-8191-10atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
