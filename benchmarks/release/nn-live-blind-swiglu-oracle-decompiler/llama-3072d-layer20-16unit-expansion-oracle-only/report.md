# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.996028807883924`
- max MAE: `0.013147489844684133`
- max worst error: `0.1197387336033246`
- min subspace principal cosine: `0.9996660748268255`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `129953`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units0-17-123-456-789-912-1024-1757-2048-2235-3580-4096-4100-5266-6144-8191: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-912-1024-1757-2048-2235-3580-4096-4100-5266-6144-8191-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
