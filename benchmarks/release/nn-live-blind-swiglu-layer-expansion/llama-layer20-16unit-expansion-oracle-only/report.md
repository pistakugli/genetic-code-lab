# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.996028807883924`
- max MAE: `0.013147489844684133`
- max worst error: `0.1197387336033246`
- min subspace principal cosine: `0.9996660748268255`
- total raw oracle calls: `129953`
- access invariant: `true`

- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
