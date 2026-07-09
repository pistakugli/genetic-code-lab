# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9960753383385221`
- max MAE: `0.01302306518755879`
- max worst error: `0.12042929999008084`
- min subspace principal cosine: `0.9996660748268255`
- total raw oracle calls: `129953`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
