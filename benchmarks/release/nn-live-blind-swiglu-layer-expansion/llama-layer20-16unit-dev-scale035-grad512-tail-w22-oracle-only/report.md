# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9926587098121563`
- max MAE: `0.018060447838133792`
- max worst error: `0.21002088537701083`
- min subspace principal cosine: `0.9996660748268255`
- total raw oracle calls: `138401`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
