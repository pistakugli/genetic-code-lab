# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `10`
- min R2: `0.9950616546567232`
- max MAE: `0.0012622919284927562`
- max worst error: `0.019690672359748522`
- min subspace principal cosine: `0.9999950507602267`
- total raw oracle calls: `108885`
- access invariant: `true`

- mistral layer 20 10-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-10unit-expansion-oracle-only/report.json)
