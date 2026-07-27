# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9968338818843243`
- max MAE: `0.010769079482525288`
- max worst error: `0.13665403934664205`
- min subspace principal cosine: `0.9999976562289214`
- total raw oracle calls: `175337`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
