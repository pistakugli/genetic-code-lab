# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `0`
- min R2: `0`
- max MAE: `0`
- max worst error: `0`
- min subspace principal cosine: `0`
- total raw oracle calls: `0`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false; maxUnitCount=null<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
