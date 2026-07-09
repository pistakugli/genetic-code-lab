# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9896260610578036`
- max MAE: `0.07725267840423847`
- max worst error: `1.1800129370025223`
- min subspace principal cosine: `0.9999469628979306`
- total raw oracle calls: `154401`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
