# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9952415178321019`
- max MAE: `0.010381675027569176`
- max worst error: `0.11170013115197852`
- min subspace principal cosine: `0.9991382295919516`
- total raw oracle calls: `129953`
- access invariant: `true`

- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
