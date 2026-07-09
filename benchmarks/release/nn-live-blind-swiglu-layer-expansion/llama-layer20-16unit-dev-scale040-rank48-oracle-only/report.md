# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9934450028936462`
- max MAE: `0.022673777156074083`
- max worst error: `0.2661162372553245`
- min subspace principal cosine: `0.9999996438427642`
- total raw oracle calls: `234385`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
