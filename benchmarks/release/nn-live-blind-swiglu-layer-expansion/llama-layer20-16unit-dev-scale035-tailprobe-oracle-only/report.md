# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9957814615966589`
- max MAE: `0.01323516179574758`
- max worst error: `0.145617724777229`
- min subspace principal cosine: `0.9996660748268255`
- total raw oracle calls: `134049`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
