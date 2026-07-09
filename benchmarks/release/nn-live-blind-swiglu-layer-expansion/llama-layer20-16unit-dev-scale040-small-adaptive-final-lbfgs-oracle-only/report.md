# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9934559852828101`
- max MAE: `0.022160488105608583`
- max worst error: `0.26369126484887007`
- min subspace principal cosine: `0.9998458489586742`
- total raw oracle calls: `126545`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
