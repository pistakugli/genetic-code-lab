# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9931553388745133`
- max MAE: `0.022255830700168896`
- max worst error: `0.26636602526134884`
- min subspace principal cosine: `0.9998458489586742`
- total raw oracle calls: `129953`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
