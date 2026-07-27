# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/2`
- max expanded unit count: `16`
- min R2: `0.9944436996014887`
- max MAE: `0.0146286895098221`
- max worst error: `0.22596283452325888`
- min subspace principal cosine: `0.9997919983764019`
- total raw oracle calls: `259906`
- access invariant: `true`

- llama layer 1 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
