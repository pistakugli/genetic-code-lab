# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9898074852523271`
- max MAE: `0.07288726181226854`
- max worst error: `1.2617263536772096`
- min subspace principal cosine: `0.9999994542253061`
- total raw oracle calls: `252737`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
