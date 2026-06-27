# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9973170168506432`
- max MAE: `0.0012842742350251879`
- max worst error: `0.013479211039965633`
- min subspace principal cosine: `0.9999979444870536`
- total raw oracle calls: `188193`
- access invariant: `true`

- mistral layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-16unit-expansion-oracle-only/report.json)
