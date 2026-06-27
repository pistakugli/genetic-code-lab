# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `12`
- min R2: `0.9952350522320941`
- max MAE: `0.0013654271559954592`
- max worst error: `0.013953604091311328`
- min subspace principal cosine: `0.9999973679313839`
- total raw oracle calls: `128697`
- access invariant: `true`

- mistral layer 20 12-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-12unit-expansion-oracle-only/report.json)
