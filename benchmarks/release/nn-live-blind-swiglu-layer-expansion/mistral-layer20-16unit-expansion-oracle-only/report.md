# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9969881717700748`
- max MAE: `0.0013780126849811608`
- max worst error: `0.009548484562662135`
- min subspace principal cosine: `0.9999979444870536`
- total raw oracle calls: `188193`
- access invariant: `true`

- mistral layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-16unit-expansion-oracle-only/report.json)
