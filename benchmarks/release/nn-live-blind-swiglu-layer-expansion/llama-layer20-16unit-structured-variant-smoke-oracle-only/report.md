# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9812241010105195`
- max MAE: `0.10643979589818175`
- max worst error: `1.1600932967143156`
- min subspace principal cosine: `0.9999469628979306`
- total raw oracle calls: `110753`
- access invariant: `true`

- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
