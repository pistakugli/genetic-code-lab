# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9788085506276295`
- max MAE: `0.1087335799466309`
- max worst error: `1.2748208999038306`
- min subspace principal cosine: `0.9999469628979306`
- total raw oracle calls: `112289`
- access invariant: `true`

- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
