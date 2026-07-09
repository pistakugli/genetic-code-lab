# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9798919580812425`
- max MAE: `0.11162318249870311`
- max worst error: `0.7323828590528756`
- min subspace principal cosine: `0.9999469628979306`
- total raw oracle calls: `110753`
- access invariant: `true`

- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
