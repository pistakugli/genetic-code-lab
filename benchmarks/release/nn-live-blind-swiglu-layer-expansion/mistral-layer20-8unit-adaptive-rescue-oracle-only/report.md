# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `8`
- min R2: `0.9966155130108486`
- max MAE: `0.0009254176411342425`
- max worst error: `0.010542759345426856`
- min subspace principal cosine: `0.9999966163799717`
- total raw oracle calls: `92049`
- access invariant: `true`

- mistral layer 20 8-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-8unit-expansion-oracle-only/report.json)
