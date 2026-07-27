# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9991602047715517`
- max MAE: `0.006134974735926905`
- max worst error: `0.05285810130612889`
- min subspace principal cosine: `0.9999976562319123`
- total raw oracle calls: `16384`
- access invariant: `true`

- llama layer 9 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
