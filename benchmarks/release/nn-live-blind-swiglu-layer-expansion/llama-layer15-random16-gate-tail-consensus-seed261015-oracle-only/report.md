# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9991777532015129`
- max MAE: `0.005725646988955818`
- max worst error: `0.04819578588409401`
- min subspace principal cosine: `0.9999857637238745`
- total raw oracle calls: `20480`
- access invariant: `true`

- llama layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
