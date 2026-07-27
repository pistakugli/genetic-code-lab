# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9968331465755671`
- max MAE: `0.010729194791711982`
- max worst error: `0.1369934384462007`
- min subspace principal cosine: `0.9999976562289214`
- total raw oracle calls: `174825`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
