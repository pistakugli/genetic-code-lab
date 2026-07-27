# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.996629323676471`
- max MAE: `0.010909926811465173`
- max worst error: `0.14006396585681552`
- min subspace principal cosine: `0.9999976562289214`
- total raw oracle calls: `24576`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
