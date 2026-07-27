# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9967843115773182`
- max MAE: `0.010635533625240817`
- max worst error: `0.13571022512826825`
- min subspace principal cosine: `0.9999976562319123`
- total raw oracle calls: `16384`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
