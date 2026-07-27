# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9963010779768818`
- max MAE: `0.010949420188610223`
- max worst error: `0.2311969763358116`
- min subspace principal cosine: `0.9999976562289214`
- total raw oracle calls: `178921`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-expansion-oracle-only/report.json)
