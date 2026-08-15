# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `32`
- min R2: `0.9882781253283933`
- max MAE: `0.03027649942396946`
- max worst error: `0.2462262278698874`
- min subspace principal cosine: `0.9997704576111829`
- min post-hoc paired atom joint cosine: `0.18922121075296774`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.7727076260472346`
- minimum source pair correlation: `-0.04506938912151085`
- total raw oracle calls: `252481`
- access invariant: `true`

- llama layer 1 32-unit: `fail sweep summary pass=false; sweep exit 1` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-32unit-expansion-oracle-only/report.json)
