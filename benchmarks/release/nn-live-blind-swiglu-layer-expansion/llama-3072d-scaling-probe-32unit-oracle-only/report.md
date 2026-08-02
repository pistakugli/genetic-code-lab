# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `0`
- min R2: `0`
- max MAE: `0`
- max worst error: `0`
- min subspace principal cosine: `0`
- min post-hoc paired atom joint cosine: `Infinity`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `Infinity`
- minimum source pair correlation: `Infinity`
- total raw oracle calls: `0`
- access invariant: `true`

- llama layer 1 32-unit: `fail sweep summary pass=false; maxUnitCount=null<32; sweep exit 1` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-32unit-expansion-oracle-only/report.json)
