# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/2`
- max expanded unit count: `16`
- min R2: `0.9739725952612162`
- max MAE: `0.032096388717845`
- max worst error: `0.30986240705975504`
- min subspace principal cosine: `0.9999530973154566`
- min post-hoc paired atom joint cosine: `0.06854566068159593`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6763121834163919`
- minimum source pair correlation: `-0.03947501364017383`
- total raw oracle calls: `300866`
- access invariant: `true`

- llama layer 1 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
