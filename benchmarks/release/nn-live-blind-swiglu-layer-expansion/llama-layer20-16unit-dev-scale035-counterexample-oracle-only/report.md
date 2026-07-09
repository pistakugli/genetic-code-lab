# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9943014449800781`
- max MAE: `0.01397615207380914`
- max worst error: `0.3419731445977565`
- min subspace principal cosine: `0.9996660748268255`
- total raw oracle calls: `151201`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
