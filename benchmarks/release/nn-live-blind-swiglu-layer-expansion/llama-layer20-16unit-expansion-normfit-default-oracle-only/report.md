# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9896630810736502`
- max MAE: `0.07586709374805956`
- max worst error: `1.2603260618554648`
- min subspace principal cosine: `0.9999469628979306`
- total raw oracle calls: `156450`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
