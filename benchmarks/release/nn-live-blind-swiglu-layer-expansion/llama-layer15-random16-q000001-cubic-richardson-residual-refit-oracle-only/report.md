# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9996515769035299`
- max MAE: `0.0038307358859897135`
- max worst error: `0.031238018542964846`
- min subspace principal cosine: `0.9999962539759294`
- min post-hoc paired atom joint cosine: `0.9929854299572484`
- min paired atoms above threshold: `16`
- total raw oracle calls: `261057`
- access invariant: `true`

- llama layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
