# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9995440406106985`
- max MAE: `0.004315024051586014`
- max worst error: `0.04367946287594893`
- min subspace principal cosine: `0.9999952999253275`
- min post-hoc paired atom joint cosine: `0.992509414607041`
- min paired atoms above threshold: `16`
- total raw oracle calls: `261057`
- access invariant: `true`

- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
