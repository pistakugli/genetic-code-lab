# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9996022410326608`
- max MAE: `0.0036226998892415745`
- max worst error: `0.03205623326481288`
- min subspace principal cosine: `0.9999935586279501`
- min post-hoc paired atom joint cosine: `0.9943894160746654`
- min paired atoms above threshold: `16`
- total raw oracle calls: `261057`
- access invariant: `true`

- llama layer 9 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
