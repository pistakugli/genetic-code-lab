# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9976154835340468`
- max MAE: `0.00890032031117774`
- max worst error: `0.07041405409521567`
- min subspace principal cosine: `0.9909685718257543`
- min post-hoc paired atom joint cosine: `0.9887688240398791`
- min paired atoms above threshold: `15`
- total raw oracle calls: `144289`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-q000001-expansion-oracle-only/report.json)
