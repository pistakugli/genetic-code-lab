# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `2/2`
- max expanded unit count: `16`
- min R2: `0.9993531509315209`
- max MAE: `0.005235521283167137`
- max worst error: `0.04507800353440916`
- min subspace principal cosine: `0.9999482868797673`
- min post-hoc paired atom joint cosine: `0.9942495918505425`
- min paired atoms above threshold: `16`
- total raw oracle calls: `284482`
- access invariant: `true`

- llama layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
