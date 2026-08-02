# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `1/2`
- max expanded unit count: `16`
- min R2: `0.9986197880720584`
- max MAE: `0.007325520441823712`
- max worst error: `0.11110972379914336`
- min subspace principal cosine: `0.9999530973154566`
- min post-hoc paired atom joint cosine: `0.23861138806479312`
- min paired atoms above threshold: `14`
- min source cancellation fraction: `0.6810587415472`
- minimum source pair correlation: `-0.04464800348325854`
- total raw oracle calls: `288578`
- access invariant: `true`

- llama layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
