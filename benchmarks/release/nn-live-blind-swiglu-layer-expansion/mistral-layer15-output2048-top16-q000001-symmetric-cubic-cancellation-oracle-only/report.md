# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9125318082557777`
- max MAE: `0.00008453385653242291`
- max worst error: `0.0007668425532936642`
- min subspace principal cosine: `0.6622584293375741`
- min post-hoc paired atom joint cosine: `-0.047469815838226376`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6899803389557442`
- minimum source pair correlation: `-0.05116038563544401`
- total raw oracle calls: `634496`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
