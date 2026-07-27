# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9792319337751766`
- max MAE: `0.000040864780470485434`
- max worst error: `0.0005210991674243392`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `-0.061545973396517926`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.685456002800046`
- minimum source pair correlation: `-0.04716293159517883`
- total raw oracle calls: `1158784`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
