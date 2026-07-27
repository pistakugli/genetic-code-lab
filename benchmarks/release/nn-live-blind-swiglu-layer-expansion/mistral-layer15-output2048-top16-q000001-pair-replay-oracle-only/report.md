# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9985167459524236`
- max MAE: `0.000011494504746791434`
- max worst error: `0.00007521856025931892`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9767338526662039`
- min paired atoms above threshold: `12`
- min source cancellation fraction: `0.6935123485059438`
- minimum source pair correlation: `-0.04442603547600881`
- total raw oracle calls: `16384`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
