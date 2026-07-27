# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9994162688892091`
- max MAE: `0.000007229904540356296`
- max worst error: `0.00005486008018525529`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9671755291538968`
- min paired atoms above threshold: `14`
- min source cancellation fraction: `0.684963528861127`
- minimum source pair correlation: `-0.04406152273669756`
- total raw oracle calls: `101538`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
