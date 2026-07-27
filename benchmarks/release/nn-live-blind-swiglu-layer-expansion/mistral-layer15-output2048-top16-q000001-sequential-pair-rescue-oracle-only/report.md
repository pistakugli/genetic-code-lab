# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9976440956596437`
- max MAE: `0.000014050645102733393`
- max worst error: `0.00011631635935048469`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.08378074120982693`
- min paired atoms above threshold: `8`
- min source cancellation fraction: `0.6935123485059438`
- minimum source pair correlation: `-0.04442603547600881`
- total raw oracle calls: `1150592`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s2-0-1-4w10-expansion-oracle-only/report.json)
