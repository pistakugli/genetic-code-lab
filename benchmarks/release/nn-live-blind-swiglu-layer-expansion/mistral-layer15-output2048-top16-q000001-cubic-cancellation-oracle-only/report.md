# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.17767096458876264`
- max MAE: `0.00027325402994042477`
- max worst error: `0.0013373960207129336`
- min subspace principal cosine: `0.002014020544312016`
- min post-hoc paired atom joint cosine: `-0.013760827974344282`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6921537929392196`
- minimum source pair correlation: `-0.06703731421220446`
- total raw oracle calls: `204705`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
