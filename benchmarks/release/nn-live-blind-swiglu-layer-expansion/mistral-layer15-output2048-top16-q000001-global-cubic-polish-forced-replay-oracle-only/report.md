# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9993853502992724`
- max MAE: `0.000007423685093645544`
- max worst error: `0.000039696800168462774`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9977841900415598`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6935123485059438`
- minimum source pair correlation: `-0.04442603547600881`
- total raw oracle calls: `16384`
- access invariant: `true`

- mistral layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
