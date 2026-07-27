# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9996478192267217`
- max MAE: `0.0000058048029765860524`
- max worst error: `0.00003073354338659442`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9991902414824323`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.684963528861127`
- minimum source pair correlation: `-0.04406152273669756`
- total raw oracle calls: `177244`
- access invariant: `true`

- mistral layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
