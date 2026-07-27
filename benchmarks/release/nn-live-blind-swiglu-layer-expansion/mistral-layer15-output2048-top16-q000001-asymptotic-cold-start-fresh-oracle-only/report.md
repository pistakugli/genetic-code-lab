# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9995787659698843`
- max MAE: `0.00000616945597616291`
- max worst error: `0.000032305729901930614`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9972788963034496`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6887747348810841`
- minimum source pair correlation: `-0.044888687721043014`
- total raw oracle calls: `1279238`
- access invariant: `true`

- mistral layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s0-35-0-25w0-expansion-oracle-only/report.json)
