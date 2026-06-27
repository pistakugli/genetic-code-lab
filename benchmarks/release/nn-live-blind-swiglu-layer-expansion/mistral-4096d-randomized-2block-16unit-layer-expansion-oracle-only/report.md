# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `2/2`
- max expanded unit count: `16`
- min R2: `0.9968086229936512`
- max MAE: `0.0013585229495551043`
- max worst error: `0.018289335513027943`
- min subspace principal cosine: `0.999989791877386`
- total raw oracle calls: `376386`
- access invariant: `true`

- mistral layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-16unit-expansion-oracle-only/report.json)
- mistral layer 11 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer11-16unit-expansion-oracle-only/report.json)
