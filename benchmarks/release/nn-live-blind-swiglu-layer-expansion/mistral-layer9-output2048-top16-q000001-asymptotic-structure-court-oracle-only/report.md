# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9996824964424941`
- max MAE: `0.000005019407967445103`
- max worst error: `0.00002731915870133052`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.9989749892979034`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6811694685734337`
- minimum source pair correlation: `-0.035935491081521574`
- total raw oracle calls: `107696`
- access invariant: `true`

- mistral layer 9 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-16unit-output2048-q000001-expansion-oracle-only/report.json)
