# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9970241602135861`
- max MAE: `0.000014913854753039543`
- max worst error: `0.00009852395145545272`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.9842222685942671`
- min paired atoms above threshold: `13`
- min source cancellation fraction: `0.6811694685734337`
- minimum source pair correlation: `-0.035935491081521574`
- total raw oracle calls: `16384`
- access invariant: `true`

- mistral layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-16unit-output2048-q000001-expansion-oracle-only/report.json)
