# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9929982752205192`
- max MAE: `0.00002326627317537317`
- max worst error: `0.00015573703561534913`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.2773761459982943`
- min paired atoms above threshold: `2`
- min source cancellation fraction: `0.6789421688633468`
- minimum source pair correlation: `-0.034901440311763995`
- total raw oracle calls: `43080`
- access invariant: `true`

- mistral layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-16unit-output2048-q000001-expansion-oracle-only/report.json)
