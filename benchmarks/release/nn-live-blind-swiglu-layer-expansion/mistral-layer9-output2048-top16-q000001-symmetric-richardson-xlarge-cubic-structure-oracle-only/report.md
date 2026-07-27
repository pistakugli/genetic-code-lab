# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9927064132277419`
- max MAE: `0.00002294618776565714`
- max worst error: `0.00015101317187022592`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.2773761459982943`
- min paired atoms above threshold: `2`
- min source cancellation fraction: `0.6860616086677447`
- minimum source pair correlation: `-0.03201511955079522`
- total raw oracle calls: `1674880`
- access invariant: `true`

- mistral layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-16unit-output2048-q000001-cubic2048h512s2-0-1-4w10-expansion-oracle-only/report.json)
