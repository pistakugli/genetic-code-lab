# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9824053671606762`
- max MAE: `0.00003513618303735926`
- max worst error: `0.0003476888040364096`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.006188913816324287`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6811694685734337`
- minimum source pair correlation: `-0.035935491081521574`
- total raw oracle calls: `1674880`
- access invariant: `true`

- mistral layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-16unit-output2048-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
