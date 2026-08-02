# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.997435675278273`
- max MAE: `0.000013177504558141229`
- max worst error: `0.00011375767707405581`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.04669053643787195`
- min paired atoms above threshold: `3`
- min source cancellation fraction: `0.6863059310402382`
- minimum source pair correlation: `-0.037702389114347215`
- total raw oracle calls: `20480`
- access invariant: `true`

- mistral layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-16unit-output2048-q000001-expansion-oracle-only/report.json)
