# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9968695092259744`
- max MAE: `0.00001604656212518726`
- max worst error: `0.00012727892009867074`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.06030989956308541`
- min paired atoms above threshold: `5`
- min source cancellation fraction: `0.6935123485059438`
- minimum source pair correlation: `-0.04442603547600881`
- total raw oracle calls: `1150592`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s2-0-1-4w10-expansion-oracle-only/report.json)
