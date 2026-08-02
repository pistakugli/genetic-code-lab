# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.20932557856357958`
- max MAE: `0.00027371890612757277`
- max worst error: `0.001462911018102499`
- min subspace principal cosine: `0.0004234821189857783`
- min post-hoc paired atom joint cosine: `0.046644515260047634`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6793548639541604`
- minimum source pair correlation: `-0.07267213285078375`
- total raw oracle calls: `317783`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
