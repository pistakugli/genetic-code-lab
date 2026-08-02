# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `1/3`
- max expanded unit count: `16`
- min R2: `0.9936416201869898`
- max MAE: `0.015930591937818074`
- max worst error: `0.20086170989863827`
- min subspace principal cosine: `0.9973485731809377`
- min post-hoc paired atom joint cosine: `0.17081204133338923`
- min paired atoms above threshold: `11`
- total raw oracle calls: `439011`
- access invariant: `true`

- llama layer 9 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-q000001-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-expansion-oracle-only/report.json)
- llama layer 20 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-q000001-expansion-oracle-only/report.json)
