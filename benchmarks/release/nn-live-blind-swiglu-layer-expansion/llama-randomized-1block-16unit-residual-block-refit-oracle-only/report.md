# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9973163417192888`
- max MAE: `0.00916608904588034`
- max worst error: `0.1298692543958965`
- min subspace principal cosine: `0.9999482868797673`
- total raw oracle calls: `142241`
- access invariant: `true`

- llama layer 1 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
