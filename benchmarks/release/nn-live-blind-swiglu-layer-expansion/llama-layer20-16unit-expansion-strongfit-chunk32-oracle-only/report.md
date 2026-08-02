# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `0`
- min R2: `0`
- max MAE: `0`
- max worst error: `0`
- min subspace principal cosine: `0`
- total raw oracle calls: `0`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false; maxUnitCount=null<16; Error: spawnSync ts-node ETIMEDOUT` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
