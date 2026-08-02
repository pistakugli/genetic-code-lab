# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9932701674227643`
- max MAE: `0.02666722797022555`
- max worst error: `0.3901571989495969`
- min subspace principal cosine: `0.99991022687528`
- total raw oracle calls: `154401`
- access invariant: `true`

- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-expansion-oracle-only/report.json)
