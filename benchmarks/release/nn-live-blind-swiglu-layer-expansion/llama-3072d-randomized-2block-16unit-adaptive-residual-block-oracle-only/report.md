# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.


- pass: `false`
- cases: `1/2`
- max expanded unit count: `16`
- min R2: `0.9969739664843292`
- max MAE: `0.010487457146681946`
- max worst error: `0.13462588510413603`
- min subspace principal cosine: `0.9999482868797673`
- total raw oracle calls: `288578`
- access invariant: `true`

- llama layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
