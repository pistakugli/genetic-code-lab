# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9963079611055821`
- max MAE: `0.011942695445661719`
- max worst error: `0.1345624769905193`
- min subspace principal cosine: `0.9999857637238745`
- total raw oracle calls: `24576`
- access invariant: `true`

- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
