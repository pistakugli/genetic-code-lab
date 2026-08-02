# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/2`
- max expanded unit count: `16`
- min R2: `0.9981781313105312`
- max MAE: `0.008086910496668931`
- max worst error: `0.10551562347990848`
- min subspace principal cosine: `0.9999952999253275`
- min post-hoc paired atom joint cosine: `0.21416161036019987`
- min paired atoms above threshold: `14`
- total raw oracle calls: `497538`
- access invariant: `true`

- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
- llama layer 20 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-16unit-q000001-cubic2048h512s0-35-0-25w10-expansion-oracle-only/report.json)
