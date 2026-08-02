# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `true`
- cases: `2/2`
- max expanded unit count: `16`
- min R2: `0.9992051540834496`
- max MAE: `0.005833540487146457`
- max worst error: `0.06428347992920569`
- min subspace principal cosine: `0.9997557479909298`
- min post-hoc paired atom joint cosine: `0.9937228873084201`
- min paired atoms above threshold: `16`
- total raw oracle calls: `284482`
- access invariant: `true`

- llama layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-q000001-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-expansion-oracle-only/report.json)
