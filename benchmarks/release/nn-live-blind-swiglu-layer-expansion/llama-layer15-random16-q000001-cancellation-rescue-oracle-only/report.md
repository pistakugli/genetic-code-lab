# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9992081590257325`
- max MAE: `0.005812568238911098`
- max worst error: `0.05921484499462426`
- min subspace principal cosine: `0.9999457028618801`
- min post-hoc paired atom joint cosine: `0.9934124983784789`
- min paired atoms above threshold: `16`
- total raw oracle calls: `16384`
- access invariant: `true`

- llama layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-expansion-oracle-only/report.json)
