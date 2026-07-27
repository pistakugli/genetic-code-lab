# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9984285050805913`
- max MAE: `0.007535263448469097`
- max worst error: `0.12282348198305465`
- min subspace principal cosine: `0.9999457028618801`
- min post-hoc paired atom joint cosine: `0.18429600112992964`
- min paired atoms above threshold: `14`
- total raw oracle calls: `142241`
- access invariant: `true`

- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-expansion-oracle-only/report.json)
