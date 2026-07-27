# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `1/2`
- max expanded unit count: `16`
- min R2: `0.9984546684427439`
- max MAE: `0.007338289835113133`
- max worst error: `0.10557907656586657`
- min subspace principal cosine: `0.9997557479909298`
- min post-hoc paired atom joint cosine: `0.18429600112992964`
- min paired atoms above threshold: `14`
- total raw oracle calls: `288578`
- access invariant: `true`

- llama layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-q000001-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-q000001-expansion-oracle-only/report.json)
