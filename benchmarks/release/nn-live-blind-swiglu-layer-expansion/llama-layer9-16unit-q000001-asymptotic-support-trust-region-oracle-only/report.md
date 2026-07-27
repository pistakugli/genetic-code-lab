# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.9999922485589756`
- max MAE: `0.0006021497608951118`
- max worst error: `0.003997934982926221`
- min subspace principal cosine: `0.9999976562319123`
- min post-hoc paired atom joint cosine: `0.9999711941934621`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6728898818172195`
- minimum source pair correlation: `-0.047097926804822`
- total raw oracle calls: `109808`
- access invariant: `true`

- llama layer 9 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-16unit-q000001-expansion-oracle-only/report.json)
