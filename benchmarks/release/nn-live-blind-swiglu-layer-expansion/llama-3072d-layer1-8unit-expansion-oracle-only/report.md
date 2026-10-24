# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `8`
- min R2: `0.9999932523395926`
- max MAE: `0.0012646223141371737`
- max worst error: `0.010350319334377467`
- min subspace principal cosine: `0.9999791660156062`
- min post-hoc paired atom joint cosine: `0.9999995673189114`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.5383697471211562`
- minimum source pair correlation: `-0.032797557951469575`
- total raw oracle calls: `74065`
- access invariant: `true`

- llama layer 1 8-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-8unit-expansion-oracle-only/report.json)
