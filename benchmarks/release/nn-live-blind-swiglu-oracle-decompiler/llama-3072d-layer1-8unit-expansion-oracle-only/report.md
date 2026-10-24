# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9999932523395926`
- max MAE: `0.0012646223141371737`
- max worst error: `0.010350319334377467`
- min subspace principal cosine: `0.9999791660156062`
- min post-hoc paired atom joint cosine: `0.9999995673189114`
- min paired atoms above threshold: `null`
- min source cancellation fraction: `0.5383697471211562`
- minimum source pair correlation: `-0.032797557951469575`
- max unit count: `8`
- max atom count: `8`
- total raw oracle calls: `74065`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units0-17-123-456-789-1024-2048-4096: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
