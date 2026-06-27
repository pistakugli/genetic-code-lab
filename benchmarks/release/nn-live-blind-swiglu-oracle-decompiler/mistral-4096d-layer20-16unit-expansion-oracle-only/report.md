# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9969881717700748`
- max MAE: `0.0013780126849811608`
- max worst error: `0.009548484562662135`
- min subspace principal cosine: `0.9999979444870536`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `188193`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units0-17-123-456-789-1024-2048-4096-6144-6331-8191-10000-11772-12288-13200-14000: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-2048-4096-6144-6331-8191-10000-11772-12288-13200-14000-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
