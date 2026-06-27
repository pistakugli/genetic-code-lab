# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9966155130108486`
- max MAE: `0.0009254176411342425`
- max worst error: `0.010542759345426856`
- min subspace principal cosine: `0.9999966163799717`
- max unit count: `8`
- max atom count: `8`
- total raw oracle calls: `92049`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units0-17-123-456-789-1024-2048-4096: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-units0-17-123-456-789-1024-2048-4096-8atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
