# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `4/4`
- min R2: `0.9999845855712838`
- max MAE: `0.000015160815735029332`
- max worst error: `0.00017546677141010852`
- min subspace principal cosine: `0.9999997892743723`
- total raw oracle calls: `73984`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer8-unit123: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer8-unit123-oracle-only/report.json)
- layer16-unit2048: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer16-unit2048-oracle-only/report.json)
- layer20-unit0: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer20-unit0-oracle-only/report.json)
- layer24-unit17: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer24-unit17-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
