# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9962650545162728`
- max MAE: `0.0005119471166077025`
- max worst error: `0.010120884383530829`
- min subspace principal cosine: `0.9999999878109772`
- max unit count: `3`
- max atom count: `3`
- total raw oracle calls: `123936`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units7309-8811-12272: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units7309-8811-12272-3atom-oracle-only/report.json)
- random1-layer10-units5191-5754-11682: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer10-units5191-5754-11682-3atom-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
