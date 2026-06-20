# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9952423306790068`
- max MAE: `0.0009230608551877102`
- max worst error: `0.012063083889497478`
- min subspace principal cosine: `0.9999999518366974`
- max unit count: `6`
- max atom count: `6`
- total raw oracle calls: `230976`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-5191-7309-8811-11682-12272: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-5191-7309-8811-11682-12272-6atom-oracle-only/report.json)
- random1-layer12-units279-3125-8561-9962-11673-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer12-units279-3125-8561-9962-11673-13025-6atom-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
