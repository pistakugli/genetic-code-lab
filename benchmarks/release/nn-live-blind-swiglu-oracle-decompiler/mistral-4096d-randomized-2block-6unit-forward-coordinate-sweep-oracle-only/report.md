# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9952220102880769`
- max MAE: `0.0009253719069818783`
- max worst error: `0.012323574241066268`
- min subspace principal cosine: `0.9999991210977408`
- max unit count: `6`
- max atom count: `6`
- total raw oracle calls: `124506`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-5191-7309-8811-11682-12272: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-5191-7309-8811-11682-12272-6atom-forward-coordinate-oracle-only/report.json)
- random1-layer12-units279-3125-8561-9962-11673-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer12-units279-3125-8561-9962-11673-13025-6atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
