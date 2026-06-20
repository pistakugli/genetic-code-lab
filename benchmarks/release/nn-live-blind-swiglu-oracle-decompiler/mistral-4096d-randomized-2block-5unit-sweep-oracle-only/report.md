# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9951436177161573`
- max MAE: `0.0008022104114991162`
- max worst error: `0.010051115148071464`
- min subspace principal cosine: `0.9999997906113538`
- max unit count: `5`
- max atom count: `5`
- total raw oracle calls: `195296`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-5191-7309-8811-12272: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-5191-7309-8811-12272-5atom-oracle-only/report.json)
- random1-layer26-units5754-8561-9962-11673-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer26-units5754-8561-9962-11673-13025-5atom-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
