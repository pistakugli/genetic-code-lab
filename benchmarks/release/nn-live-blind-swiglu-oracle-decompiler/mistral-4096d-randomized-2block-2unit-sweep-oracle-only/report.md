# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9994744138350042`
- max MAE: `0.00014261562921587213`
- max worst error: `0.0027002211714715688`
- min subspace principal cosine: `0.9999999823144813`
- max unit count: `2`
- max atom count: `2`
- total raw oracle calls: `88256`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units7309-8811: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units7309-8811-2atom-oracle-only/report.json)
- random1-layer27-units4733-5191: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer27-units4733-5191-2atom-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
