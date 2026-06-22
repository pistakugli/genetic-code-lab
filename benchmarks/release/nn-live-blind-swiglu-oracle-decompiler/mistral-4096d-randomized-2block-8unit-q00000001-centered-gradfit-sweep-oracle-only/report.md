# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9957594628054579`
- max MAE: `0.0010088005020659832`
- max worst error: `0.009621486908707531`
- min subspace principal cosine: `0.9999828597286717`
- max unit count: `8`
- max atom count: `8`
- total raw oracle calls: `171298`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-5191-5754-7309-8811-11682-12272-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-5191-5754-7309-8811-11682-12272-13025-8atom-q1e-8-forward-coordinate-oracle-only/report.json)
- random1-layer19-units279-3125-3222-4429-8940-9962-11673-14251: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer19-units279-3125-3222-4429-8940-9962-11673-14251-8atom-q1e-8-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
