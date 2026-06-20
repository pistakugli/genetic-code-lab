# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.995727068595925`
- max MAE: `0.0006359374939938832`
- max worst error: `0.008566143276326953`
- min subspace principal cosine: `0.9999999725417702`
- max unit count: `4`
- max atom count: `4`
- total raw oracle calls: `159616`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-7309-8811-12272: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-7309-8811-12272-4atom-oracle-only/report.json)
- random1-layer11-units5754-8561-11682-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer11-units5754-8561-11682-13025-4atom-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
