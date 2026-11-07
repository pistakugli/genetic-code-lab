# Blind 3584D SwiGLU Oracle Decompiler Sweep (Kaggle T4)

- pass: `True`
- cases: `8/8`
- min R2: `0.9999999957331516`
- max MAE: `1.0189573986133082e-05`
- max worst error: `0.00016841797026345695`
- min subspace principal cosine: `0.9999999980342267`
- max unit count: `1`
- total raw oracle calls: `131584`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-unit9659: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer1-unit9659-oracle-only-kaggle-t4-2026-11-07/report.json)
- random1-layer17-unit16216: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer17-unit16216-oracle-only-kaggle-t4-2026-11-07/report.json)
- random2-layer9-unit6859: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer9-unit6859-oracle-only-kaggle-t4-2026-11-07/report.json)
- random3-layer22-unit7603: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer22-unit7603-oracle-only-kaggle-t4-2026-11-07/report.json)
- random4-layer25-unit11313: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer25-unit11313-oracle-only-kaggle-t4-2026-11-07/report.json)
- random5-layer19-unit15426: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer19-unit15426-oracle-only-kaggle-t4-2026-11-07/report.json)
- random6-layer0-unit4130: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer0-unit4130-oracle-only-kaggle-t4-2026-11-07/report.json)
- random7-layer6-unit18831: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/qwen-3584d-layer6-unit18831-oracle-only-kaggle-t4-2026-11-07/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
