# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `8/8`
- min R2: `0.9999570341720895`
- max MAE: `0.000024326629529818456`
- max worst error: `0.00044138616892425153`
- min subspace principal cosine: `0.9999999957944806`
- total raw oracle calls: `147968`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-unit7309: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-unit7309-oracle-only/report.json)
- random1-layer19-unit12272: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer19-unit12272-oracle-only/report.json)
- random2-layer10-unit5191: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer10-unit5191-oracle-only/report.json)
- random3-layer26-unit5754: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer26-unit5754-oracle-only/report.json)
- random4-layer29-unit8561: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer29-unit8561-oracle-only/report.json)
- random5-layer22-unit11673: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer22-unit11673-oracle-only/report.json)
- random6-layer0-unit3125: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer0-unit3125-oracle-only/report.json)
- random7-layer7-unit14251: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer7-unit14251-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
