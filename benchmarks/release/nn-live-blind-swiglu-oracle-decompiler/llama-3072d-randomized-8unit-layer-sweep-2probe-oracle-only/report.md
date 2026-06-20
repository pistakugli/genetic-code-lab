# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `8/8`
- min R2: `0.9999981324158254`
- max MAE: `0.00016030715095856503`
- max worst error: `0.001568794345920943`
- min subspace principal cosine: `0.9999994111232651`
- total raw oracle calls: `115200`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-unit4176: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-unit4176-oracle-only/report.json)
- random1-layer17-unit7012: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer17-unit7012-oracle-only/report.json)
- random2-layer9-unit2966: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-unit2966-oracle-only/report.json)
- random3-layer22-unit3288: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer22-unit3288-oracle-only/report.json)
- random4-layer25-unit4892: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer25-unit4892-oracle-only/report.json)
- random5-layer19-unit6670: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer19-unit6670-oracle-only/report.json)
- random6-layer0-unit1786: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer0-unit1786-oracle-only/report.json)
- random7-layer6-unit8143: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer6-unit8143-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
