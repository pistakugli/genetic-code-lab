# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9968086229936512`
- max MAE: `0.0013585229495551043`
- max worst error: `0.01298486925016918`
- min subspace principal cosine: `0.999989791877386`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `188193`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units75-1393-2126-2469-2944-4288-6099-6803-7551-7821-8715-9325-9689-13343-14111-14277: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units75-1393-2126-2469-2944-4288-6099-6803-7551-7821-8715-9325-9689-13343-14111-14277-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
