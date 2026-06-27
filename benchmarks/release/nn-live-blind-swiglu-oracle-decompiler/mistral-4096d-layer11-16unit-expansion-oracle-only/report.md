# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9971375925273832`
- max MAE: `0.0012542861266610903`
- max worst error: `0.018289335513027943`
- min subspace principal cosine: `0.9999932242854456`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `188193`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer11-units1678-2682-2882-3089-4822-5569-6749-7335-7649-7652-9524-12422-12505-12785-13438-13449: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer11-units1678-2682-2882-3089-4822-5569-6749-7335-7649-7652-9524-12422-12505-12785-13438-13449-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
