# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9996022410326608`
- max MAE: `0.0036226998892415745`
- max worst error: `0.03205623326481288`
- min subspace principal cosine: `0.9999935586279501`
- min post-hoc paired atom joint cosine: `0.9943894160746654`
- min paired atoms above threshold: `16`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `261057`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer9-units192-331-1612-5025-5229-5288-5508-5553-5623-5701-6097-6372-6510-6653-7591-7666: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-units192-331-1612-5025-5229-5288-5508-5553-5623-5701-6097-6372-6510-6653-7591-7666-16atom-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
