# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9999989656096891`
- max MAE: `0.0003507732790537803`
- max worst error: `0.002729982480284221`
- min subspace principal cosine: `0.9999992930409601`
- min post-hoc paired atom joint cosine: `0.999999896454588`
- min paired atoms above threshold: `null`
- min source cancellation fraction: `0.3893537121368381`
- minimum source pair correlation: `-0.021986979484849047`
- max unit count: `4`
- max atom count: `4`
- total raw oracle calls: `72017`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units0-17-123-456: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units0-17-123-456-4atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
