# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9915249275292025`
- max MAE: `0.025860451651904412`
- max worst error: `0.2709458526175194`
- min subspace principal cosine: `0.9998458489586742`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `138401`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units0-17-123-456-789-912-1024-1757-2048-2235-3580-4096-4100-5266-6144-8191: `fail summary.pass=false; r2=0.9915249275292025<0.995; mae=0.025860451651904412>0.015; worstError=0.2709458526175194>0.12` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units0-17-123-456-789-912-1024-1757-2048-2235-3580-4096-4100-5266-6144-8191-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
