# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9994183507857979`
- max MAE: `0.004865868826406336`
- max worst error: `0.03331589242704447`
- min subspace principal cosine: `0.9997557479909298`
- min post-hoc paired atom joint cosine: `0.9968646816847491`
- min paired atoms above threshold: `16`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `142241`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units27-57-413-500-1607-1970-2215-2529-2847-3530-3786-4033-4236-4906-7283-7720: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units27-57-413-500-1607-1970-2215-2529-2847-3530-3786-4033-4236-4906-7283-7720-16atom-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
