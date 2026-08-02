# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9985677573817285`
- max MAE: `0.007624356439002252`
- max worst error: `0.06379003054895938`
- min subspace principal cosine: `0.9999530973154566`
- min post-hoc paired atom joint cosine: `0.990872448218614`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6817551629017323`
- minimum source pair correlation: `-0.039982811825832104`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `146337`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units27-57-413-500-1607-1970-2215-2529-2847-3530-3786-4033-4236-4906-7283-7720: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units27-57-413-500-1607-1970-2215-2529-2847-3530-3786-4033-4236-4906-7283-7720-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
