# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `Infinity`
- max MAE: `-Infinity`
- max worst error: `-Infinity`
- min subspace principal cosine: `Infinity`
- min post-hoc paired atom joint cosine: `Infinity`
- min paired atoms above threshold: `Infinity`
- min source cancellation fraction: `Infinity`
- minimum source pair correlation: `Infinity`
- max unit count: `-Infinity`
- max atom count: `-Infinity`
- total raw oracle calls: `0`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer1-units27-57-413-460-500-618-778-1141-1607-1970-2062-2215-2529-2847-3530-3627-3679-3786-4033-4186-4236-4624-4906-5014-5041-5409-6455-7283-7434-7598-7663-7720: `fail missing report.json; process killed by SIGTERM; stderr tail: {"type": "progress", "phase": "oracle-ready", "dimension": 3072, "unitCount": 32, "layer": 1} | {"type": "progress", "phase": "basis-recovered", "activeRank": 64, "gradientWitnessCount": 65, "energyRatio": 0.999999999999999} | {"type": "progress", "phase": "train-sampled", "trainCount": 32000, "rawOracleCalls": 231745}` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units27-57-413-460-500-618-778-1141-1607-1970-2062-2215-2529-2847-3530-3627-3679-3786-4033-4186-4236-4624-4906-5014-5041-5409-6455-7283-7434-7598-7663-7720-32atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
