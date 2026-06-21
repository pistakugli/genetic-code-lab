# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `12/[279, 3125, 8561, 9962, 11673, 13025]`
- raw oracle calls: `62253`
- validation R2: `0.9976865432600855`
- validation MAE: `0.0005876331035189414`
- subspace min principal cosine: `0.9999991210977408`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
