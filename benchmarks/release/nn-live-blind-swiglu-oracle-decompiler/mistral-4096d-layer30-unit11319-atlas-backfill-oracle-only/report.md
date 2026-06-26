# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `30/[11319]`
- raw oracle calls: `18496`
- validation R2: `0.9999999366494735`
- validation MAE: `9.234236268911176e-07`
- subspace min principal cosine: `0.9999999999121852`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
