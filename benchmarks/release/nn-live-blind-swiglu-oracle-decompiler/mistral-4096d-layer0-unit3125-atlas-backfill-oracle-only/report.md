# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `0/[3125]`
- raw oracle calls: `18496`
- validation R2: `0.9999572674827869`
- validation MAE: `2.3960429305297835e-05`
- subspace min principal cosine: `0.9999999997599606`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
