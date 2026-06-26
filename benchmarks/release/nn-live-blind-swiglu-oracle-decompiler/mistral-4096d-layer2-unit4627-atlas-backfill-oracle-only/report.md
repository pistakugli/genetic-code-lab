# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `2/[4627]`
- raw oracle calls: `18496`
- validation R2: `0.999999999196481`
- validation MAE: `1.6168313913195662e-07`
- subspace min principal cosine: `0.9999999996247264`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
