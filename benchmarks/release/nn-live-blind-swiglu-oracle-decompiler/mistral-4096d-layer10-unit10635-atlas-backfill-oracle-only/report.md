# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `10/[10635]`
- raw oracle calls: `18496`
- validation R2: `0.9999928846354375`
- validation MAE: `9.876789864590715e-06`
- subspace min principal cosine: `0.9999999976843506`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
