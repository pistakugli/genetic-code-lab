# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `5/[14048]`
- raw oracle calls: `18496`
- validation R2: `0.9999999992204276`
- validation MAE: `1.5261228548399194e-07`
- subspace min principal cosine: `0.999999999747602`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
