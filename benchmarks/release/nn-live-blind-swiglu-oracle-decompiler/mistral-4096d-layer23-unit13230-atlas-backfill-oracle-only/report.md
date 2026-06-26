# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `23/[13230]`
- raw oracle calls: `60512`
- validation R2: `0.9999970332853755`
- validation MAE: `7.3885763971455725e-06`
- subspace min principal cosine: `0.9999999998108878`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
