# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `12/[3678]`
- raw oracle calls: `14400`
- validation R2: `0.9999999997416598`
- validation MAE: `1.6507240537125818e-06`
- subspace min principal cosine: `0.9999999999060746`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
