# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `16/[159, 1786, 1841, 2531, 5108, 5692, 6670, 8143]`
- raw oracle calls: `68241`
- validation R2: `0.9999781080257685`
- validation MAE: `0.002327151614302204`
- subspace min principal cosine: `0.999936352892748`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
