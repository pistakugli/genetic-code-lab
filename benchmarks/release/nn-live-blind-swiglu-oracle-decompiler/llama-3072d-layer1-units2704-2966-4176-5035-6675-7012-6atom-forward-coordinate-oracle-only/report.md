# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[2704, 2966, 4176, 5035, 6675, 7012]`
- raw oracle calls: `48941`
- validation R2: `0.9999952190867781`
- validation MAE: `0.0008688058664558853`
- subspace min principal cosine: `0.9999900918580495`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
