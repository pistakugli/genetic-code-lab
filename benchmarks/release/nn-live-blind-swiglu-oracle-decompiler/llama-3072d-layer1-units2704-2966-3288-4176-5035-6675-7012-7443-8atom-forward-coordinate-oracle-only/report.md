# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[2704, 2966, 3288, 4176, 5035, 6675, 7012, 7443]`
- raw oracle calls: `68241`
- validation R2: `0.9999831138300853`
- validation MAE: `0.0019752329027485404`
- subspace min principal cosine: `0.9999676383453313`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
