# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[2704, 2966, 4176, 5035, 6675, 7012]`
- raw oracle calls: `88864`
- validation R2: `0.9999999730266301`
- validation MAE: `6.078547077844751e-05`
- subspace min principal cosine: `0.9999999238264758`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
