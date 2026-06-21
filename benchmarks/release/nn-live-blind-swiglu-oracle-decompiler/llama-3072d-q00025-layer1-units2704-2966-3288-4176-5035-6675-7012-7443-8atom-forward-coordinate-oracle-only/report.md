# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[2704, 2966, 3288, 4176, 5035, 6675, 7012, 7443]`
- raw oracle calls: `213793`
- validation R2: `0.9996458860257502`
- validation MAE: `0.009205129181165679`
- subspace min principal cosine: `0.9995635612093632`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
