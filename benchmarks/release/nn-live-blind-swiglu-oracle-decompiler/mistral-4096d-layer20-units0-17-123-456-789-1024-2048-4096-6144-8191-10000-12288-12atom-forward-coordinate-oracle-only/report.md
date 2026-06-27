# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 1024, 2048, 4096, 6144, 8191, 10000, 12288]`
- raw oracle calls: `128697`
- validation R2: `0.9952350522320941`
- validation MAE: `0.0013654271559954592`
- subspace min principal cosine: `0.9999973679313839`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
