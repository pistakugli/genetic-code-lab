# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 1024, 2048, 4096]`
- raw oracle calls: `118425`
- validation R2: `0.9963343164135441`
- validation MAE: `0.0010131811501723571`
- subspace min principal cosine: `0.9999930621807005`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
