# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 1024, 2048, 4096]`
- raw oracle calls: `68241`
- validation R2: `0.9999913818259203`
- validation MAE: `0.0017664391250281214`
- subspace min principal cosine: `0.9999919539106246`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
