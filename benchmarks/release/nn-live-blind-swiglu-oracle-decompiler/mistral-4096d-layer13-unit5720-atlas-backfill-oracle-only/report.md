# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `13/[5720]`
- raw oracle calls: `18496`
- validation R2: `0.9999739685264046`
- validation MAE: `2.1144765597480086e-05`
- subspace min principal cosine: `0.9999999990897485`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
