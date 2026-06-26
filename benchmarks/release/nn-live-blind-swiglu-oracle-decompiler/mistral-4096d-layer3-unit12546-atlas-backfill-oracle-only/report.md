# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `3/[12546]`
- raw oracle calls: `18496`
- validation R2: `0.9999999994572247`
- validation MAE: `1.2251641913332798e-07`
- subspace min principal cosine: `0.9999999998123645`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
