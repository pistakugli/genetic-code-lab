# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `14/[13639]`
- raw oracle calls: `18496`
- validation R2: `0.99999323833154`
- validation MAE: `1.1135759713817718e-05`
- subspace min principal cosine: `0.9999999984893235`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
