# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `24/[6813]`
- raw oracle calls: `18496`
- validation R2: `0.9999929725597355`
- validation MAE: `1.0899747100874565e-05`
- subspace min principal cosine: `0.9999999998863721`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
