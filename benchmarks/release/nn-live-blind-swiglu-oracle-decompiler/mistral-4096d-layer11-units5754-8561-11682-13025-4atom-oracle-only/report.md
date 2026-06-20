# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `11/[5754, 8561, 11682, 13025]`
- raw oracle calls: `79808`
- validation R2: `0.995727068595925`
- validation MAE: `0.0006359374939938832`
- subspace min principal cosine: `0.9999999725417702`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
