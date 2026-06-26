# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `28/[9817]`
- raw oracle calls: `18496`
- validation R2: `0.9999972610027758`
- validation MAE: `7.009058210551625e-06`
- subspace min principal cosine: `0.9999999997220904`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
