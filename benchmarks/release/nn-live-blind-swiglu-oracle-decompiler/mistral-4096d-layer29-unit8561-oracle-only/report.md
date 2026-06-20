# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `29/8561`
- raw oracle calls: `18496`
- validation R2: `0.9999877842354408`
- validation MAE: `1.3637489816799884e-05`
- subspace min principal cosine: `0.9999999957944806`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a single real SwiGLU-unit black-box gate, not full LLM decompilation.
