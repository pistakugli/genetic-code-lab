# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `31/[4902]`
- raw oracle calls: `18496`
- validation R2: `0.9999869953814583`
- validation MAE: `1.445194556868763e-05`
- subspace min principal cosine: `0.9999999914365378`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
