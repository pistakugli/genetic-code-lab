# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `27/[1898]`
- raw oracle calls: `18496`
- validation R2: `0.9999413531229668`
- validation MAE: `3.3574095846065474e-05`
- subspace min principal cosine: `0.999999980914922`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
