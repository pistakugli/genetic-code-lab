# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `29/[3400]`
- raw oracle calls: `18496`
- validation R2: `0.9999999992747053`
- validation MAE: `1.5900490861993595e-07`
- subspace min principal cosine: `0.9999999997426968`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
