# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `16/2048`
- raw oracle calls: `18496`
- validation R2: `0.9999993307980108`
- validation MAE: `3.5707296934936635e-06`
- subspace min principal cosine: `0.9999997892743723`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a single real SwiGLU-unit black-box gate, not full LLM decompilation.
