# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `22/[5311]`
- raw oracle calls: `18496`
- validation R2: `0.9999699854013766`
- validation MAE: `2.119507065536974e-05`
- subspace min principal cosine: `0.9999999988824975`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
