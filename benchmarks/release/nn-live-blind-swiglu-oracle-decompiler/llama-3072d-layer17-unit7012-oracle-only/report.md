# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `17/7012`
- raw oracle calls: `14400`
- validation R2: `0.9999999997595209`
- validation MAE: `2.4327716816962254e-06`
- subspace min principal cosine: `0.9999999998777513`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a single real SwiGLU-unit black-box gate, not full LLM decompilation.
