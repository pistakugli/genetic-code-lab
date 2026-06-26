# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `5/[3257]`
- raw oracle calls: `14400`
- validation R2: `0.9999999997268147`
- validation MAE: `2.3685539224242534e-06`
- subspace min principal cosine: `0.9999999999031076`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
