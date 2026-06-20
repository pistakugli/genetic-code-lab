# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `11/[159, 1786, 4892, 5692, 6670, 7443]`
- raw oracle calls: `88864`
- validation R2: `0.9999999419380987`
- validation MAE: `8.572753027982732e-05`
- subspace min principal cosine: `0.9999998049413197`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
