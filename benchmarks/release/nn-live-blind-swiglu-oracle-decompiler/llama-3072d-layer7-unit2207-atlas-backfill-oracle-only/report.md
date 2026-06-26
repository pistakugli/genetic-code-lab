# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `7/[2207]`
- raw oracle calls: `14400`
- validation R2: `0.9999999847064683`
- validation MAE: `1.2042596339996148e-05`
- subspace min principal cosine: `0.9999999911482416`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
