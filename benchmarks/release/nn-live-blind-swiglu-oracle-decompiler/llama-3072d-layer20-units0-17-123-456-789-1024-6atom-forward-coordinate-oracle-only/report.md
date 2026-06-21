# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 1024]`
- raw oracle calls: `48941`
- validation R2: `0.9999975420086665`
- validation MAE: `0.0008233366167048662`
- subspace min principal cosine: `0.9999960757027645`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
