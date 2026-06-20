# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `26/[5754, 8561, 9962, 11673, 13025]`
- raw oracle calls: `97648`
- validation R2: `0.9951436177161573`
- validation MAE: `0.0008022104114991162`
- subspace min principal cosine: `0.9999999948414455`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
