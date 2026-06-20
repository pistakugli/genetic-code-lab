# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 5191, 7309, 8811, 12272]`
- raw oracle calls: `97648`
- validation R2: `0.9975004893948433`
- validation MAE: `0.0005850146019813779`
- subspace min principal cosine: `0.9999997906113538`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
