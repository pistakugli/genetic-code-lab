# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `11/[159, 1786, 4892, 5692, 6670, 7443]`
- raw oracle calls: `48941`
- validation R2: `0.9999880522847288`
- validation MAE: `0.0012813828353731324`
- subspace min principal cosine: `0.9999574008675185`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
