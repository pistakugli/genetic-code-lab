# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 5191, 5754, 7309, 8811, 11682, 12272, 13025]`
- raw oracle calls: `118425`
- validation R2: `0.995860081804988`
- validation MAE: `0.0010036287991970745`
- subspace min principal cosine: `0.9999870289209187`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
