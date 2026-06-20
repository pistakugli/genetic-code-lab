# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `16/2048`
- raw oracle calls: `14400`
- validation R2: `0.9999999967514999`
- validation MAE: `1.0737975340964412e-05`
- subspace min principal cosine: `0.9999999984260104`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a single real SwiGLU-unit black-box gate, not full LLM decompilation.
