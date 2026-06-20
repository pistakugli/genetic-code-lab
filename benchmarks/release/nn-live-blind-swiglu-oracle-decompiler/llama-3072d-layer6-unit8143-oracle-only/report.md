# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `6/8143`
- raw oracle calls: `14400`
- validation R2: `0.9999999961670245`
- validation MAE: `1.0479134746608307e-05`
- subspace min principal cosine: `0.9999999974553423`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a single real SwiGLU-unit black-box gate, not full LLM decompilation.
