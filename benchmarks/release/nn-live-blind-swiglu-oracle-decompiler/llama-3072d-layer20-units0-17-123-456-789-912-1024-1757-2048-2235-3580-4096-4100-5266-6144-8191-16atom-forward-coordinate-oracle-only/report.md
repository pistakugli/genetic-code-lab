# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 912, 1024, 1757, 2048, 2235, 3580, 4096, 4100, 5266, 6144, 8191]`
- raw oracle calls: `138401`
- validation R2: `0.9915249275292025`
- validation MAE: `0.025860451651904412`
- subspace min principal cosine: `0.9998458489586742`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
