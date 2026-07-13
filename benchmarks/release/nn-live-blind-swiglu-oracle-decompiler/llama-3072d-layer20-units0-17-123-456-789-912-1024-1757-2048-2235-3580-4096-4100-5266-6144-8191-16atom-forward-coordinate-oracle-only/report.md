# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 912, 1024, 1757, 2048, 2235, 3580, 4096, 4100, 5266, 6144, 8191]`
- raw oracle calls: `129953`
- validation R2: `0.996028807883924`
- validation MAE: `0.013147489844684133`
- subspace min principal cosine: `0.9996660748268255`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
