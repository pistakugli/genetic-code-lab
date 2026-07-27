# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `9/[958, 1532, 1647, 1765, 2755, 3182, 3856, 4191, 4371, 4373, 5442, 7098, 7145, 7305, 7679, 7685]`
- raw oracle calls: `16384`
- validation R2: `0.9991602047715517`
- validation MAE: `0.006134974735926905`
- subspace min principal cosine: `0.9999976562319123`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
