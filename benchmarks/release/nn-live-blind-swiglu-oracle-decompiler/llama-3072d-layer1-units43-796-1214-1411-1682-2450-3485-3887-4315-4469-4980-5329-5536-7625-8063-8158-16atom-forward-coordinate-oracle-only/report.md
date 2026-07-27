# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[43, 796, 1214, 1411, 1682, 2450, 3485, 3887, 4315, 4469, 4980, 5329, 5536, 7625, 8063, 8158]`
- raw oracle calls: `129953`
- validation R2: `0.9955437504320315`
- validation MAE: `0.012203156738449156`
- subspace min principal cosine: `0.9997919983764019`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
