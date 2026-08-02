# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `15/[460, 618, 778, 1141, 1279, 2062, 3627, 3679, 4186, 5014, 5041, 5409, 6455, 7434, 7598, 7663]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `146337`
- validation R2: `0.9986197880720584`
- validation MAE: `0.006923393240320399`
- subspace min principal cosine: `0.9999854124669576`
- post-hoc paired atom min joint cosine: `0.23861138806479312`
- post-hoc paired atoms >= 0.99: `14/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
