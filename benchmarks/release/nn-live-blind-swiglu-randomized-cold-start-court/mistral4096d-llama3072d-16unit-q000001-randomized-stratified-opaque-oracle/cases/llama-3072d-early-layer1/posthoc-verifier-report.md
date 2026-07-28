# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[1667, 122, 3394, 6789, 3581, 257, 5164, 2514, 3421, 3366, 4261, 3890, 6626, 193, 7734, 2265]`
- oracle mode: `down-proj-output-channel`
- output channel: `1947`
- raw oracle calls: `4096`
- validation R2: `0.9985120140939311`
- validation MAE: `0.0003116977985006124`
- subspace min principal cosine: `0.9982453221496379`
- post-hoc paired atom min joint cosine: `-0.042941443942983035`
- post-hoc paired atoms >= 0.99: `12/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
