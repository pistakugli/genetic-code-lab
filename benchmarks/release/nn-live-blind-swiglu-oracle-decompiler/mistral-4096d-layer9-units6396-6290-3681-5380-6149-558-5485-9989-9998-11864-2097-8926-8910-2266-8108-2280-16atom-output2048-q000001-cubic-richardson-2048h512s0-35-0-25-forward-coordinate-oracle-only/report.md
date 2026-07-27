# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `9/[6396, 6290, 3681, 5380, 6149, 558, 5485, 9989, 9998, 11864, 2097, 8926, 8910, 2266, 8108, 2280]`
- oracle mode: `down-proj-output-channel`
- output channel: `2048`
- raw oracle calls: `327617`
- validation R2: `0.3136435320527343`
- validation MAE: `0.0002281195703873046`
- subspace min principal cosine: `0.06709786586252944`
- post-hoc paired atom min joint cosine: `0.006218186712332804`
- post-hoc paired atoms >= 0.99: `0/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
