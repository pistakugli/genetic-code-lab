# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `9/[6396, 6290, 3681, 5380, 6149, 558, 5485, 9989, 9998, 11864, 2097, 8926, 8910, 2266, 8108, 2280]`
- oracle mode: `down-proj-output-channel`
- output channel: `2048`
- raw oracle calls: `107696`
- validation R2: `0.9996824964424941`
- validation MAE: `5.019407967445103e-06`
- subspace min principal cosine: `0.9997226549029208`
- post-hoc paired atom min joint cosine: `0.9989749892979034`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
