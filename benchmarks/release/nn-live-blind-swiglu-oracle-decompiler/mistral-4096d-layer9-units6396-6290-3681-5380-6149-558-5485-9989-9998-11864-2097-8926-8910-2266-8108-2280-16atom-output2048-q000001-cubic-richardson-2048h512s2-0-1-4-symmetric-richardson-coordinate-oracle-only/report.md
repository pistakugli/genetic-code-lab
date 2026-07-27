# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `9/[6396, 6290, 3681, 5380, 6149, 558, 5485, 9989, 9998, 11864, 2097, 8926, 8910, 2266, 8108, 2280]`
- oracle mode: `down-proj-output-channel`
- output channel: `2048`
- raw oracle calls: `1674880`
- validation R2: `0.9927064132277419`
- validation MAE: `2.294618776565714e-05`
- subspace min principal cosine: `0.9997226549029208`
- post-hoc paired atom min joint cosine: `0.2773761459982943`
- post-hoc paired atoms >= 0.99: `2/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
