# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `28/[14215, 5565, 5889, 2863, 9223, 9895, 8226, 13209, 206, 9472, 3117, 1045, 13052, 10353, 2588, 4768]`
- oracle mode: `down-proj-output-channel`
- output channel: `443`
- raw oracle calls: `4096`
- validation R2: `0.9925367000746828`
- validation MAE: `6.437900093358e-06`
- subspace min principal cosine: `0.9790955523785366`
- post-hoc paired atom min joint cosine: `0.06574003182738468`
- post-hoc paired atoms >= 0.99: `11/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
