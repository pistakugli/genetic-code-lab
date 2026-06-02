# Local Transformer MLP Hook Benchmark

Generated: 2026-06-02T19:55:19.229Z

## Claim Scope

- Proves if pass: GCL can recover explicit GELU source structure for a hooked local transformer MLP output slice and map recovered atoms to source expansion neurons using real captured activation rows and source provenance.
- Does not claim: full GPT-2/LLM FFN extraction, arbitrary transformer block decompilation, output-only original-neuron identity without hooks, or broad SOTA status.

## Summary

- Pass: true
- Hook rows: 384
- Effective channels: 3
- Max hook MAE: 0.005720353338158273
- Hook R2: 0.9998897778278036
- Worst hook error: 0.014246698868125185
- Provenance confidence min: 0.9106904584057108
- Raw oracle calls: 19540

## Hook Integrity

- Max preactivation error: 0
- Max activation error: 0
- Max target output error: 2.220446049250313e-16

## Provenance Matches

- recovered atom 1 -> local.transformer.h0.mlp.c_fc[4]
  - source ids: local.transformer.h0.mlp.c_fc[4]
  - layer: local.transformer.h0.mlp.c_fc
  - indices: 4
  - confidence: 0.9557456281557765
  - direction cosine: 0.9999951331472965
  - contribution correlation: 0.9999842975659389
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 2 -> local.transformer.h0.mlp.c_fc[8]
  - source ids: local.transformer.h0.mlp.c_fc[8]
  - layer: local.transformer.h0.mlp.c_fc
  - indices: 8
  - confidence: 0.9503587716912458
  - direction cosine: 0.9999888457696828
  - contribution correlation: 0.999956168059748
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 0 -> local.transformer.h0.mlp.c_fc[1]
  - source ids: local.transformer.h0.mlp.c_fc[1]
  - layer: local.transformer.h0.mlp.c_fc
  - indices: 1
  - confidence: 0.9106904584057108
  - direction cosine: 0.9999969258145516
  - contribution correlation: 0.9998884591087599
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.

## Conclusion

The local transformer hook gate passed. GCL recovered explicit GELU source structure from a hooked local transformer MLP output slice and mapped recovered atoms back to source expansion neurons with why-reacts evidence.
