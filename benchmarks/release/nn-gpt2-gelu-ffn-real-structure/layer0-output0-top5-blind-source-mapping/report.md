# Real GPT-2 GELU FFN Structure Benchmark

Generated: 2026-06-05T16:17:21.326Z

## Claim Scope

- Proves if pass: GCL can recover explicit GELU atoms from a real GPT-2 layer-0 FFN top-K output-channel slice and map them to exact GPT-2 source channels through provenance verification.
- Does not claim: full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary tokenizer/model execution, output-only original-neuron identity without source manifest/hooks, or broad SOTA status.

## Black-Box Boundary

- Recovery oracle: GCL receives f(vector) for the compact real GPT-2 FFN output slice during recovery
- Post-recovery provenance: Source manifest is used only after recovery to verify exact GPT-2 channel ids, weights, and why-reacts evidence.
- Structure gate: Pass requires recovered atom count and per-source provenance matches, not only output R2.

## Fixture

- Source fixture type: gpt2-layer0-ffn-full-output-channel-0
- Source fixture SHA-256: ae9a59434c1e25bf4921c31c8c6ebd49d3a01c64aebf16119952580010f532dd
- Dimension: 768
- Output channel: 0
- Top-K: 5
- Selected by: abs(w_proj[j]) * ||W_fc_T[j]||

## Summary

- Pass: true
- Effective channels: 5
- Source groups: 5
- R2: 0.9998249080496286
- MAE: 0.010216135944799058
- Worst error: 0.06915410304531516
- Agreement MAE: 0.010216135944799058
- Min provenance confidence: 0.9433099991784843
- Min direction cosine: 0.9997811523380504
- Raw oracle calls: 64820

## Real GPT-2 Source Channels

- gpt2.transformer.h0.mlp.c_fc[1614]->c_proj[0]
  - contribution rank: 1
  - GPT-2 source index: 1614
  - input norm: 6.2323015639747625
  - input bias: -0.24261945486068726
  - output weight: 0.315574586391449
- gpt2.transformer.h0.mlp.c_fc[2452]->c_proj[0]
  - contribution rank: 2
  - GPT-2 source index: 2452
  - input norm: 4.718052010412211
  - input bias: -0.3757937252521515
  - output weight: -0.3931875228881836
- gpt2.transformer.h0.mlp.c_fc[2053]->c_proj[0]
  - contribution rank: 3
  - GPT-2 source index: 2053
  - input norm: 6.089242318774106
  - input bias: -0.4621184766292572
  - output weight: 0.25748389959335327
- gpt2.transformer.h0.mlp.c_fc[486]->c_proj[0]
  - contribution rank: 4
  - GPT-2 source index: 486
  - input norm: 4.414083936895953
  - input bias: -0.4505223333835602
  - output weight: -0.354936420917511
- gpt2.transformer.h0.mlp.c_fc[2172]->c_proj[0]
  - contribution rank: 5
  - GPT-2 source index: 2172
  - input norm: 3.973872319365409
  - input bias: -0.4422215223312378
  - output weight: -0.3758499324321747

## Provenance Matches

- recovered atom 4 -> gpt2.transformer.h0.mlp.c_fc[2172]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2172]->c_proj[0]
  - confidence: 0.9735265377686241
  - direction cosine: 0.9998586629358526
  - norm error: 0.11339889069417897
  - bias error: 0.0026731672159248654
  - output weight error: 0.009284416105700632
  - contribution correlation: 0.9997679217068937
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 1 -> gpt2.transformer.h0.mlp.c_fc[2452]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2452]->c_proj[0]
  - confidence: 0.9700286611694419
  - direction cosine: 0.9999466394501012
  - norm error: 0.18855355993945988
  - bias error: 0.004067585354775971
  - output weight error: 0.011020777075720578
  - contribution correlation: 0.9999058216684069
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 2 -> gpt2.transformer.h0.mlp.c_fc[486]
  - source ids: gpt2.transformer.h0.mlp.c_fc[486]->c_proj[0]
  - confidence: 0.9663674624426674
  - direction cosine: 0.9997811523380504
  - norm error: 0.19520967192725625
  - bias error: 0.0013194542773818374
  - output weight error: 0.01304555800861662
  - contribution correlation: 0.9996694146325225
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 0 -> gpt2.transformer.h0.mlp.c_fc[1614]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1614]->c_proj[0]
  - confidence: 0.9637493625105665
  - direction cosine: 0.9999381672644937
  - norm error: 0.6496354615676783
  - bias error: 0.004506121020959236
  - output weight error: 0.031969233169365785
  - contribution correlation: 0.9998202242174615
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 3 -> gpt2.transformer.h0.mlp.c_fc[2053]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2053]->c_proj[0]
  - confidence: 0.9433099991784843
  - direction cosine: 0.9997941114363316
  - norm error: 0.8899950349740999
  - bias error: 0.04009763743810418
  - output weight error: 0.03883370680051296
  - contribution correlation: 0.9994628345815689
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.

## Conclusion

The real GPT-2 compact FFN structure gate passed. GCL recovered explicit GELU source atoms from oracle calls over real GPT-2 layer-0 weights and verified source-channel provenance against the compact source manifest.
