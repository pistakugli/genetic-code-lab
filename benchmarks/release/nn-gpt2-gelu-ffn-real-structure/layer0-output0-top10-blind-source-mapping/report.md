# Real GPT-2 GELU FFN Structure Benchmark

Generated: 2026-06-03T11:31:39.774Z

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
- Top-K: 10
- Selected by: abs(w_proj[j]) * ||W_fc_T[j]||

## Summary

- Pass: true
- Effective channels: 10
- Source groups: 10
- R2: 0.9996487749720989
- MAE: 0.01549465009745577
- Worst error: 0.10184722252510248
- Agreement MAE: 0.01549465009745577
- Min provenance confidence: 0.8397038910070366
- Min direction cosine: 0.998244983491102
- Raw oracle calls: 219268

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
- gpt2.transformer.h0.mlp.c_fc[1981]->c_proj[0]
  - contribution rank: 6
  - GPT-2 source index: 1981
  - input norm: 5.29590616166252
  - input bias: 0.08856736868619919
  - output weight: -0.2817477285861969
- gpt2.transformer.h0.mlp.c_fc[137]->c_proj[0]
  - contribution rank: 7
  - GPT-2 source index: 137
  - input norm: 5.705716975469597
  - input bias: -0.3142358064651489
  - output weight: -0.25851255655288696
- gpt2.transformer.h0.mlp.c_fc[257]->c_proj[0]
  - contribution rank: 8
  - GPT-2 source index: 257
  - input norm: 5.256108475110736
  - input bias: -0.19954240322113037
  - output weight: 0.2738417983055115
- gpt2.transformer.h0.mlp.c_fc[1554]->c_proj[0]
  - contribution rank: 9
  - GPT-2 source index: 1554
  - input norm: 4.781127244855866
  - input bias: -0.05089453235268593
  - output weight: 0.29472604393959045
- gpt2.transformer.h0.mlp.c_fc[2211]->c_proj[0]
  - contribution rank: 10
  - GPT-2 source index: 2211
  - input norm: 5.410943362149176
  - input bias: -0.20066377520561218
  - output weight: 0.25622081756591797

## Provenance Matches

- recovered atom 8 -> gpt2.transformer.h0.mlp.c_fc[2211]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2211]->c_proj[0]
  - confidence: 0.980372179729044
  - direction cosine: 0.9998924158026464
  - norm error: 0.2841143562154276
  - bias error: 0.007067119686427037
  - output weight error: 0.013012898116568417
  - contribution correlation: 0.9998313024955339
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 6 -> gpt2.transformer.h0.mlp.c_fc[257]
  - source ids: gpt2.transformer.h0.mlp.c_fc[257]->c_proj[0]
  - confidence: 0.9756121343529943
  - direction cosine: 0.9998405848385982
  - norm error: 0.29549296548664206
  - bias error: 0.0012000562296416994
  - output weight error: 0.014895913515060866
  - contribution correlation: 0.9997920516694442
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 3 -> gpt2.transformer.h0.mlp.c_fc[486]
  - source ids: gpt2.transformer.h0.mlp.c_fc[486]->c_proj[0]
  - confidence: 0.9661091285264186
  - direction cosine: 0.9998800719838138
  - norm error: 0.15927048441220215
  - bias error: 0.006134284568393877
  - output weight error: 0.009132121795182024
  - contribution correlation: 0.9998090278358233
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 4 -> gpt2.transformer.h0.mlp.c_fc[2172]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2172]->c_proj[0]
  - confidence: 0.9658701737251407
  - direction cosine: 0.9997110472469561
  - norm error: 0.14502315943222266
  - bias error: 0.009799158069006775
  - output weight error: 0.017309158048497753
  - contribution correlation: 0.9994776495984289
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 1 -> gpt2.transformer.h0.mlp.c_fc[1614]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1614]->c_proj[0]
  - confidence: 0.9657878500484789
  - direction cosine: 0.9998603583941154
  - norm error: 0.393056942150376
  - bias error: 0.006121123576177778
  - output weight error: 0.016728545400776063
  - contribution correlation: 0.9997581684222597
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 5 -> gpt2.transformer.h0.mlp.c_fc[137]
  - source ids: gpt2.transformer.h0.mlp.c_fc[137]->c_proj[0]
  - confidence: 0.960822526412711
  - direction cosine: 0.999890526681576
  - norm error: 0.4859696728613221
  - bias error: 0.0010901301929024387
  - output weight error: 0.019646015689128937
  - contribution correlation: 0.9997586035406861
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 9 -> gpt2.transformer.h0.mlp.c_fc[1981]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1981]->c_proj[0]
  - confidence: 0.9155311569708363
  - direction cosine: 0.9995561549632689
  - norm error: 0.03211408515872893
  - bias error: 0.04113046562632655
  - output weight error: 0.024783881628179627
  - contribution correlation: 0.9993648573857712
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 0 -> gpt2.transformer.h0.mlp.c_fc[2452]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2452]->c_proj[0]
  - confidence: 0.8766033902031813
  - direction cosine: 0.9993566862991713
  - norm error: 0.3721070334293124
  - bias error: 0.04404660841370889
  - output weight error: 0.07741091725647675
  - contribution correlation: 0.999082795610749
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 7 -> gpt2.transformer.h0.mlp.c_fc[2053]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2053]->c_proj[0]
  - confidence: 0.8681727920192805
  - direction cosine: 0.9994980333609708
  - norm error: 0.46083743647998876
  - bias error: 0.0038621466447033015
  - output weight error: 0.007530715190949333
  - contribution correlation: 0.9989922816351698
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 2 -> gpt2.transformer.h0.mlp.c_fc[1554]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1554]->c_proj[0]
  - confidence: 0.8397038910070366
  - direction cosine: 0.998244983491102
  - norm error: 0.5308592210056071
  - bias error: 0.01328869604898162
  - output weight error: 0.08510185004313126
  - contribution correlation: 0.997624539756804
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.

## Conclusion

The real GPT-2 compact FFN structure gate passed. GCL recovered explicit GELU source atoms from oracle calls over real GPT-2 layer-0 weights and verified source-channel provenance against the compact source manifest.
