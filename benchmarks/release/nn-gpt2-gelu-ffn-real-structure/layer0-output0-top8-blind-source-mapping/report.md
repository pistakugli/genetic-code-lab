# Real GPT-2 GELU FFN Structure Benchmark

Generated: 2026-06-03T08:39:42.625Z

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
- Top-K: 8
- Selected by: abs(w_proj[j]) * ||W_fc_T[j]||

## Summary

- Pass: true
- Effective channels: 8
- Source groups: 8
- R2: 0.9998280690875673
- MAE: 0.010762522489570131
- Worst error: 0.07145730981694887
- Agreement MAE: 0.010762522489570131
- Min provenance confidence: 0.9651104040584654
- Min direction cosine: 0.9998577803120876
- Raw oracle calls: 213512

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

## Provenance Matches

- recovered atom 1 -> gpt2.transformer.h0.mlp.c_fc[2452]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2452]->c_proj[0]
  - confidence: 0.983847691614797
  - direction cosine: 0.9999350303940073
  - norm error: 0.08869294797779315
  - bias error: 0.00037933553879443416
  - output weight error: 0.005693832565568768
  - contribution correlation: 0.999913897819637
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 7 -> gpt2.transformer.h0.mlp.c_fc[257]
  - source ids: gpt2.transformer.h0.mlp.c_fc[257]->c_proj[0]
  - confidence: 0.976678186912419
  - direction cosine: 0.9998577803120876
  - norm error: 0.198999902866003
  - bias error: 0.0038940454669909585
  - output weight error: 0.009628785066881695
  - contribution correlation: 0.9997759010832158
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 4 -> gpt2.transformer.h0.mlp.c_fc[1981]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1981]->c_proj[0]
  - confidence: 0.9765391381544166
  - direction cosine: 0.9999147542374535
  - norm error: 0.2714427823967265
  - bias error: 0.010052542173894508
  - output weight error: 0.019083912809190973
  - contribution correlation: 0.9998734818144054
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 3 -> gpt2.transformer.h0.mlp.c_fc[486]
  - source ids: gpt2.transformer.h0.mlp.c_fc[486]->c_proj[0]
  - confidence: 0.9759076378562531
  - direction cosine: 0.9999307710504219
  - norm error: 0.11112168863908778
  - bias error: 0.005263834977756232
  - output weight error: 0.006263384941049999
  - contribution correlation: 0.9998941545064284
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 0 -> gpt2.transformer.h0.mlp.c_fc[1614]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1614]->c_proj[0]
  - confidence: 0.9740600533888043
  - direction cosine: 0.9999434774228408
  - norm error: 0.28237285516198085
  - bias error: 0.006223747612286473
  - output weight error: 0.010814752244746595
  - contribution correlation: 0.9998881586903843
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 5 -> gpt2.transformer.h0.mlp.c_fc[2172]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2172]->c_proj[0]
  - confidence: 0.9728014624295126
  - direction cosine: 0.9998665909796525
  - norm error: 0.07407075405288532
  - bias error: 0.010143491246975611
  - output weight error: 0.004380816091062056
  - contribution correlation: 0.9998120253544439
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 2 -> gpt2.transformer.h0.mlp.c_fc[2053]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2053]->c_proj[0]
  - confidence: 0.9700243428070371
  - direction cosine: 0.9998832887150932
  - norm error: 0.7053306435418287
  - bias error: 0.04481847047053056
  - output weight error: 0.033502016744713925
  - contribution correlation: 0.9997701401247575
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 6 -> gpt2.transformer.h0.mlp.c_fc[137]
  - source ids: gpt2.transformer.h0.mlp.c_fc[137]->c_proj[0]
  - confidence: 0.9651104040584654
  - direction cosine: 0.9998597354519718
  - norm error: 0.35088959848420576
  - bias error: 0.004206588444486459
  - output weight error: 0.012878904808474034
  - contribution correlation: 0.9997580050871819
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.

## Conclusion

The real GPT-2 compact FFN structure gate passed. GCL recovered explicit GELU source atoms from oracle calls over real GPT-2 layer-0 weights and verified source-channel provenance against the compact source manifest.
