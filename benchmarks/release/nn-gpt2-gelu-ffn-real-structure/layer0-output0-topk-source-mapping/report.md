# Real GPT-2 GELU FFN Structure Benchmark

Generated: 2026-06-03T06:32:46.895Z

## Claim Scope

- Proves if pass: GCL can recover explicit GELU atoms from a real GPT-2 layer-0 FFN top-K output-channel slice and map them to exact GPT-2 source channels through provenance verification.
- Does not claim: full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary tokenizer/model execution, output-only original-neuron identity without source manifest/hooks, or broad SOTA status.

## Fixture

- Source fixture type: gpt2-layer0-ffn-full-output-channel-0
- Source fixture SHA-256: ae9a59434c1e25bf4921c31c8c6ebd49d3a01c64aebf16119952580010f532dd
- Dimension: 768
- Output channel: 0
- Top-K: 3
- Selected by: abs(w_proj[j]) * ||W_fc_T[j]||

## Summary

- Pass: true
- Effective channels: 3
- Source groups: 3
- R2: 0.9942876980025065
- MAE: 0.04964509934587569
- Worst error: 0.1585760339175079
- Agreement MAE: 0.04964509934587569
- Min provenance confidence: 0.7918710775184039
- Min direction cosine: 0.9998877789545968
- Raw oracle calls: 116988

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

## Provenance Matches

- recovered atom 1 -> gpt2.transformer.h0.mlp.c_fc[2452]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2452]->c_proj[0]
  - confidence: 0.8324176323426415
  - direction cosine: 0.9998877789545968
  - norm error: 2.1049883753970846
  - bias error: 0.16750074633266548
  - output weight error: 0.3022186143713579
  - contribution correlation: 0.9981233074627494
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 0 -> gpt2.transformer.h0.mlp.c_fc[1614]
  - source ids: gpt2.transformer.h0.mlp.c_fc[1614]->c_proj[0]
  - confidence: 0.8199867868268813
  - direction cosine: 0.9999544880659922
  - norm error: 3.491641744210471
  - bias error: 0.16071381955684771
  - output weight error: 0.38001658630075474
  - contribution correlation: 0.9971792590867726
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
- recovered atom 2 -> gpt2.transformer.h0.mlp.c_fc[2053]
  - source ids: gpt2.transformer.h0.mlp.c_fc[2053]->c_proj[0]
  - confidence: 0.7918710775184039
  - direction cosine: 0.9999221619898583
  - norm error: 3.6348294115458413
  - bias error: 0.29673002904093215
  - output weight error: 0.36329930901467733
  - contribution correlation: 0.9963984417139329
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.

## Conclusion

The real GPT-2 compact FFN structure gate passed. GCL recovered explicit GELU source atoms from oracle calls over real GPT-2 layer-0 weights and verified source-channel provenance against the compact source manifest.
