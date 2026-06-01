# GELU FFN Source-Neuron Provenance Verifier Benchmark

Generated: 2026-06-01T17:04:11.620Z

## Claim Scope

- Proves if pass: GCL can map recovered GELU atoms to exact source neuron ids/layers/indices for distinguishable source neurons, and to exact source groups for duplicate-split neurons, while recording why each mapped source reacts.
- Does not claim: Exact original-neuron identity from output-only access without provenance information, mathematical uniqueness for arbitrary neural networks, full GPT-2 FFN extraction, or arbitrary 3072-channel LLM block decompilation.

## Summary

- Pass: true
- Cases: 2
- Passed cases: 2
- Failed cases: none
- Max MAE: 0.004659687044604257
- Min R2: 0.9998833857503274
- Max model agreement MAE: 0.004659687044604257
- Min provenance confidence: 0.9050399718933542
- Duplicate-group cases: 1

## Cases

| Case | Pass | Source neurons | Source groups | Effective channels | MAE | R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| distinct-source-neurons-24d | true | 3 | 3 | 3 | 0.004659687044604257 | 0.9998833857503274 |
| duplicate-split-source-group-24d | true | 3 | 2 | 2 | 0.0013510293868686397 | 0.9999796455622774 |

## Provenance Matches

### distinct-source-neurons-24d

- recovered atom 2 -> transformer.h0.mlp.c_fc[203]
  - source ids: transformer.h0.mlp.c_fc[203]
  - layer: transformer.h0.mlp.c_fc
  - indices: 203
  - duplicate split: false
  - confidence: 0.9741690737839342
  - direction cosine: 0.9998822686829256
  - contribution correlation: 0.999800149060857
  - contribution MAE: 0.0023374458526689806
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
  - formula: `0.370000 * GELU(0.0375864*x0 + -0.291008*x1 + -0.132409*x2 + -0.0230113*x3 + -0.0266178*x4 + 0.193853*x5 + -0.194418*x6 + 0.0679737*x7 + -0.0556175*x8 + -0.157909*x9 + 0.168427*x10 + 0.210154*x11 + 0.0967100*x12 + -0.250109*x13 + -0.231166*x14 + -0.260524*x15 + 0.185186*x16 + -0.187472*x17 + -0.125398*x18 + -0.0147503*x19 + 0.0487101*x20 + 0.204147*x21 + -0.244345*x22 + -0.0679624*x23 + -0.0800000)`
- recovered atom 1 -> transformer.h0.mlp.c_fc[42]
  - source ids: transformer.h0.mlp.c_fc[42]
  - layer: transformer.h0.mlp.c_fc
  - indices: 42
  - duplicate split: false
  - confidence: 0.9393093516535741
  - direction cosine: 0.9998627839608416
  - contribution correlation: 0.999785955390672
  - contribution MAE: 0.006072548635663222
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
  - formula: `-0.610000 * GELU(0.291536*x0 + 0.0417953*x1 + 0.154285*x2 + -0.0408485*x3 + 0.134184*x4 + -0.540100*x5 + -0.0406162*x6 + -0.00794859*x7 + -0.0964815*x8 + -0.386751*x9 + -0.240952*x10 + 0.193679*x11 + -0.103227*x12 + -0.287139*x13 + -0.134867*x14 + -0.134131*x15 + -0.0255442*x16 + 0.0188200*x17 + 0.127824*x18 + 0.106024*x19 + -0.0718423*x20 + -0.0295009*x21 + 0.206931*x22 + -0.191788*x23 + 0.240000)`
- recovered atom 0 -> transformer.h0.mlp.c_fc[17]
  - source ids: transformer.h0.mlp.c_fc[17]
  - layer: transformer.h0.mlp.c_fc
  - indices: 17
  - duplicate split: false
  - confidence: 0.9050399718933542
  - direction cosine: 0.9999558958819256
  - contribution correlation: 0.9998481935850544
  - contribution MAE: 0.012496861963984219
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
  - formula: `0.820000 * GELU(-0.260353*x0 + 0.143909*x1 + 0.299591*x2 + 0.0609103*x3 + -0.0797647*x4 + 0.433482*x5 + -0.281471*x6 + 0.0431322*x7 + -0.480023*x8 + -0.124609*x9 + -0.108308*x10 + 0.0652683*x11 + -0.260419*x12 + -0.110930*x13 + 0.116088*x14 + -0.0800916*x15 + -0.294972*x16 + 0.180539*x17 + -0.369868*x18 + 0.375451*x19 + -0.0681693*x20 + -0.290413*x21 + -0.00287484*x22 + -0.294563*x23 + -0.200000)`

### duplicate-split-source-group-24d

- recovered atom 1 -> transformer.h1.mlp.c_fc[77]
  - source ids: transformer.h1.mlp.c_fc[77]
  - layer: transformer.h1.mlp.c_fc
  - indices: 77
  - duplicate split: false
  - confidence: 0.986851479827466
  - direction cosine: 0.9999454019664195
  - contribution correlation: 0.9999287643327719
  - contribution MAE: 0.0027428336009453127
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
  - formula: `-0.520000 * GELU(-0.301012*x0 + 0.101682*x1 + -0.0112128*x2 + -0.128510*x3 + -0.106372*x4 + 0.185919*x5 + 0.000837430*x6 + -0.292781*x7 + 0.302300*x8 + -0.0260534*x9 + -0.118248*x10 + -0.239489*x11 + -0.137375*x12 + -0.319435*x13 + 0.0696695*x14 + 0.149986*x15 + -0.166130*x16 + -0.349455*x17 + -0.0299015*x18 + 0.247101*x19 + -0.0350210*x20 + 0.0945310*x21 + 0.0824913*x22 + -0.143212*x23 + 0.140000)`
- recovered atom 0 -> transformer.h1.mlp.c_fc[8+9]
  - source ids: transformer.h1.mlp.c_fc[8], transformer.h1.mlp.c_fc[9]
  - layer: transformer.h1.mlp.c_fc
  - indices: 8, 9
  - duplicate split: true
  - confidence: 0.9633454870448102
  - direction cosine: 0.9999866273187501
  - contribution correlation: 0.9999667708378319
  - contribution MAE: 0.004240946958794905
  - why: This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.
  - formula: `0.740000 * GELU(0.103239*x0 + -0.287191*x1 + 0.312123*x2 + 0.0259996*x3 + 0.181312*x4 + -0.120578*x5 + -0.000873848*x6 + 0.103924*x7 + 0.196374*x8 + -0.0660505*x9 + 0.164969*x10 + -0.0969681*x11 + -0.109575*x12 + -0.101816*x13 + 0.197363*x14 + 0.160529*x15 + 0.0646075*x16 + -0.0652801*x17 + 0.319323*x18 + 0.123538*x19 + 0.681969*x20 + 0.137921*x21 + -0.265434*x22 + 0.0786851*x23 + -0.180000)`

## Conclusion

The provenance verifier gate passed. GCL can now make source-neuron identity claims only in provenance mode, with exact source ids/groups and why-reacts evidence.
