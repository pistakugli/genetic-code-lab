# NN Live Blind SwiGLU Deep Atlas Backfill

Status: **PASS**

Mode: missing
Requested cases: 0
Passed cases: 0
Skipped existing cases: 0
Existing backfill artifacts: 32
Passed backfill artifacts: 32
Coverage complete: true

## Coverage Audit

| family | observed layers | expected layers | coverage | missing layers |
| --- | ---: | ---: | ---: | --- |
| Mistral-family | 32 | 32 | 1 | - |
| LLaMA-family | 28 | 28 | 1 | - |

## Results

| family | layer | unit | pass | R2 | subspace | report |
| --- | ---: | ---: | --- | ---: | ---: | --- |

## Existing Backfill Artifacts

| model | layer | units | pass | R2 | subspace | raw calls | report |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| llama | 10 | 4728 | true | 0.999999999472969 | 0.9999999997800794 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer10-unit4728-atlas-backfill-oracle-only/report.json |
| llama | 12 | 3678 | true | 0.9999999997416598 | 0.9999999999060746 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer12-unit3678-atlas-backfill-oracle-only/report.json |
| llama | 13 | 7249 | true | 0.9999999957907597 | 0.9999999984744686 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer13-unit7249-atlas-backfill-oracle-only/report.json |
| llama | 14 | 2628 | true | 0.9999999996027115 | 0.9999999996236987 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer14-unit2628-atlas-backfill-oracle-only/report.json |
| llama | 15 | 6199 | true | 0.9999999997445771 | 0.9999999999263673 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-unit6199-atlas-backfill-oracle-only/report.json |
| llama | 18 | 528 | true | 0.9999999996662983 | 0.9999999999048358 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer18-unit528-atlas-backfill-oracle-only/report.json |
| llama | 2 | 736 | true | 0.9999999988973046 | 0.9999999992987361 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer2-unit736-atlas-backfill-oracle-only/report.json |
| llama | 21 | 3049 | true | 0.9999999993957612 | 0.9999999997081972 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer21-unit3049-atlas-backfill-oracle-only/report.json |
| llama | 23 | 1999 | true | 0.9999999991869707 | 0.9999999995642418 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer23-unit1999-atlas-backfill-oracle-only/report.json |
| llama | 26 | 4520 | true | 0.999999988217913 | 0.9999999926969814 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer26-unit4520-atlas-backfill-oracle-only/report.json |
| llama | 27 | 8091 | true | 0.9999984142667003 | 0.9999990739743683 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer27-unit8091-atlas-backfill-oracle-only/report.json |
| llama | 3 | 4307 | true | 0.9999999995206063 | 0.9999999998688374 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer3-unit4307-atlas-backfill-oracle-only/report.json |
| llama | 4 | 7878 | true | 0.9999999994115103 | 0.9999999997528787 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer4-unit7878-atlas-backfill-oracle-only/report.json |
| llama | 5 | 3257 | true | 0.9999999997268147 | 0.9999999999031076 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer5-unit3257-atlas-backfill-oracle-only/report.json |
| llama | 7 | 2207 | true | 0.9999999847064683 | 0.9999999911482416 | 14400 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer7-unit2207-atlas-backfill-oracle-only/report.json |
| mistral | 13 | 5720 | true | 0.9999739685264046 | 0.9999999990897485 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer13-unit5720-atlas-backfill-oracle-only/report.json |
| mistral | 14 | 13639 | true | 0.99999323833154 | 0.9999999984893235 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer14-unit13639-atlas-backfill-oracle-only/report.json |
| mistral | 15 | 7222 | true | 0.9999986929164764 | 0.9999999996548515 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-unit7222-atlas-backfill-oracle-only/report.json |
| mistral | 17 | 8724 | true | 0.9999999943178767 | 0.9999999992068026 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer17-unit8724-atlas-backfill-oracle-only/report.json |
| mistral | 18 | 2307 | true | 0.9998851220290215 | 0.999999999878048 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer18-unit2307-atlas-backfill-oracle-only/report.json |
| mistral | 2 | 4627 | true | 0.999999999196481 | 0.9999999996247264 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer2-unit4627-atlas-backfill-oracle-only/report.json |
| mistral | 21 | 11728 | true | 0.9998031673146996 | 0.999999998704192 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer21-unit11728-atlas-backfill-oracle-only/report.json |
| mistral | 23 | 13230 | true | 0.9999970332853755 | 0.9999999998108878 | 60512 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer23-unit13230-atlas-backfill-oracle-only/report.json |
| mistral | 25 | 396 | true | 0.9999998694949087 | 0.9999999304817423 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer25-unit396-atlas-backfill-oracle-only/report.json |
| mistral | 28 | 9817 | true | 0.9999972610027758 | 0.9999999997220904 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer28-unit9817-atlas-backfill-oracle-only/report.json |
| mistral | 3 | 12546 | true | 0.9999999994572247 | 0.9999999998123645 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer3-unit12546-atlas-backfill-oracle-only/report.json |
| mistral | 30 | 11319 | true | 0.9999999366494735 | 0.9999999999121852 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer30-unit11319-atlas-backfill-oracle-only/report.json |
| mistral | 31 | 4902 | true | 0.9999869953814583 | 0.9999999914365378 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer31-unit4902-atlas-backfill-oracle-only/report.json |
| mistral | 4 | 6129 | true | 0.9999939208274271 | 0.9999999998609898 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer4-unit6129-atlas-backfill-oracle-only/report.json |
| mistral | 5 | 14048 | true | 0.9999999992204276 | 0.999999999747602 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer5-unit14048-atlas-backfill-oracle-only/report.json |
| mistral | 6 | 7631 | true | 0.9999999991463636 | 0.9999999996818859 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer6-unit7631-atlas-backfill-oracle-only/report.json |
| mistral | 9 | 2716 | true | 0.9999999929199335 | 0.9999999974479217 | 18496 | benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-unit2716-atlas-backfill-oracle-only/report.json |

## Boundaries

- This runner backfills missing deep-atlas layers with real-model oracle-only single-unit SwiGLU recoveries.
- Default mode is missing-layer gap fill. Set GCL_ATLAS_BACKFILL_MODE=all with GCL_ATLAS_BACKFILL_FORCE=1 for a full deterministic layer rerun.
- It expands layer coverage; it does not claim all-channel FFN extraction.
- If a case fails, the required action is to repair that case or add a stronger oracle-only recovery configuration, not to hide the layer gap.