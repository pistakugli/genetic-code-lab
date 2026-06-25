# NN Live Blind SwiGLU Deep Atlas

Status: **PASS**

This atlas reads all current real-model oracle-only Mistral/LLaMA-family SwiGLU decompiler artifacts and builds a layer-by-layer recovery map.

## Family Summary

- LLaMA-family: 13/28 layers, rows 30, selected 24, multi-unit selected 12, finite-precision rows 4, min R2 0.999574211683, min subspace 0.999284096699, max worst 0.120805323663.
  Layers: 0, 1, 6, 8, 9, 11, 16, 17, 19, 20, 22, 24, 25
  Buckets: early 0, 1, 6, 8; middle 9, 11, 16, 17; late 19, 20, 22, 24, 25.
- Mistral-family: 15/32 layers, rows 56, selected 43, multi-unit selected 31, finite-precision rows 13, min R2 0.995143617716, min subspace 0.999353644734, max worst 0.014497109405.
  Layers: 0, 1, 7, 8, 10, 11, 12, 16, 19, 20, 22, 24, 26, 27, 29
  Buckets: early 0, 1, 7, 8; middle 10, 11, 12, 16, 19, 20; late 22, 24, 26, 27, 29.

## Layer Atlas

| family | layer | rows | max units | worst R2 | max worst error | min subspace | finite rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaMA-family | 0 | 1 | 1 | 0.999999972452 | 0.000388099974 | 0.999999983998 | 0 |
| LLaMA-family | 1 | 5 | 8 | 0.999645886026 | 0.07539645059 | 0.999563561209 | 1 |
| LLaMA-family | 6 | 1 | 1 | 0.999999996167 | 0.000119893499 | 0.999999997455 | 0 |
| LLaMA-family | 8 | 1 | 1 | 0.999999999651 | 2.641473e-5 | 0.999999999841 | 0 |
| LLaMA-family | 9 | 1 | 1 | 0.999999997337 | 7.884152e-5 | 0.999999998569 | 0 |
| LLaMA-family | 11 | 2 | 6 | 0.999988052285 | 0.012340230225 | 0.999957400868 | 0 |
| LLaMA-family | 16 | 3 | 8 | 0.999574211683 | 0.078378745416 | 0.999284096699 | 1 |
| LLaMA-family | 17 | 1 | 1 | 0.99999999976 | 2.147516e-5 | 0.999999999878 | 0 |
| LLaMA-family | 19 | 1 | 1 | 0.999999999553 | 2.854854e-5 | 0.999999999834 | 0 |
| LLaMA-family | 20 | 5 | 8 | 0.999597610028 | 0.120805323663 | 0.999441573954 | 1 |
| LLaMA-family | 22 | 1 | 1 | 0.99999976731 | 0.00062679162 | 0.999999886275 | 0 |
| LLaMA-family | 24 | 1 | 1 | 0.999999999467 | 3.458678e-5 | 0.999999999889 | 0 |
| LLaMA-family | 25 | 1 | 1 | 0.999999998672 | 4.993587e-5 | 0.999999999413 | 0 |
| Mistral-family | 0 | 1 | 1 | 0.999957034172 | 0.000441386169 | 0.99999999976 | 0 |
| Mistral-family | 1 | 11 | 8 | 0.995222010288 | 0.010330999089 | 0.999606641022 | 3 |
| Mistral-family | 7 | 1 | 1 | 0.999992817073 | 0.000201062846 | 0.999999999849 | 0 |
| Mistral-family | 8 | 1 | 1 | 0.999999996657 | 3.742177e-6 | 0.999999999726 | 0 |
| Mistral-family | 10 | 2 | 3 | 0.996294143908 | 0.010120884384 | 0.999999987811 | 0 |
| Mistral-family | 11 | 1 | 4 | 0.995727068596 | 0.008566143276 | 0.999999972542 | 0 |
| Mistral-family | 12 | 2 | 6 | 0.997665430984 | 0.012323574241 | 0.999999121098 | 0 |
| Mistral-family | 16 | 1 | 1 | 0.999999330798 | 3.574106e-5 | 0.999999789274 | 0 |
| Mistral-family | 19 | 5 | 8 | 0.995867637063 | 0.014497109405 | 0.999440768464 | 3 |
| Mistral-family | 20 | 12 | 8 | 0.995241421111 | 0.012667784693 | 0.999353644734 | 3 |
| Mistral-family | 22 | 1 | 1 | 0.999989985506 | 0.000177402161 | 0.999999999774 | 0 |
| Mistral-family | 24 | 1 | 1 | 0.999998569306 | 8.157225e-5 | 0.999999999948 | 0 |
| Mistral-family | 26 | 2 | 5 | 0.995143617716 | 0.010051115148 | 0.999999994841 | 0 |
| Mistral-family | 27 | 1 | 2 | 0.99965005995 | 0.00170566244 | 0.999999982314 | 0 |
| Mistral-family | 29 | 1 | 1 | 0.999987784235 | 0.000331519643 | 0.999999995794 | 0 |

## Boundaries

- This is a deep multi-layer atlas over existing real-model oracle-only SwiGLU decompiler artifacts.
- It is not a claim that the entire LLM brain, every channel, or semantic hidden-neuron identity has been recovered.
- The atlas proves multi-layer selected-block recovery coverage for the covered real Mistral/LLaMA-family scalar SwiGLU oracle rows.
- If coverage gates fail, the remedy is additional oracle-only recovery/backfill runs, not lowering thresholds or hiding the gap.

