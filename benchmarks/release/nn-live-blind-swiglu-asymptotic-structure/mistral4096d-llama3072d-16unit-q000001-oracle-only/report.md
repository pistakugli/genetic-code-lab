# Live Blind SwiGLU Asymptotic Structure Court

Status: **PASS**

- regenerated: `true`
- cases: `3/3`
- model families: `llama, mistral`
- dimensions: `3072, 4096`
- units per scalar block: `16`
- output quantization step: `0.000001`
- min R2: `0.999647819803`
- max MAE: `0.000602149761`
- max worst error: `0.003997934983`
- min subspace cosine: `0.999698557097`
- min paired gate/up joint cosine: `0.998974989298`
- min paired atoms above 0.99: `16/16`
- max raw oracle calls per case: `109808`
- total raw oracle calls: `319042`
- access invariant: `true`

| case | family | dimension | layer | R2 | MAE | worst | paired min | pairs >=0.99 | calls | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mistral-layer9-output2048 | mistral | 4096 | 9 | 0.999682496442 | 5.019408e-6 | 2.731916e-5 | 0.998974989298 | 16/16 | 107696 | true |
| mistral-layer15-output2048 | mistral | 4096 | 15 | 0.999647819803 | 5.804778e-6 | 3.073278e-5 | 0.999190732959 | 16/16 | 101538 | true |
| llama-layer9-selected16 | llama | 3072 | 9 | 0.999992248559 | 0.000602149761 | 0.003997934983 | 0.999971194193 | 16/16 | 109808 | true |

