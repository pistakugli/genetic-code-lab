# 4096D Active-Subspace FFN Stress Benchmark

Generated: 2026-06-03T11:02:28.175Z

## Claim Scope

- Proves if pass: GCL can scale the active-subspace FFN decompiler to LLM-width synthetic 4096D GELU/SwiGLU-style channels with explicit query budget and baseline-court accounting
- Does not claim: real LLM weight recovery, production transformer hook decompilation, exact GELU/SwiGLU parameter recovery, multi-channel FFN block recovery, tokenizer/model integration, or broad SOTA status

## Summary

- Pass: true
- Teachers: 2/2
- Dimensions: 4096
- Kinds: gelu-single-channel-4096d, swiglu-two-direction-4096d
- Max MAE: 0.0008334928131969248
- Min R2: 0.9998586505688375
- Min active energy ratio: 0.9999999998896398
- Max gradient residual ratio: 0.00006586831900720046
- Min true-direction projection score: 0.9999999998732209
- GCL MAE wins: 2/2
- Max GCL/best-baseline MAE ratio: 0.0024958331275584146
- Max raw oracle calls: 103504

## Results

### gelu-4096d-4101

- Kind: gelu-single-channel-4096d
- Dimension: 4096
- Active rank: 1
- Pass: true
- GCL MAE: 0.000028990563937732265
- GCL R2: 0.9999998907940526
- Active energy ratio: 0.9999999998896398
- Min projection score: 0.9999999998732209
- Raw oracle calls: 68336
- Best baseline: mean-output
- Best baseline MAE: 0.7044681273176859
- GCL MAE ratio vs best baseline: 0.000041152413875864

Ranking:
- gcl-active-subspace: MAE 0.000028990563937732265, R2 0.9999998907940526, queries 68336
- mean-output: MAE 0.7044681273176859, R2 -0.0007144307347382028, queries 2800
- random-projection-local-quadratic-rank1: MAE 0.762921653863735, R2 -0.39106503943017246, queries 2800
- coordinate-head-local-quadratic-rank1: MAE 0.7636901652794472, R2 -0.468332741346942, queries 2800

### swiglu-4096d-4201

- Kind: swiglu-two-direction-4096d
- Dimension: 4096
- Active rank: 2
- Pass: true
- GCL MAE: 0.0008334928131969248
- GCL R2: 0.9998586505688375
- Active energy ratio: 0.9999999998929874
- Min projection score: 0.999999999992763
- Raw oracle calls: 103504
- Best baseline: mean-output
- Best baseline MAE: 0.3339537423370533
- GCL MAE ratio vs best baseline: 0.0024958331275584146

Ranking:
- gcl-active-subspace: MAE 0.0008334928131969248, R2 0.9998586505688375, queries 103504
- mean-output: MAE 0.3339537423370533, R2 -0.00018610815271435932, queries 5200
- random-projection-local-quadratic-rank2: MAE 0.40850408805927557, R2 -0.2353409911909654, queries 5200
- coordinate-head-local-quadratic-rank2: MAE 0.4266069501205935, R2 -0.31933931963028406, queries 5200

## Conclusion

4096D active-subspace FFN stress gate passed for controlled GELU and SwiGLU teachers; next step is mixed multi-channel FFN blocks plus package-backed external baselines.
