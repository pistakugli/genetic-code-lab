# 4096D Active-Subspace FFN Stress Benchmark

Generated: 2026-05-31T12:58:42.240Z

## Claim Scope

- Proves if pass: GCL can scale the active-subspace FFN decompiler to LLM-width synthetic 4096D GELU/SwiGLU-style channels with explicit query budget and baseline-court accounting
- Does not claim: real LLM weight recovery, production transformer hook decompilation, exact GELU/SwiGLU parameter recovery, multi-channel FFN block recovery, tokenizer/model integration, or broad SOTA status

## Summary

- Pass: true
- Teachers: 2/2
- Dimensions: 4096
- Kinds: gelu-single-channel-4096d, swiglu-two-direction-4096d
- Max MAE: 0.0003824201749357419
- Min R2: 0.9999571961681036
- Min active energy ratio: 0.9999999999999987
- Max gradient residual ratio: 3.948644492908337e-8
- Min true-direction projection score: 0.9999999999999992
- GCL MAE wins: 2/2
- Max GCL/best-baseline MAE ratio: 0.0011451291794471713
- Max raw oracle calls: 103504

## Results

### gelu-4096d-4101

- Kind: gelu-single-channel-4096d
- Dimension: 4096
- Active rank: 1
- Pass: true
- GCL MAE: 0.000008920809780524622
- GCL R2: 0.9999999490222489
- Active energy ratio: 0.9999999999999987
- Min projection score: 1.0000000000000009
- Raw oracle calls: 68336
- Best baseline: mean-output
- Best baseline MAE: 0.7044681273176859
- GCL MAE ratio vs best baseline: 0.000012663184372146487

Ranking:
- gcl-active-subspace: MAE 0.000008920809780524622, R2 0.9999999490222489, queries 68336
- mean-output: MAE 0.7044681273176859, R2 -0.0007144307347382028, queries 2800
- random-projection-local-quadratic-rank1: MAE 0.762921653863735, R2 -0.39106503943017246, queries 2800
- coordinate-head-local-quadratic-rank1: MAE 0.7636901652794472, R2 -0.468332741346942, queries 2800

### swiglu-4096d-4201

- Kind: swiglu-two-direction-4096d
- Dimension: 4096
- Active rank: 2
- Pass: true
- GCL MAE: 0.0003824201749357419
- GCL R2: 0.9999571961681036
- Active energy ratio: 1.0000000000000016
- Min projection score: 0.9999999999999992
- Raw oracle calls: 103504
- Best baseline: mean-output
- Best baseline MAE: 0.3339537423370533
- GCL MAE ratio vs best baseline: 0.0011451291794471713

Ranking:
- gcl-active-subspace: MAE 0.0003824201749357419, R2 0.9999571961681036, queries 103504
- mean-output: MAE 0.3339537423370533, R2 -0.00018610815271435932, queries 5200
- random-projection-local-quadratic-rank2: MAE 0.40850408805927557, R2 -0.2353409911909654, queries 5200
- coordinate-head-local-quadratic-rank2: MAE 0.4266069501205935, R2 -0.31933931963028406, queries 5200

## Conclusion

4096D active-subspace FFN stress gate passed for controlled GELU and SwiGLU teachers; next step is mixed multi-channel FFN blocks plus package-backed external baselines.
