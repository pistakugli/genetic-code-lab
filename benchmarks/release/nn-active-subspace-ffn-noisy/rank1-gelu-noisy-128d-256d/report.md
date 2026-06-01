# Noisy Active-Subspace FFN Decompiler Benchmark

Generated: 2026-06-01T16:16:52.878Z

## Claim Scope

- Proves if pass: GCL can recover a high-dimensional rank-1 smooth FFN active direction and faithful surrogate from noisy oracle calls under explicit query budgets
- Does not claim: real LLM decompilation, arbitrary stochastic API extraction, exact GELU parameter recovery, mixed multi-channel FFN block recovery, or broad SOTA status

## Summary

- Pass: true
- Teachers: 2/2
- Dimensions: 128, 256
- Noise sigma: 0.01
- Max MAE: 0.04567550692434054
- Min R2: 0.993172477248586
- Min projection score: 0.9980221387236469
- Max raw oracle calls: 8720
- GCL MAE wins: 2/2
- Max GCL/best-baseline MAE ratio: 0.06368152554857574

## Results

### gelu-rank1-noisy-128d

- Dimension: 128
- Noise sigma: 0.01
- Pass: true
- GCL MAE: 0.03783156658426089
- GCL R2: 0.9955477269153223
- Direction projection score: 0.9987336234787271
- Direction estimator: finite-difference
- Finite-difference step: 0.6000000000000001
- Raw oracle calls: 4360
- Best baseline: mean-noisy-output
- Best baseline MAE: 0.7204731363877511
- GCL MAE ratio vs best baseline: 0.05250933681432965

Ranking:
- gcl-active-subspace-noisy: MAE 0.03783156658426089, R2 0.9955477269153223, queries 4360
- mean-noisy-output: MAE 0.7204731363877511, R2 -0.0000010325277555178758, queries 1800
- coordinate-head-local-quadratic-rank1: MAE 0.7940559510975882, R2 -0.5115144969552861, queries 1800
- random-projection-local-quadratic-rank1: MAE 0.8417978073241266, R2 -0.6613612070971013, queries 1800

### gelu-rank1-noisy-256d

- Dimension: 256
- Noise sigma: 0.01
- Pass: true
- GCL MAE: 0.04567550692434054
- GCL R2: 0.993172477248586
- Direction projection score: 0.9980221387236469
- Direction estimator: finite-difference
- Finite-difference step: 0.6000000000000001
- Raw oracle calls: 8720
- Best baseline: mean-noisy-output
- Best baseline MAE: 0.7172489435652674
- GCL MAE ratio vs best baseline: 0.06368152554857574

Ranking:
- gcl-active-subspace-noisy: MAE 0.04567550692434054, R2 0.993172477248586, queries 8720
- mean-noisy-output: MAE 0.7172489435652674, R2 -0.001383186018335314, queries 3600
- random-projection-local-quadratic-rank1: MAE 0.7709491664463993, R2 -0.5207932882773343, queries 3600
- coordinate-head-local-quadratic-rank1: MAE 0.8097672080970248, R2 -0.5753649542317552, queries 3600

## Conclusion

Noisy active-subspace FFN gate passed for controlled 128D/256D rank-1 GELU teachers; next step is rank-adaptive mixed-channel noisy FFN recovery.
