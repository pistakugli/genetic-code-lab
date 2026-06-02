# GELU FFN Structure Finder Benchmark

Generated: 2026-06-02T10:01:07.782Z

## Claim Scope

- Proves if pass: GCL can recover explicit GELU atom structure for controlled high-dimensional one/two-channel FFN teachers, including directions and approximate atom parameters
- Does not claim: full GPT-2 FFN extraction, arbitrary channel count recovery, exact parameter identity under non-identifiability, SwiGLU recovery, or broad SOTA status

## Summary

- Pass: true
- Teachers: 2/2
- Max MAE: 0.002229589390805771
- Min R2: 0.9999894428635375
- Min direction projection: 0.9999995030788708
- Max weight-norm error: 0.044204547411414996
- Max bias error: 0.006098221054376296
- Max raw oracle calls: 6088

## Results

### gelu-structure-single-96d

- Pass: true
- Dimension: 96
- Channel count: 1
- MAE: 0.00022745631345987545
- R2: 0.9999998657083056
- Min direction projection: 0.9999999855706445
- Max weight-norm error: 0.00038711121745582666
- Max bias error: 0.0005093207257401422
- Raw oracle calls: 4220

### gelu-structure-two-channel-96d

- Pass: true
- Dimension: 96
- Channel count: 2
- MAE: 0.002229589390805771
- R2: 0.9999894428635375
- Min direction projection: 0.9999995030788708
- Max weight-norm error: 0.044204547411414996
- Max bias error: 0.006098221054376296
- Raw oracle calls: 6088

## Conclusion

GELU FFN structural finder gate passed for controlled single/two-channel high-dimensional teachers; next step is GPT-2 top-K structural stress and rank selection.
