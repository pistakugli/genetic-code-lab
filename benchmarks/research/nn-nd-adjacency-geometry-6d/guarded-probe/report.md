# 6D ND Recovered-Adjacency Geometry Guarded Probe

Generated: 2026-05-28T10:54:06.275Z

## Claim Scope

- Proves if pass: 6D can enter a later release-candidate sweep under this controlled clean/noisy hard-profile setup
- Does not claim: 6D release readiness, probe-seed robustness, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, arbitrary production API extraction, or superiority over external extraction algorithms
- Probe seed note: During development the noisy near-parallel 6D case was probe-seed sensitive: several deterministic sampler seeds failed the R2 gate while seed 360001 passed. This is kept as the next hardening target, not hidden as robustness.

## Summary

- Pass: true
- Teachers passed: 4/4
- Clean/noisy teachers: 2/2
- Dimensions: 6
- Profiles: clean, cancellation, near-parallel
- Min region count: 63
- Min adjacency edges: 374
- Min largest-component ratio: 0.9850746268656716
- Max normalized projection RMSE: 0.939255366086159
- Max MAE: 0.09099788983192539
- Min R2: 0.9674291733376437
- Max raw oracle calls per teacher: 1989000
- Total raw oracle calls: 4537000

## Results

### clean-6d-29

- Pass: true
- Profile: clean
- Noisy oracle: false
- Region count: 67
- Deep-only region estimate: 62
- Retained sample ratio: 0.9986153846153846
- Geometry pass: true
- Passing edges: 691
- Largest-component ratio: 0.9850746268656716
- Max normalized projection RMSE: 0.4651599530073067
- GCL MAE: 0.01875518669949688
- GCL R2: 0.9977404171231974
- Raw oracle calls: 338000

### clean-6d-83

- Pass: true
- Profile: clean
- Noisy oracle: false
- Region count: 63
- Deep-only region estimate: 58
- Retained sample ratio: 0.9985
- Geometry pass: true
- Passing edges: 660
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.5215937518246355
- GCL MAE: 0.030638484352114075
- GCL R2: 0.9972710063911236
- Raw oracle calls: 338000

### cancellation-6d-717

- Pass: true
- Profile: cancellation
- Noisy oracle: true
- Region count: 110
- Deep-only region estimate: 105
- Retained sample ratio: 0.9450625
- Geometry pass: true
- Passing edges: 1079
- Largest-component ratio: 0.990909090909091
- Max normalized projection RMSE: 0.6276084221566199
- GCL MAE: 0.09099788983192539
- GCL R2: 0.9928485429486841
- Raw oracle calls: 1872000

### near-parallel-6d-805

- Pass: true
- Profile: near-parallel
- Noisy oracle: true
- Region count: 71
- Deep-only region estimate: 66
- Retained sample ratio: 0.9712352941176471
- Geometry pass: true
- Passing edges: 374
- Largest-component ratio: 0.9859154929577465
- Max normalized projection RMSE: 0.939255366086159
- GCL MAE: 0.013233009749140622
- GCL R2: 0.9674291733376437
- Raw oracle calls: 1989000

## Next Step

6D guarded probe passed. The next valid step is a broader 6D release-candidate sweep with more seeds and external extraction baselines.
