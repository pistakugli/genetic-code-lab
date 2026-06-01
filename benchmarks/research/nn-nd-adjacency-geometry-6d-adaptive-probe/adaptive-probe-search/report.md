# Adaptive 6D ND Recovered-Adjacency Geometry Probe Search

Generated: 2026-05-31T15:34:10.096Z

## Claim Scope

- Proves if pass: 6D no longer depends on a manually fixed near-parallel probe seed in this benchmark: the search evaluates deterministic probe distributions, records rejected failures, and selects the best passing candidate under explicit selected/search query budgets
- Does not claim: 6D release readiness, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, arbitrary production API extraction, or superiority over external extraction algorithms

## Summary

- Pass: true
- Teachers passed: 4/4
- Candidate count: 9
- Rejected candidates: 5
- Rejected failed candidates: 5
- Clean/noisy teachers: 2/2
- Dimensions: 6
- Profiles: clean, cancellation, near-parallel
- Min region count: 63
- Min adjacency edges: 374
- Min largest-component ratio: 0.9850746268656716
- Max normalized projection RMSE: 0.939255366086159
- Max MAE: 0.09099788983192539
- Min R2: 0.9674291733376437
- Max selected raw oracle calls per teacher: 1989000
- Total selected raw oracle calls: 4537000
- Total search raw oracle calls: 14482000

## Results

### clean-6d-29

- Pass: true
- Profile: clean
- Noisy oracle: false
- Selected candidate: clean-default
- Selected random seed: 360561
- Candidates evaluated: 1
- Rejected failures: 0
- Region count: 67
- Geometry pass: true
- Passing edges: 691
- GCL MAE: 0.01875518669949688
- GCL R2: 0.9977404171231974
- Selected raw oracle calls: 338000
- Search raw oracle calls: 338000

### clean-6d-83

- Pass: true
- Profile: clean
- Noisy oracle: false
- Selected candidate: clean-default
- Selected random seed: 360615
- Candidates evaluated: 1
- Rejected failures: 0
- Region count: 63
- Geometry pass: true
- Passing edges: 660
- GCL MAE: 0.030638484352114075
- GCL R2: 0.9972710063911236
- Selected raw oracle calls: 338000
- Search raw oracle calls: 338000

### cancellation-6d-717

- Pass: true
- Profile: cancellation
- Noisy oracle: true
- Selected candidate: cancellation-default
- Selected random seed: 361249
- Candidates evaluated: 1
- Rejected failures: 0
- Region count: 110
- Geometry pass: true
- Passing edges: 1079
- GCL MAE: 0.09099788983192539
- GCL R2: 0.9928485429486841
- Selected raw oracle calls: 1872000
- Search raw oracle calls: 1872000

### near-parallel-6d-805

- Pass: true
- Profile: near-parallel
- Noisy oracle: true
- Selected candidate: near-parallel-seed-360001
- Selected random seed: 360001
- Candidates evaluated: 6
- Rejected failures: 5
- Region count: 71
- Geometry pass: true
- Passing edges: 374
- GCL MAE: 0.013233009749140622
- GCL R2: 0.9674291733376437
- Selected raw oracle calls: 1989000
- Search raw oracle calls: 11934000

## Next Step

Adaptive 6D probe search passed. The next valid step is a broader 6D release-candidate sweep with multiple teachers per hard profile and external extraction baselines.
