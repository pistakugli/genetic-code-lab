# 6D ND Recovered-Adjacency Geometry Sweep

Generated: 2026-05-28T11:09:10.379Z

## Claim Scope

- Proves if pass: 6D recovered-adjacency geometry survives a broader clean/noisy hard-profile sweep with adaptive probe-distribution selection and explicit failure ledger
- Does not claim: SOTA, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, arbitrary production API extraction, or superiority over external extraction algorithms

## Summary

- Pass: true
- Teachers passed: 6/6
- Candidate count: 12
- Rejected candidates: 6
- Rejected failed candidates: 4
- Clean/noisy teachers: 2/4
- Dimensions: 6
- Profiles: clean, cancellation, near-parallel, low-amplitude, weak-support
- Min region count: 57
- Min adjacency edges: 374
- Min largest-component ratio: 0.9824561403508771
- Max normalized projection RMSE: 0.939255366086159
- Max MAE: 0.09099788983192539
- Min R2: 0.9674291733376437
- Max selected raw oracle calls per teacher: 1989000
- Total selected raw oracle calls: 8105500
- Total search raw oracle calls: 19220500

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
- Candidates evaluated: 3
- Rejected failures: 2
- Region count: 71
- Geometry pass: true
- Passing edges: 374
- GCL MAE: 0.013233009749140622
- GCL R2: 0.9674291733376437
- Selected raw oracle calls: 1989000
- Search raw oracle calls: 5967000

### low-amplitude-6d-634

- Pass: true
- Profile: low-amplitude
- Noisy oracle: true
- Selected candidate: low-amplitude-seed-460001
- Selected random seed: 460001
- Candidates evaluated: 3
- Rejected failures: 2
- Region count: 57
- Geometry pass: true
- Passing edges: 485
- GCL MAE: 0.030383372680069527
- GCL R2: 0.9925207123486528
- Selected raw oracle calls: 1755000
- Search raw oracle calls: 5265000

### weak-support-6d-911

- Pass: true
- Profile: weak-support
- Noisy oracle: true
- Selected candidate: weak-support-seed-360001
- Selected random seed: 360001
- Candidates evaluated: 3
- Rejected failures: 0
- Region count: 68
- Geometry pass: true
- Passing edges: 566
- GCL MAE: 0.03784733737640974
- GCL R2: 0.9892346155670978
- Selected raw oracle calls: 1813500
- Search raw oracle calls: 5440500

## Next Step

6D sweep passed. Next valid step is external piecewise-linear extraction baselines and then an explicit release-gate promotion decision.
