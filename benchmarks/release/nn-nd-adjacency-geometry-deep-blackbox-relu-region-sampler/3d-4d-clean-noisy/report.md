# ND Recovered-Adjacency Geometry for Deep Black-Box ReLU Region Samplers

Generated: 2026-05-28T05:11:13.482Z

## Claim Scope

- Proves: benchmark-scoped ND adjacency geometry recovered from affine-region assignments using cross-region contact witnesses and affine-equality hyperplanes
- Does not claim: topology-perfect full ND cell complex recovery, unique hidden-layer identity, exact layer attribution, arbitrary production API extraction, or globally optimal query complexity

## Summary

- Pass: true
- Teachers passed: 4/4
- Clean/noisy teachers: 2/2
- Dimensions: 3, 4
- Profiles: clean, low-amplitude, cancellation
- Min adjacency edges: 62
- Min largest-component ratio: 0.9622641509433962
- Max normalized projection RMSE: 0.9926122288520228
- Max MAE: 0.035446586421537286
- Min R2: 0.9973558277920783
- Total raw oracle calls: 805100

## Results

### clean-3d-29

- Pass: true
- Dimension: 3
- Profile: clean
- Noisy oracle: false
- Region count: 32
- Deep-only region estimate: 27
- Retained sample ratio: 0.9961538461538462
- Geometry pass: true
- Candidate pairs: 161
- Passing edges: 126
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.49923920610288597
- GCL MAE: 0.006121471649271745
- GCL R2: 0.9998941437795155
- Raw oracle calls: 45500

### clean-4d-29

- Pass: true
- Dimension: 4
- Profile: clean
- Noisy oracle: false
- Region count: 53
- Deep-only region estimate: 48
- Retained sample ratio: 0.9988
- Geometry pass: true
- Candidate pairs: 374
- Passing edges: 268
- Largest-component ratio: 0.9622641509433962
- Max normalized projection RMSE: 0.5477145791267598
- GCL MAE: 0.00900537556182865
- GCL R2: 0.9987838435692198
- Raw oracle calls: 90000

### low-amplitude-3d-634

- Pass: true
- Dimension: 3
- Profile: low-amplitude
- Noisy oracle: true
- Region count: 17
- Deep-only region estimate: 12
- Retained sample ratio: 0.9919444444444444
- Geometry pass: true
- Candidate pairs: 87
- Passing edges: 62
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.9926122288520228
- GCL MAE: 0.016637998526256872
- GCL R2: 0.9973558277920783
- Raw oracle calls: 378000

### cancellation-4d-717

- Pass: true
- Dimension: 4
- Profile: cancellation
- Noisy oracle: true
- Region count: 40
- Deep-only region estimate: 35
- Retained sample ratio: 0.9808333333333333
- Geometry pass: true
- Candidate pairs: 415
- Passing edges: 261
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.5949886728316554
- GCL MAE: 0.035446586421537286
- GCL R2: 0.9988312313620711
- Raw oracle calls: 291600

