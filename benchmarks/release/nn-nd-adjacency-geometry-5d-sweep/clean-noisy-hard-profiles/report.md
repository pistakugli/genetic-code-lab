# 5D ND Recovered-Adjacency Geometry Sweep

Generated: 2026-06-02T19:39:43.527Z

## Claim Scope

- Proves: benchmark-scoped 5D clean/noisy ND adjacency geometry recovered from affine-region assignments across multiple deterministic folded-ReLU hard profiles
- Does not claim: 6D readiness, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, arbitrary production API extraction, or superiority over all model-extraction algorithms

## Summary

- Pass: true
- Teachers passed: 6/6
- Clean/noisy teachers: 2/4
- Dimensions: 5
- Profiles: clean, low-amplitude, cancellation, near-parallel, weak-support
- Min region count: 57
- Min adjacency edges: 462
- Min largest-component ratio: 0.9672131147540983
- Max normalized projection RMSE: 1.2786982674727818
- Max MAE: 0.07984871493240715
- Min R2: 0.985430220526229
- Max raw oracle calls per teacher: 1188000
- Total raw oracle calls: 5049000

## Results

### clean-5d-29

- Pass: true
- Dimension: 5
- Profile: clean
- Noisy oracle: false
- Region count: 61
- Deep-only region estimate: 56
- Retained sample ratio: 0.9981666666666666
- Geometry pass: true
- Candidate pairs: 697
- Passing edges: 493
- Largest-component ratio: 0.9836065573770492
- Max normalized projection RMSE: 0.546750949761767
- GCL MAE: 0.01810015802661558
- GCL R2: 0.9979180365142903
- Raw oracle calls: 198000

### clean-5d-83

- Pass: true
- Dimension: 5
- Profile: clean
- Noisy oracle: false
- Region count: 61
- Deep-only region estimate: 56
- Retained sample ratio: 0.9986666666666667
- Geometry pass: true
- Candidate pairs: 682
- Passing edges: 462
- Largest-component ratio: 0.9672131147540983
- Max normalized projection RMSE: 0.6748049258380523
- GCL MAE: 0.010573115880095154
- GCL R2: 0.9984496109441349
- Raw oracle calls: 198000

### low-amplitude-5d-634

- Pass: true
- Dimension: 5
- Profile: low-amplitude
- Noisy oracle: true
- Region count: 57
- Deep-only region estimate: 52
- Retained sample ratio: 0.9891666666666666
- Geometry pass: true
- Candidate pairs: 670
- Passing edges: 464
- Largest-component ratio: 1
- Max normalized projection RMSE: 1.2786982674727818
- GCL MAE: 0.02346648239561803
- GCL R2: 0.9951919562181445
- Raw oracle calls: 1188000

### cancellation-5d-717

- Pass: true
- Dimension: 5
- Profile: cancellation
- Noisy oracle: true
- Region count: 107
- Deep-only region estimate: 102
- Retained sample ratio: 0.9586666666666667
- Geometry pass: true
- Candidate pairs: 1495
- Passing edges: 869
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.690166306553455
- GCL MAE: 0.07984871493240715
- GCL R2: 0.9927793418039836
- Raw oracle calls: 1188000

### near-parallel-5d-805

- Pass: true
- Dimension: 5
- Profile: near-parallel
- Noisy oracle: true
- Region count: 66
- Deep-only region estimate: 61
- Retained sample ratio: 0.9850833333333333
- Geometry pass: true
- Candidate pairs: 1058
- Passing edges: 599
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.7912590248310803
- GCL MAE: 0.03759163875066559
- GCL R2: 0.9882657838606778
- Raw oracle calls: 1188000

### weak-support-5d-911

- Pass: true
- Dimension: 5
- Profile: weak-support
- Noisy oracle: true
- Region count: 64
- Deep-only region estimate: 59
- Retained sample ratio: 0.9840909090909091
- Geometry pass: true
- Candidate pairs: 796
- Passing edges: 502
- Largest-component ratio: 1
- Max normalized projection RMSE: 0.8114987421422325
- GCL MAE: 0.06637261591590274
- GCL R2: 0.985430220526229
- Raw oracle calls: 1089000
