# 6D ND Recovered-Adjacency Geometry Query-Budget Ladder

Generated: 2026-05-31T05:55:11.605Z

## Claim Scope

- Proves if pass: the 6D recovered-adjacency lane keeps the hard-profile accuracy/geometry gates, eliminates high-tier selections, and reduces selected raw oracle calls versus the prior 6D sweep
- Does not claim: SOTA query efficiency, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, or arbitrary production API extraction

## Summary

- Pass: true
- Teachers passed: 6/6
- Selected tiers: low, low, mid, mid, mid, low
- Low/mid/high selections: 3/3/0
- Rejected failed candidates: 5
- Min region count: 58
- Min adjacency edges: 468
- Max MAE: 0.09301806108793807
- Min R2: 0.9601449089658317
- Max selected raw oracle calls per teacher: 1501500
- Previous selected raw oracle calls: 8105500
- New selected raw oracle calls: 5050500
- Selected-call reduction: 0.3769045709703288
- Previous search raw oracle calls: 19220500
- New search raw oracle calls: 10692500
- Search-call reduction: 0.4436929320257017

## Results

### clean-6d-29

- Pass: true
- Profile: clean
- Noisy oracle: false
- Selected candidate: clean-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 65
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 667
- GCL MAE: 0.02514738606591309
- GCL R2: 0.996313022034061
- Selected raw oracle calls: 182000
- Search raw oracle calls: 182000

### clean-6d-83

- Pass: true
- Profile: clean
- Noisy oracle: false
- Selected candidate: clean-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 59
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 596
- GCL MAE: 0.03946414407557655
- GCL R2: 0.9956624586754947
- Selected raw oracle calls: 182000
- Search raw oracle calls: 182000

### cancellation-6d-717

- Pass: true
- Profile: cancellation
- Noisy oracle: true
- Selected candidate: cancellation-mid
- Selected tier: mid
- Candidates attempted: 2
- Rejected failures: 1
- Region count: 102
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 1048
- GCL MAE: 0.09301806108793807
- GCL R2: 0.9925411574680458
- Selected raw oracle calls: 1274000
- Search raw oracle calls: 2366000

### near-parallel-6d-805

- Pass: true
- Profile: near-parallel
- Noisy oracle: true
- Selected candidate: near-parallel-lhs
- Selected tier: mid
- Candidates attempted: 3
- Rejected failures: 2
- Region count: 80
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 468
- GCL MAE: 0.014981339622159851
- GCL R2: 0.9601449089658317
- Selected raw oracle calls: 1501500
- Search raw oracle calls: 4049500

### low-amplitude-6d-634

- Pass: true
- Profile: low-amplitude
- Noisy oracle: true
- Selected candidate: low-amplitude-coverage
- Selected tier: mid
- Candidates attempted: 3
- Rejected failures: 2
- Region count: 77
- Effective min region support: 4
- Geometry pass: true
- Passing edges: 527
- GCL MAE: 0.031089536651472263
- GCL R2: 0.992196673628252
- Selected raw oracle calls: 910000
- Search raw oracle calls: 2912000

### weak-support-6d-911

- Pass: true
- Profile: weak-support
- Noisy oracle: true
- Selected candidate: weak-support-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 58
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 520
- GCL MAE: 0.04180015414938247
- GCL R2: 0.9871745801121442
- Selected raw oracle calls: 1001000
- Search raw oracle calls: 1001000

## Next Step

6D query-budget ladder passed with zero high-tier selections. Next efficiency target is stronger external extraction baselines and wider 6D randomized coverage.
