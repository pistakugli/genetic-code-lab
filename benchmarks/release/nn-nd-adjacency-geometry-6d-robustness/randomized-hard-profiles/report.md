# 6D ND Recovered-Adjacency Geometry Robustness Gate

Generated: 2026-06-02T07:15:13.981Z

## Claim Scope

- Proves if pass: the 6D recovered-adjacency lane keeps the fixed hard-profile geometry/accuracy gates on a wider deterministic randomized/adversarial seed set while still avoiding high-tier selections
- Does not claim: broad SOTA status, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, or arbitrary production API extraction

## Summary

- Pass: true
- Teachers passed: 12/12
- Selected tiers: low, low, mid, mid, mid, low, mid, low, low, low, mid, low
- Low/mid/high selections: 7/5/0
- Rejected failed candidates: 8
- Min region count: 56
- Min adjacency edges: 468
- Max MAE: 0.09301806108793807
- Min R2: 0.9601449089658317
- Max selected raw oracle calls per teacher: 1501500
- Average selected raw oracle calls per teacher: 804375
- Fixed query-budget average selected raw oracle calls: 841750
- Average selected-call reduction vs fixed query-budget proof: 0.044401544401544424
- Average search raw oracle calls per teacher: 1456541.6666666667
- Fixed query-budget average search raw oracle calls: 1782083.3333333333
- Average search-call reduction vs fixed query-budget proof: 0.1826747720364741

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

### clean-6d-151

- Pass: true
- Profile: clean
- Noisy oracle: false
- Selected candidate: clean-mid
- Selected tier: mid
- Candidates attempted: 2
- Rejected failures: 1
- Region count: 56
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 469
- GCL MAE: 0.02706279487397642
- GCL R2: 0.9970428214183646
- Selected raw oracle calls: 234000
- Search raw oracle calls: 416000

### clean-6d-211

- Pass: true
- Profile: clean
- Noisy oracle: false
- Selected candidate: clean-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 57
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 619
- GCL MAE: 0.037607319823290956
- GCL R2: 0.9971755828213908
- Selected raw oracle calls: 182000
- Search raw oracle calls: 182000

### cancellation-6d-742

- Pass: true
- Profile: cancellation
- Noisy oracle: true
- Selected candidate: cancellation-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 91
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 960
- GCL MAE: 0.05805537550049842
- GCL R2: 0.9926343502227222
- Selected raw oracle calls: 1092000
- Search raw oracle calls: 1092000

### near-parallel-6d-829

- Pass: true
- Profile: near-parallel
- Noisy oracle: true
- Selected candidate: near-parallel-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 95
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 729
- GCL MAE: 0.018791016075038764
- GCL R2: 0.9785253611754295
- Selected raw oracle calls: 1183000
- Search raw oracle calls: 1183000

### low-amplitude-6d-658

- Pass: true
- Profile: low-amplitude
- Noisy oracle: true
- Selected candidate: low-amplitude-coverage
- Selected tier: mid
- Candidates attempted: 3
- Rejected failures: 2
- Region count: 65
- Effective min region support: 4
- Geometry pass: true
- Passing edges: 538
- GCL MAE: 0.0342815325985024
- GCL R2: 0.9941575530659655
- Selected raw oracle calls: 910000
- Search raw oracle calls: 2912000

### weak-support-6d-937

- Pass: true
- Profile: weak-support
- Noisy oracle: true
- Selected candidate: weak-support-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 112
- Effective min region support: 8
- Geometry pass: true
- Passing edges: 1069
- GCL MAE: 0.08742366261199823
- GCL R2: 0.9943721976433024
- Selected raw oracle calls: 1001000
- Search raw oracle calls: 1001000

## Next Step

6D randomized/adversarial robustness passed with zero high-tier selections. Next target is external extraction baselines on the same teacher manifest.
