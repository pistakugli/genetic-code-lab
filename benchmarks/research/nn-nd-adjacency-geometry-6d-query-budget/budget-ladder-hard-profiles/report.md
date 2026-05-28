# 6D ND Recovered-Adjacency Geometry Query-Budget Ladder

Generated: 2026-05-28T11:28:26.137Z

## Claim Scope

- Proves if pass: the 6D recovered-adjacency lane keeps the hard-profile accuracy/geometry gates while reducing selected raw oracle calls versus the prior 6D sweep
- Does not claim: SOTA query efficiency, topology-perfect full ND cell-complex recovery, unique hidden-layer identity, exact layer attribution, GELU/SwiGLU recovery, or arbitrary production API extraction

## Summary

- Pass: true
- Teachers passed: 6/6
- Selected tiers: low, low, mid, high, high, low
- Low/mid/high selections: 3/1/2
- Rejected failed candidates: 5
- Min region count: 56
- Min adjacency edges: 441
- Max MAE: 0.09301806108793807
- Min R2: 0.961794949120783
- Max selected raw oracle calls per teacher: 1547000
- Previous selected raw oracle calls: 8105500
- New selected raw oracle calls: 5460000
- Selected-call reduction: 0.326383319967923
- Previous search raw oracle calls: 19220500
- New search raw oracle calls: 11102000
- Search-call reduction: 0.42238755495434566

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
- Selected candidate: near-parallel-high
- Selected tier: high
- Candidates attempted: 3
- Rejected failures: 2
- Region count: 79
- Geometry pass: true
- Passing edges: 441
- GCL MAE: 0.014703635902778069
- GCL R2: 0.961794949120783
- Selected raw oracle calls: 1547000
- Search raw oracle calls: 4095000

### low-amplitude-6d-634

- Pass: true
- Profile: low-amplitude
- Noisy oracle: true
- Selected candidate: low-amplitude-high
- Selected tier: high
- Candidates attempted: 3
- Rejected failures: 2
- Region count: 56
- Geometry pass: true
- Passing edges: 451
- GCL MAE: 0.031144792182850257
- GCL R2: 0.9921539052880295
- Selected raw oracle calls: 1274000
- Search raw oracle calls: 3276000

### weak-support-6d-911

- Pass: true
- Profile: weak-support
- Noisy oracle: true
- Selected candidate: weak-support-low
- Selected tier: low
- Candidates attempted: 1
- Rejected failures: 0
- Region count: 58
- Geometry pass: true
- Passing edges: 520
- GCL MAE: 0.04180015414938247
- GCL R2: 0.9871745801121442
- Selected raw oracle calls: 1001000
- Search raw oracle calls: 1001000

## Next Step

6D query-budget ladder passed. Next valid step is promotion-gate integration or a second efficiency pass that targets the near-parallel/cancellation high-cost profiles.
