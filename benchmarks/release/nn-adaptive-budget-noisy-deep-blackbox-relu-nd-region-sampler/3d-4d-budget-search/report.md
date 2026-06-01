# Adaptive-Budget Noisy Deep Black-Box ReLU ND Region-Sampler Benchmark

Generated: 2026-06-01T16:14:28.001Z

## Claim Scope

- Proves: benchmark-scoped query-budget reduction for noisy deep-ND output-equivalent recovery using sequential noisy holdout promotion and explicit rejected-candidate accounting
- Does not claim: unique hidden-layer recovery, exact layer attribution, topology-perfect ND region adjacency, arbitrary production API extraction, or globally optimal query complexity

## Summary

- Pass: true
- Teachers passed: 7/7
- Dimensions: 3, 4
- Profiles: near-parallel, cancellation, low-amplitude, mixed-oblique
- Selected local-affine attempts: 3
- Selected low/mid/high budgets: 6/0/1
- Rejected candidates recorded: 7
- Max attempted candidates per teacher: 5
- Min recovered regions: 5
- Median deep-only region estimate: 22
- Max MAE: 0.03291800518211379
- Min R2: 0.9865538047691716
- Min linear MAE improvement ratio: 5.074298352405206
- Total selected raw oracle calls: 1862760
- Total adaptive-search raw oracle calls: 3404120
- Selected oracle reduction vs previous release: 0.5641810320624403
- Search oracle reduction vs previous adaptive search: 0.7591347269629403

## Results

### near-parallel-3d-600

- Pass: true
- Dimension: 3
- Profile: near-parallel
- Selected attempt: high-region-vote
- Selected budget: high
- Selected prediction mode: region-vote
- Attempted candidates: 5
- Rejected candidates: 4
- Recovered regions: 27
- Deep-only region estimate: 22
- GCL MAE: 0.011282288800366786
- GCL R2: 0.9865538047691716
- Selection R2: 0.9826927499278406
- Linear improvement ratio: 13.479784751876782
- Selected raw oracle calls: 383880
- Adaptive-search raw oracle calls: 1185800

### cancellation-3d-617

- Pass: true
- Dimension: 3
- Profile: cancellation
- Selected attempt: low-region-vote
- Selected budget: low
- Selected prediction mode: region-vote
- Attempted candidates: 1
- Rejected candidates: 0
- Recovered regions: 50
- Deep-only region estimate: 45
- GCL MAE: 0.0166405973899647
- GCL R2: 0.9994142796851194
- Selection R2: 0.9990916592871938
- Linear improvement ratio: 47.489506491842675
- Selected raw oracle calls: 152880
- Adaptive-search raw oracle calls: 152880

### low-amplitude-3d-634

- Pass: true
- Dimension: 3
- Profile: low-amplitude
- Selected attempt: low-local-affine
- Selected budget: low
- Selected prediction mode: local-affine-knn
- Attempted candidates: 2
- Rejected candidates: 1
- Recovered regions: 16
- Deep-only region estimate: 11
- GCL MAE: 0.014788524937942688
- GCL R2: 0.9980137957059823
- Selection R2: 0.9975720476299238
- Linear improvement ratio: 14.541499338269377
- Selected raw oracle calls: 152880
- Adaptive-search raw oracle calls: 305760

### near-parallel-4d-700

- Pass: true
- Dimension: 4
- Profile: near-parallel
- Selected attempt: low-local-affine
- Selected budget: low
- Selected prediction mode: local-affine-knn
- Attempted candidates: 2
- Rejected candidates: 1
- Recovered regions: 5
- Deep-only region estimate: 0
- GCL MAE: 0.03291800518211379
- GCL R2: 0.989487792145741
- Selection R2: 0.9907219998217379
- Linear improvement ratio: 5.074298352405206
- Selected raw oracle calls: 293280
- Adaptive-search raw oracle calls: 586560

### cancellation-4d-717

- Pass: true
- Dimension: 4
- Profile: cancellation
- Selected attempt: low-region-vote
- Selected budget: low
- Selected prediction mode: region-vote
- Attempted candidates: 1
- Rejected candidates: 0
- Recovered regions: 39
- Deep-only region estimate: 34
- GCL MAE: 0.030633755616588567
- GCL R2: 0.9991865130369987
- Selection R2: 0.9991540328154574
- Linear improvement ratio: 26.96155270996184
- Selected raw oracle calls: 293280
- Adaptive-search raw oracle calls: 293280

### low-amplitude-4d-734

- Pass: true
- Dimension: 4
- Profile: low-amplitude
- Selected attempt: low-local-affine
- Selected budget: low
- Selected prediction mode: local-affine-knn
- Attempted candidates: 2
- Rejected candidates: 1
- Recovered regions: 11
- Deep-only region estimate: 6
- GCL MAE: 0.02036945248336141
- GCL R2: 0.9876511384225717
- Selection R2: 0.9845573234252493
- Linear improvement ratio: 6.191373633859825
- Selected raw oracle calls: 293280
- Adaptive-search raw oracle calls: 586560

### mixed-oblique-4d-751

- Pass: true
- Dimension: 4
- Profile: mixed-oblique
- Selected attempt: low-region-vote
- Selected budget: low
- Selected prediction mode: region-vote
- Attempted candidates: 1
- Rejected candidates: 0
- Recovered regions: 38
- Deep-only region estimate: 33
- GCL MAE: 0.02436779540702078
- GCL R2: 0.998048097786422
- Selection R2: 0.9965830445289798
- Linear improvement ratio: 18.372949913629796
- Selected raw oracle calls: 293280
- Adaptive-search raw oracle calls: 293280

