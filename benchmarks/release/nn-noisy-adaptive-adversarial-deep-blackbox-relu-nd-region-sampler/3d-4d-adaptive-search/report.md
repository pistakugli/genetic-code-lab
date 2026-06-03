# Noisy Adaptive-Adversarial Deep Black-Box ReLU ND Region-Sampler Benchmark

Generated: 2026-06-03T11:00:02.794Z

## Claim Scope

- Proves: benchmark-scoped noisy output-equivalent deep-ND recovery across generated near-parallel, cancellation, low-amplitude, and mixed-oblique folded teachers with adaptive region-vote/local-affine attempt selection
- Does not claim: unique hidden-layer recovery, exact layer attribution, topology-perfect ND region adjacency, arbitrary production API extraction, or superiority over all model-extraction algorithms

## Summary

- Pass: true
- Teachers passed: 7/7
- Dimensions: 3, 4
- Profiles: near-parallel, cancellation, low-amplitude, mixed-oblique
- Selected local-affine attempts: 2
- Min recovered regions: 6
- Median deep-only region estimate: 25
- Max MAE: 0.026450213096249696
- Min R2: 0.9830270507435355
- Min linear MAE improvement ratio: 7.392219720751436
- Total selected raw oracle calls: 4274160
- Total adaptive-search raw oracle calls: 14132880

## Results

### near-parallel-3d-600

- Pass: true
- Dimension: 3
- Profile: near-parallel
- Selected attempt: region-vote-screen
- Selected prediction mode: region-vote
- Recovered regions: 30
- Deep-only region estimate: 25
- Retained sample ratio: 0.9759523809523809
- GCL MAE: 0.01084360634770528
- GCL R2: 0.9830270507435355
- Linear MAE: 0.15257986758548048
- Linear improvement ratio: 14.070952291418193
- Selected raw oracle calls: 385080

### cancellation-3d-617

- Pass: true
- Dimension: 3
- Profile: cancellation
- Selected attempt: region-vote-screen
- Selected prediction mode: region-vote
- Recovered regions: 43
- Deep-only region estimate: 38
- Retained sample ratio: 0.9047619047619048
- GCL MAE: 0.009741056831680082
- GCL R2: 0.9996903905411649
- Linear MAE: 0.7843126362376979
- Linear improvement ratio: 80.51617496850433
- Selected raw oracle calls: 385080

### low-amplitude-3d-634

- Pass: true
- Dimension: 3
- Profile: low-amplitude
- Selected attempt: local-affine-refined
- Selected prediction mode: local-affine-knn
- Recovered regions: 19
- Deep-only region estimate: 14
- Retained sample ratio: 0.9932142857142857
- GCL MAE: 0.01184634389739626
- GCL R2: 0.9985354232081954
- Linear MAE: 0.21163811428982576
- Linear improvement ratio: 17.865268484763664
- Selected raw oracle calls: 590880

### near-parallel-4d-700

- Pass: true
- Dimension: 4
- Profile: near-parallel
- Selected attempt: region-vote-screen
- Selected prediction mode: region-vote
- Recovered regions: 6
- Deep-only region estimate: 1
- Retained sample ratio: 0.9979032258064516
- GCL MAE: 0.007067514420940592
- GCL R2: 0.9979918568178046
- Linear MAE: 0.17131006581384875
- Linear improvement ratio: 24.23908259829906
- Selected raw oracle calls: 728280

### cancellation-4d-717

- Pass: true
- Dimension: 4
- Profile: cancellation
- Selected attempt: region-vote-screen
- Selected prediction mode: region-vote
- Recovered regions: 56
- Deep-only region estimate: 51
- Retained sample ratio: 0.9619354838709677
- GCL MAE: 0.026450213096249696
- GCL R2: 0.9990384851356013
- Linear MAE: 0.8251851255028345
- Linear improvement ratio: 31.19767400361078
- Selected raw oracle calls: 728280

### low-amplitude-4d-734

- Pass: true
- Dimension: 4
- Profile: low-amplitude
- Selected attempt: local-affine-screen
- Selected prediction mode: local-affine-knn
- Recovered regions: 19
- Deep-only region estimate: 14
- Retained sample ratio: 0.9966129032258064
- GCL MAE: 0.01737370491317421
- GCL R2: 0.9913035108728268
- Linear MAE: 0.12843024408168252
- Linear improvement ratio: 7.392219720751436
- Selected raw oracle calls: 728280

### mixed-oblique-4d-751

- Pass: true
- Dimension: 4
- Profile: mixed-oblique
- Selected attempt: region-vote-screen
- Selected prediction mode: region-vote
- Recovered regions: 47
- Deep-only region estimate: 42
- Retained sample ratio: 0.9719354838709677
- GCL MAE: 0.019791173183916613
- GCL R2: 0.9986468901298954
- Linear MAE: 0.4484283811675755
- Linear improvement ratio: 22.65799894732834
- Selected raw oracle calls: 728280

