# 6D ND Recovered-Adjacency Geometry Baseline Comparison

Generated: 2026-05-28T11:18:10.927Z

## Claim Scope

- Proves if pass: the 6D recovered-adjacency lane has an executable comparison against several non-GCL baseline families with query budgets, winners, and failure-to-be-SOTA evidence recorded
- Does not claim: SOTA if any measured black-box baseline beats GCL on MAE; topology-perfect ND cell-complex recovery; exact layer attribution; GELU/SwiGLU recovery; arbitrary production API extraction

## Summary

- Pass: true
- Teachers: 6
- Profiles: clean, cancellation, near-parallel, low-amplitude, weak-support
- Baseline families: global-linear, global-polynomial, random-feature-relu, local-affine, tree-ensemble
- GCL MAE wins: 6/6
- Best-baseline MAE wins: 0/6
- GCL beats linear ridge: 6/6
- GCL beats quadratic ridge: 6/6
- GCL beats random ReLU ridge: 6/6
- Max GCL MAE: 0.09099788983192539
- Min GCL R2: 0.9674291733376437
- Max GCL MAE ratio vs best baseline: 0.8155502106737931
- Max GCL raw oracle calls: 1989000
- Max baseline direct raw oracle calls: 18000

## Results

### clean-6d-29

- Profile: clean
- Noisy oracle: false
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.01875518669949688
- GCL R2: 0.9977404171231974
- GCL region count: 67
- GCL adjacency edges: 691
- GCL raw oracle calls: 338000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.07661978601932518
- Best baseline R2: 0.9906587398114751
- GCL MAE ratio vs best baseline: 0.24478255126902096

Ranking:
- gcl-recovered-adjacency: MAE 0.01875518669949688, R2 0.9977404171231974, queries 338000
- local-affine-knn-96: MAE 0.07661978601932518, R2 0.9906587398114751, queries 9000
- random-relu-ridge-128: MAE 0.11666163611800644, R2 0.9793591462512042, queries 18000
- quadratic-ridge: MAE 0.16068520631971764, R2 0.9611460645457615, queries 18000
- extra-trees-24x9: MAE 0.2560559598672149, R2 0.8966023258177409, queries 12000
- linear-ridge: MAE 0.3191041893977424, R2 0.8506817610703586, queries 18000

### clean-6d-83

- Profile: clean
- Noisy oracle: false
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.030638484352114075
- GCL R2: 0.9972710063911236
- GCL region count: 63
- GCL adjacency edges: 660
- GCL raw oracle calls: 338000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.11070360278467173
- Best baseline R2: 0.9923278011476343
- GCL MAE ratio vs best baseline: 0.2767614023520863

Ranking:
- gcl-recovered-adjacency: MAE 0.030638484352114075, R2 0.9972710063911236, queries 338000
- local-affine-knn-96: MAE 0.11070360278467173, R2 0.9923278011476343, queries 9000
- random-relu-ridge-128: MAE 0.1804859069950441, R2 0.9814983639646059, queries 18000
- quadratic-ridge: MAE 0.2330563278712367, R2 0.9657221911691184, queries 18000
- extra-trees-24x9: MAE 0.3901154294050437, R2 0.9123371817992617, queries 12000
- linear-ridge: MAE 0.5241023588292878, R2 0.8347332841051146, queries 18000

### cancellation-6d-717

- Profile: cancellation
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.09099788983192539
- GCL R2: 0.9928485429486841
- GCL region count: 110
- GCL adjacency edges: 1079
- GCL raw oracle calls: 1872000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.12106105294709021
- Best baseline R2: 0.9888889295364529
- GCL MAE ratio vs best baseline: 0.7516694066067313

Ranking:
- gcl-recovered-adjacency: MAE 0.09099788983192539, R2 0.9928485429486841, queries 1872000
- local-affine-knn-96: MAE 0.12106105294709021, R2 0.9888889295364529, queries 9000
- random-relu-ridge-128: MAE 0.18707258429781876, R2 0.9778868697956032, queries 18000
- quadratic-ridge: MAE 0.2578341192588026, R2 0.958663618440119, queries 18000
- extra-trees-24x9: MAE 0.40472668821726243, R2 0.8969325686908695, queries 12000
- linear-ridge: MAE 0.4478035311852953, R2 0.8772877266064869, queries 18000

### near-parallel-6d-805

- Profile: near-parallel
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.013233009749140622
- GCL R2: 0.9674291733376437
- GCL region count: 71
- GCL adjacency edges: 374
- GCL raw oracle calls: 1989000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.05134478101068327
- Best baseline R2: 0.875982446553357
- GCL MAE ratio vs best baseline: 0.25772842903716425

Ranking:
- gcl-recovered-adjacency: MAE 0.013233009749140622, R2 0.9674291733376437, queries 1989000
- local-affine-knn-96: MAE 0.05134478101068327, R2 0.875982446553357, queries 9000
- random-relu-ridge-128: MAE 0.08857050080135559, R2 0.7283496788795084, queries 18000
- extra-trees-24x9: MAE 0.10281512389479082, R2 0.6035413507103602, queries 12000
- quadratic-ridge: MAE 0.12094947776571802, R2 0.5211900910971801, queries 18000
- linear-ridge: MAE 0.12689819731766744, R2 0.43756470317554985, queries 18000

### low-amplitude-6d-634

- Profile: low-amplitude
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.030383372680069527
- GCL R2: 0.9925207123486528
- GCL region count: 57
- GCL adjacency edges: 485
- GCL raw oracle calls: 1755000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.03725506079505188
- Best baseline R2: 0.988925545884632
- GCL MAE ratio vs best baseline: 0.8155502106737931

Ranking:
- gcl-recovered-adjacency: MAE 0.030383372680069527, R2 0.9925207123486528, queries 1755000
- local-affine-knn-96: MAE 0.03725506079505188, R2 0.988925545884632, queries 9000
- random-relu-ridge-128: MAE 0.05787243065386573, R2 0.9759991108551915, queries 18000
- quadratic-ridge: MAE 0.09517900119606985, R2 0.9340058449255751, queries 18000
- extra-trees-24x9: MAE 0.12512636275464808, R2 0.887647311182356, queries 12000
- linear-ridge: MAE 0.160543484492857, R2 0.796496943143435, queries 18000

### weak-support-6d-911

- Profile: weak-support
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.03784733737640974
- GCL R2: 0.9892346155670978
- GCL region count: 68
- GCL adjacency edges: 566
- GCL raw oracle calls: 1813500
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.04943317391278884
- Best baseline R2: 0.983486401987727
- GCL MAE ratio vs best baseline: 0.7656262865738885

Ranking:
- gcl-recovered-adjacency: MAE 0.03784733737640974, R2 0.9892346155670978, queries 1813500
- local-affine-knn-96: MAE 0.04943317391278884, R2 0.983486401987727, queries 9000
- random-relu-ridge-128: MAE 0.08931921265016825, R2 0.9521216374045528, queries 18000
- quadratic-ridge: MAE 0.10213445402251958, R2 0.9341768148722598, queries 18000
- extra-trees-24x9: MAE 0.15049571823448007, R2 0.8528311575777638, queries 12000
- linear-ridge: MAE 0.29120095690059394, R2 0.4699161952265002, queries 18000

## Conclusion

GCL won MAE against every measured baseline on this 6D suite; the next step is promotion-gate integration and stronger external packages.
