# Adaptive-Adversarial Noisy Deep Black-Box ReLU 2D Region-Graph Sweep

Generated: 2026-06-03T08:58:20.740Z

Pass: yes

## Scope

- Teachers: adaptive generated 2-input / 4-hidden / 4-hidden / 1-output ReLU MLPs
- GCL access: repeated noisy/quantized oracle calls, `f(x,y)` only
- Probing: screening pass for every candidate; refined pass for screening failures and selected hard cases
- Selection: generate adversarial pool, score difficulty, select hardest cases after final probing
- Hard rule: any final candidate failure fails the benchmark; failures are not filtered out
- Not claimed: original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, or production API extraction beyond this noise model

## Summary

- Pool candidates passed: 12/12
- Selected hard cases passed: 4/4
- Selected profiles: boundary-grazing-fold, near-parallel-fold, offset-collision-fold
- Selected teacher ids: adaptive-deep-2d-offset-collision-fold-260698, adaptive-deep-2d-boundary-grazing-fold-260989, adaptive-deep-2d-near-parallel-fold-260601, adaptive-deep-2d-offset-collision-fold-261280
- Min selected regions: 75
- Min selected deep-only region estimate: 70
- Min selected adjacency edges: 246
- Min selected kink-cell boundaries: 1984
- Max selected MAE: 0.006061807867996213
- Min selected R2: 0.9989576605269184
- Max selected first-layer MAE ratio: 0.12621348237546748
- Refined candidates: 8/12
- Total raw oracle calls: 9313740
- Max raw oracle calls per candidate: 1005780

## Results

| Teacher | Profile | Stage | Selected | Pass | Score | Regions | Deep-only | Edges | Kink boundaries | MAE | R2 | First-layer ratio | First-layer MAE | Plane MAE | Final calls | Total calls |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive-deep-2d-near-parallel-fold-260601 | near-parallel-fold | refined | yes | yes | 17.828547 | 99 | 94 | 439 | 1984 | 4.786797e-3 | 0.998957660527 | 0.126213 | 3.792619e-2 | 3.524131e-1 | 688905 | 1005780 |
| adaptive-deep-2d-offset-collision-fold-260698 | offset-collision-fold | refined | yes | yes | 20.330887 | 85 | 80 | 287 | 2343 | 1.316134e-3 | 0.999956923390 | 0.019320 | 6.812229e-2 | 3.922917e-1 | 688905 | 1005780 |
| adaptive-deep-2d-cancellation-fold-260795 | cancellation-fold | refined | no | yes | 14.837126 | 67 | 62 | 242 | 1640 | 4.683064e-3 | 0.999636547930 | 0.253924 | 1.844281e-2 | 1.642558e-1 | 688905 | 1005780 |
| adaptive-deep-2d-low-amplitude-fold-260892 | low-amplitude-fold | refined | no | yes | 14.615646 | 74 | 69 | 242 | 1850 | 8.405311e-3 | 0.998896334786 | 0.255827 | 3.285551e-2 | 1.431098e-1 | 688905 | 1005780 |
| adaptive-deep-2d-boundary-grazing-fold-260989 | boundary-grazing-fold | refined | yes | yes | 18.156259 | 114 | 109 | 383 | 2383 | 4.933814e-3 | 0.999637083014 | 0.078346 | 6.297431e-2 | 2.302970e-1 | 688905 | 1005780 |
| adaptive-deep-2d-mixed-oblique-fold-261086 | mixed-oblique-fold | screening | no | yes | 14.144449 | 57 | 52 | 196 | 1426 | 1.452924e-2 | 0.999206486782 | 0.375087 | 3.873561e-2 | 2.347062e-1 | 316875 | 316875 |
| adaptive-deep-2d-near-parallel-fold-261183 | near-parallel-fold | refined | no | yes | 16.144168 | 78 | 73 | 345 | 1705 | 2.743264e-3 | 0.999870713741 | 0.392402 | 6.990957e-3 | 3.901910e-1 | 688905 | 1005780 |
| adaptive-deep-2d-offset-collision-fold-261280 | offset-collision-fold | refined | yes | yes | 17.053990 | 75 | 70 | 246 | 2302 | 6.061808e-3 | 0.999772958450 | 0.106837 | 5.673866e-2 | 3.097750e-1 | 688905 | 1005780 |
| adaptive-deep-2d-cancellation-fold-261377 | cancellation-fold | screening | no | yes | 12.731810 | 39 | 34 | 124 | 1233 | 1.569513e-2 | 0.997492526749 | 0.757789 | 2.071174e-2 | 1.897681e-1 | 316875 | 316875 |
| adaptive-deep-2d-low-amplitude-fold-261474 | low-amplitude-fold | screening | no | yes | 13.359246 | 42 | 37 | 126 | 1275 | 9.432833e-3 | 0.999601607340 | 0.234769 | 4.017925e-2 | 1.901652e-1 | 316875 | 316875 |
| adaptive-deep-2d-boundary-grazing-fold-261571 | boundary-grazing-fold | refined | no | yes | 15.304348 | 78 | 73 | 261 | 1982 | 6.021959e-3 | 0.999193306585 | 0.138807 | 4.338358e-2 | 1.383623e-1 | 688905 | 1005780 |
| adaptive-deep-2d-mixed-oblique-fold-261668 | mixed-oblique-fold | screening | no | yes | 14.129298 | 54 | 49 | 174 | 1434 | 1.040658e-2 | 0.999392600317 | 0.244569 | 4.255062e-2 | 2.061818e-1 | 316875 | 316875 |
