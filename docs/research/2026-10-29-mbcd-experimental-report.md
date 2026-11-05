# Multi-Hypothesis Boundary Continuation and Disambiguation (MBCD): Focused Experiment

**Run:** 2026-10-29T19-23-00-783Z  
**Verdict:** not-supported  
**Dimensions:** 16, 32  
**Scenarios:** near-parallel-low-rank, multi-crossing-low-rank, small-amplitude-low-rank, contaminated-local-probes, orthogonal-full-rank  
**Methods/ablations:** asad, ransac, star-probe, adaptive, bcad, mbcd-full, mbcd-single-rank, mbcd-no-contradiction, mbcd-no-independent-validation  
**Seeds per cell:** 100  
**Raw observations:** [mbcd-2026-10-29T19-23-00-783Z-raw.jsonl](../../scratch/results/mbcd/mbcd-2026-10-29T19-23-00-783Z-raw.jsonl)

## Research question and implementation

MBCD keeps multiple oracle-derived active-rank and boundary hypotheses rather than promoting the current best candidate into a settled explanation. A candidate boundary tracks detection, normal identification, offset identification, uniqueness, and later validation separately. The contradiction-seeking policy queries where candidate planes disagree about which side contains a point. The query is one full finite-difference gradient stencil, costing exactly $2d+1$ oracle calls; no partial stencil starts when the remaining budget is smaller. Model fitting and scoring make no additional oracle calls.

The solver receives only the oracle, dimension, ranges, requested term count, seed, and budget. Teacher metadata and the shared evaluation set are used only by the benchmark after reconstruction. Existing ASAD, BCAD, RANSAC, Star-Probe, and Adaptive implementations are called without modification.

## Paired design and predeclared criteria

Each seed/dimension/scenario uses the same deterministic teacher, solver seed, solverBudget result, and 192 independently generated evaluation points for every method. The five focus scenarios are near-parallel facets, multiple crossings, small-amplitude jumps, contaminated local probes, and orthogonal full-rank controls at 16D and 32D. MBCD ablations remove rank alternatives, contradiction-seeking acquisition, or independent boundary validation one component at a time.

A primary cell qualifies if full MBCD achieves at least 25% lower MAE than the best scored existing method, or uses at least 30% fewer calls than an existing method within 5% MAE of the best baseline while staying within 5% of that best MAE. Its false-accepted-run rate must be no more than one percentage point above the lowest such comparable baseline rate. At least two of the three primary scenario classes must qualify at both dimensions. There are 100 seeds per cell and the actual oracle-call total must never exceed the common budget. The verdict is not derived from test pass status.

## Primary comparison

| D | Scenario | Best existing method | MBCD MAE | Baseline MAE | Call reduction | MAE reduction | MBCD complete rate | False-accepted increase reference | Qualifies | Reason |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 16 | near-parallel-low-rank | star-probe | 0.089202 | 1.4793e-12 | 73.75% | n/a | 0.00% | 80.00% | no | neither_predeclared_performance_threshold_met |
| 16 | multi-crossing-low-rank | star-probe | 0.12509 | 1.3565e-12 | 74.40% | n/a | 0.00% | 77.00% | no | neither_predeclared_performance_threshold_met |
| 16 | small-amplitude-low-rank | ransac | 0.0013607 | 4.2079e-12 | 68.90% | n/a | 0.00% | 15.00% | no | neither_predeclared_performance_threshold_met |
| 32 | near-parallel-low-rank | star-probe | 0.10398 | 1.0141e-12 | 70.39% | n/a | 0.00% | 74.00% | no | neither_predeclared_performance_threshold_met |
| 32 | multi-crossing-low-rank | star-probe | 0.14438 | 1.768e-12 | 71.11% | n/a | 0.00% | 81.00% | no | neither_predeclared_performance_threshold_met |
| 32 | small-amplitude-low-rank | asad | 0.0010523 | 3.1039e-10 | -56.86% | n/a | 0.00% | 11.00% | no | neither_predeclared_performance_threshold_met |

**Primary scenario classes qualifying at both dimensions:** 0/3 (none).  
**Actual oracle-budget overruns:** 0.  
**Run statuses:** 1961 complete, 5505 incomplete, 1534 threw.

## Method metrics (all dimensions and focus scenarios)

| D | Scenario | Method | Scored/runs | MAE | Oracle calls | Complete rate | Normal error (deg) | Offset error | False-accepted run rate | Mean unresolved hypotheses | Rank misestimate rate | Runtime (ms) | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | near-parallel-low-rank | asad | 100/100 | 0.51848 | 3124.1 | 2.00% | 69.805 | 0.026736 | 4.00% | n/a | 100.00% | 58.704 | 0 |
| 16 | near-parallel-low-rank | ransac | 0/100 | n/a | 8448 | 0.00% | n/a | n/a | n/a | n/a | n/a | 5.2584 | 100 |
| 16 | near-parallel-low-rank | star-probe | 100/100 | 1.4793e-12 | 10059 | 100.00% | 1.3868e-7 | 4.7231e-13 | 0.00% | n/a | n/a | 6.7875 | 0 |
| 16 | near-parallel-low-rank | adaptive | 100/100 | 1.4793e-12 | 10717 | 100.00% | 1.3868e-7 | 4.7231e-13 | 0.00% | n/a | n/a | 6.9406 | 0 |
| 16 | near-parallel-low-rank | bcad | 100/100 | 0.026189 | 2418.6 | 53.00% | 21.604 | 0.00049794 | 0.00% | n/a | n/a | 38.234 | 0 |
| 16 | near-parallel-low-rank | mbcd-full | 100/100 | 0.089202 | 2640 | 0.00% | 50.773 | 0.04253 | 80.00% | 1.91 | 100.00% | 26.831 | 0 |
| 16 | near-parallel-low-rank | mbcd-single-rank | 100/100 | 0.089202 | 2640 | 0.00% | 50.773 | 0.04253 | 80.00% | 1.91 | 100.00% | 26.64 | 0 |
| 16 | near-parallel-low-rank | mbcd-no-contradiction | 100/100 | 0.072715 | 2640 | 0.00% | 54.597 | 0.03814 | 88.00% | 1.54 | 100.00% | 32.643 | 0 |
| 16 | near-parallel-low-rank | mbcd-no-independent-validation | 100/100 | 0.088852 | 2640 | 0.00% | 50.337 | 0.040347 | 80.00% | 2 | 100.00% | 26.802 | 0 |
| 16 | multi-crossing-low-rank | asad | 100/100 | 1.0724 | 3168 | 0.00% | 88.5 | 6.2786e-13 | 1.00% | n/a | 0.00% | 62.371 | 0 |
| 16 | multi-crossing-low-rank | ransac | 0/100 | n/a | 8448 | 0.00% | n/a | n/a | n/a | n/a | n/a | 7.1614 | 100 |
| 16 | multi-crossing-low-rank | star-probe | 98/100 | 1.3565e-12 | 10312 | 98.00% | 1.1892e-7 | 8.9298e-13 | 0.00% | n/a | n/a | 8.4915 | 2 |
| 16 | multi-crossing-low-rank | adaptive | 98/100 | 0.0058741 | 11293 | 97.00% | 1.1892e-7 | 8.9298e-13 | 0.00% | n/a | n/a | 9.1671 | 2 |
| 16 | multi-crossing-low-rank | bcad | 100/100 | 0.27624 | 3128.1 | 2.00% | 81.356 | 0.00042098 | 18.00% | n/a | n/a | 65.695 | 0 |
| 16 | multi-crossing-low-rank | mbcd-full | 100/100 | 0.12509 | 2640 | 0.00% | 30.937 | 0.00035281 | 77.00% | 11.26 | 1.00% | 42.078 | 0 |
| 16 | multi-crossing-low-rank | mbcd-single-rank | 100/100 | 0.12509 | 2640 | 0.00% | 30.937 | 0.00035281 | 77.00% | 11.26 | 1.00% | 42.381 | 0 |
| 16 | multi-crossing-low-rank | mbcd-no-contradiction | 100/100 | 0.18712 | 2640 | 0.00% | 28.243 | 0.00031886 | 76.00% | 7.83 | 0.00% | 60.453 | 0 |
| 16 | multi-crossing-low-rank | mbcd-no-independent-validation | 100/100 | 0.13338 | 2640 | 0.00% | 29.722 | 0.00012882 | 78.00% | 10.77 | 14.00% | 42.529 | 0 |
| 16 | small-amplitude-low-rank | asad | 100/100 | 2.9193e-7 | 891 | 100.00% | 0.0017296 | 0.0000021332 | 0.00% | n/a | 0.00% | 7.8629 | 0 |
| 16 | small-amplitude-low-rank | ransac | 3/100 | 4.2079e-12 | 8488.8 | 3.00% | 0 | 2.0548e-11 | 0.00% | n/a | n/a | 3.7185 | 97 |
| 16 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 9189.8 | 0.00% | n/a | n/a | n/a | n/a | n/a | 3.7615 | 100 |
| 16 | small-amplitude-low-rank | adaptive | 3/100 | 4.2079e-12 | 9230.6 | 3.00% | 0 | 2.0548e-11 | 0.00% | n/a | n/a | 3.8625 | 97 |
| 16 | small-amplitude-low-rank | bcad | 100/100 | 0.00030953 | 1153 | 93.00% | 6.3 | 3.7621e-12 | 0.00% | n/a | n/a | 13.675 | 0 |
| 16 | small-amplitude-low-rank | mbcd-full | 100/100 | 0.0013607 | 2640 | 0.00% | 15.945 | 0.025811 | 15.00% | 0.8 | 0.00% | 23.659 | 0 |
| 16 | small-amplitude-low-rank | mbcd-single-rank | 100/100 | 0.0013607 | 2640 | 0.00% | 15.945 | 0.025811 | 15.00% | 0.8 | 0.00% | 23.437 | 0 |
| 16 | small-amplitude-low-rank | mbcd-no-contradiction | 100/100 | 0.0012455 | 2640 | 0.00% | 15.913 | 0.02549 | 15.00% | 0.75 | 0.00% | 25.275 | 0 |
| 16 | small-amplitude-low-rank | mbcd-no-independent-validation | 100/100 | 0.0013812 | 2640 | 0.00% | 14.388 | 0.025923 | 13.00% | 0.78 | 0.00% | 23.642 | 0 |
| 16 | contaminated-local-probes | asad | 100/100 | 0.000048227 | 891 | 100.00% | 0.0027674 | 0.0000033952 | 0.00% | n/a | 100.00% | 8.0495 | 0 |
| 16 | contaminated-local-probes | ransac | 4/100 | 7.3879e-11 | 8502.4 | 4.00% | 0 | 3.5482e-11 | 0.00% | n/a | n/a | 3.9862 | 96 |
| 16 | contaminated-local-probes | star-probe | 100/100 | 1.8908e-11 | 9874 | 100.00% | 1.3639e-7 | 3.0509e-13 | 0.00% | n/a | n/a | 5.5744 | 0 |
| 16 | contaminated-local-probes | adaptive | 100/100 | 1.8908e-11 | 10203 | 100.00% | 1.3639e-7 | 3.0509e-13 | 0.00% | n/a | n/a | 5.7244 | 0 |
| 16 | contaminated-local-probes | bcad | 100/100 | 0.0091804 | 1161.6 | 93.00% | 5.4102 | 0.0000092762 | 0.00% | n/a | n/a | 15.545 | 0 |
| 16 | contaminated-local-probes | mbcd-full | 100/100 | 0.045408 | 2640 | 0.00% | 23.962 | 0.029599 | 24.00% | 1.19 | 100.00% | 24.342 | 0 |
| 16 | contaminated-local-probes | mbcd-single-rank | 100/100 | 0.045408 | 2640 | 0.00% | 23.962 | 0.029599 | 24.00% | 1.19 | 100.00% | 24.248 | 0 |
| 16 | contaminated-local-probes | mbcd-no-contradiction | 100/100 | 0.042341 | 2640 | 0.00% | 26.352 | 0.027469 | 27.00% | 1.15 | 100.00% | 27.553 | 0 |
| 16 | contaminated-local-probes | mbcd-no-independent-validation | 100/100 | 0.047179 | 2640 | 0.00% | 24.208 | 0.029465 | 24.00% | 1.14 | 100.00% | 24.243 | 0 |
| 16 | orthogonal-full-rank | asad | 100/100 | 0.90297 | 3168 | 0.00% | 90 | n/a | 0.00% | n/a | 22.00% | 74.184 | 0 |
| 16 | orthogonal-full-rank | ransac | 0/100 | n/a | 8448 | 0.00% | n/a | n/a | n/a | n/a | n/a | 30.328 | 100 |
| 16 | orthogonal-full-rank | star-probe | 100/100 | 1.4945 | 12848 | 0.00% | 3.2063 | 3.8921e-12 | 0.00% | n/a | n/a | 31.178 | 0 |
| 16 | orthogonal-full-rank | adaptive | 100/100 | 1.4945 | 12848 | 0.00% | 35.494 | 3.8572e-12 | 99.00% | n/a | n/a | 34.274 | 0 |
| 16 | orthogonal-full-rank | bcad | 100/100 | 0.67812 | 3168 | 0.00% | 90 | n/a | 15.00% | n/a | n/a | 122.68 | 0 |
| 16 | orthogonal-full-rank | mbcd-full | 100/100 | 0.76347 | 2640 | 0.00% | 89.944 | 1.8838e-13 | 100.00% | 20.56 | 100.00% | 60.057 | 0 |
| 16 | orthogonal-full-rank | mbcd-single-rank | 100/100 | 0.76347 | 2640 | 0.00% | 89.944 | 1.8838e-13 | 100.00% | 20.56 | 100.00% | 59.723 | 0 |
| 16 | orthogonal-full-rank | mbcd-no-contradiction | 100/100 | 0.81805 | 2640 | 0.00% | 89.944 | 1.8838e-13 | 100.00% | 17.55 | 100.00% | 84.288 | 0 |
| 16 | orthogonal-full-rank | mbcd-no-independent-validation | 100/100 | 0.76237 | 2640 | 0.00% | 89.944 | 1.8838e-13 | 100.00% | 20.59 | 100.00% | 61.791 | 0 |
| 32 | near-parallel-low-rank | asad | 100/100 | 0.55076 | 6108.7 | 5.00% | 67.128 | 0.033237 | 4.00% | n/a | 100.00% | 108.39 | 0 |
| 32 | near-parallel-low-rank | ransac | 0/100 | n/a | 14592 | 0.00% | n/a | n/a | n/a | n/a | n/a | 14.208 | 100 |
| 32 | near-parallel-low-rank | star-probe | 85/100 | 1.0141e-12 | 17561 | 85.00% | 2.4091e-7 | 7.8995e-13 | 0.00% | n/a | n/a | 22.303 | 15 |
| 32 | near-parallel-low-rank | adaptive | 85/100 | 1.0141e-12 | 18747 | 85.00% | 2.4091e-7 | 7.8995e-13 | 0.00% | n/a | n/a | 23.63 | 15 |
| 32 | near-parallel-low-rank | bcad | 100/100 | 0.030583 | 4835.4 | 60.00% | 19.8 | 5.2135e-13 | 0.00% | n/a | n/a | 70.349 | 0 |
| 32 | near-parallel-low-rank | mbcd-full | 100/100 | 0.10398 | 5200 | 0.00% | 51.73 | 0.031818 | 74.00% | 1.57 | 100.00% | 48.251 | 0 |
| 32 | near-parallel-low-rank | mbcd-single-rank | 100/100 | 0.10398 | 5200 | 0.00% | 51.73 | 0.031818 | 74.00% | 1.57 | 100.00% | 48.345 | 0 |
| 32 | near-parallel-low-rank | mbcd-no-contradiction | 100/100 | 0.11248 | 5200 | 0.00% | 56.801 | 0.033475 | 83.00% | 1.41 | 100.00% | 57.79 | 0 |
| 32 | near-parallel-low-rank | mbcd-no-independent-validation | 100/100 | 0.099833 | 5200 | 0.00% | 52.357 | 0.035399 | 76.00% | 1.61 | 100.00% | 48.598 | 0 |
| 32 | multi-crossing-low-rank | asad | 100/100 | 2.1696 | 6240 | 0.00% | 88.8 | 6.8159e-13 | 2.00% | n/a | 0.00% | 122.97 | 0 |
| 32 | multi-crossing-low-rank | ransac | 0/100 | n/a | 14592 | 0.00% | n/a | n/a | n/a | n/a | n/a | 22.765 | 100 |
| 32 | multi-crossing-low-rank | star-probe | 45/100 | 1.768e-12 | 17997 | 45.00% | 1.3357e-7 | 1.4637e-12 | 0.00% | n/a | n/a | 24.94 | 55 |
| 32 | multi-crossing-low-rank | adaptive | 45/100 | 1.768e-12 | 19479 | 45.00% | 1.3357e-7 | 1.4637e-12 | 0.00% | n/a | n/a | 26.619 | 55 |
| 32 | multi-crossing-low-rank | bcad | 100/100 | 0.32163 | 6240 | 0.00% | 85.2 | 8.1046e-13 | 8.00% | n/a | n/a | 116.37 | 0 |
| 32 | multi-crossing-low-rank | mbcd-full | 100/100 | 0.14438 | 5200 | 0.00% | 29.718 | 0.00016154 | 81.00% | 11.17 | 2.00% | 77.762 | 0 |
| 32 | multi-crossing-low-rank | mbcd-single-rank | 100/100 | 0.14438 | 5200 | 0.00% | 29.718 | 0.00016154 | 81.00% | 11.17 | 2.00% | 77.778 | 0 |
| 32 | multi-crossing-low-rank | mbcd-no-contradiction | 100/100 | 0.17136 | 5200 | 0.00% | 23.742 | 0.0012221 | 65.00% | 8.14 | 0.00% | 110.56 | 0 |
| 32 | multi-crossing-low-rank | mbcd-no-independent-validation | 100/100 | 0.1485 | 5200 | 0.00% | 31.54 | 0.00035088 | 82.00% | 11.03 | 13.00% | 78.684 | 0 |
| 32 | small-amplitude-low-rank | asad | 100/100 | 3.1039e-10 | 3315 | 100.00% | 1.7761e-7 | 3.2977e-12 | 0.00% | n/a | 0.00% | 39.463 | 0 |
| 32 | small-amplitude-low-rank | ransac | 0/100 | n/a | 14592 | 0.00% | n/a | n/a | n/a | n/a | n/a | 10.314 | 100 |
| 32 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 15902 | 0.00% | n/a | n/a | n/a | n/a | n/a | 11 | 100 |
| 32 | small-amplitude-low-rank | adaptive | 0/100 | n/a | 15902 | 0.00% | n/a | n/a | n/a | n/a | n/a | 11.02 | 100 |
| 32 | small-amplitude-low-rank | bcad | 100/100 | 0.00034159 | 3557.4 | 93.00% | 6.3 | 7.3152e-12 | 0.00% | n/a | n/a | 50.189 | 0 |
| 32 | small-amplitude-low-rank | mbcd-full | 100/100 | 0.0010523 | 5200 | 0.00% | 12.052 | 0.023784 | 11.00% | 0.69 | 0.00% | 41.689 | 0 |
| 32 | small-amplitude-low-rank | mbcd-single-rank | 100/100 | 0.0010523 | 5200 | 0.00% | 12.052 | 0.023784 | 11.00% | 0.69 | 0.00% | 41.558 | 0 |
| 32 | small-amplitude-low-rank | mbcd-no-contradiction | 100/100 | 0.0010782 | 5200 | 0.00% | 14.405 | 0.021512 | 14.00% | 0.67 | 0.00% | 44.523 | 0 |
| 32 | small-amplitude-low-rank | mbcd-no-independent-validation | 100/100 | 0.0010435 | 5200 | 0.00% | 12.126 | 0.025234 | 11.00% | 0.73 | 0.00% | 41.812 | 0 |
| 32 | contaminated-local-probes | asad | 100/100 | 2.1715e-9 | 3315 | 100.00% | 1.241e-7 | 5.581e-11 | 0.00% | n/a | 100.00% | 39.91 | 0 |
| 32 | contaminated-local-probes | ransac | 0/100 | n/a | 14592 | 0.00% | n/a | n/a | n/a | n/a | n/a | 11.199 | 100 |
| 32 | contaminated-local-probes | star-probe | 100/100 | 3.55e-13 | 17362 | 100.00% | 1.6596e-7 | 6.6148e-13 | 0.00% | n/a | n/a | 18.92 | 0 |
| 32 | contaminated-local-probes | adaptive | 100/100 | 3.55e-13 | 18011 | 100.00% | 1.6596e-7 | 6.6148e-13 | 0.00% | n/a | n/a | 19.657 | 0 |
| 32 | contaminated-local-probes | bcad | 100/100 | 0.0081937 | 3553.6 | 95.00% | 4.5 | 5.0477e-10 | 2.00% | n/a | n/a | 52.671 | 0 |
| 32 | contaminated-local-probes | mbcd-full | 100/100 | 0.055152 | 5200 | 0.00% | 29.295 | 0.0282 | 30.00% | 1.01 | 100.00% | 43.627 | 0 |
| 32 | contaminated-local-probes | mbcd-single-rank | 100/100 | 0.055152 | 5200 | 0.00% | 29.295 | 0.0282 | 30.00% | 1.01 | 100.00% | 43.489 | 0 |
| 32 | contaminated-local-probes | mbcd-no-contradiction | 100/100 | 0.059697 | 5200 | 0.00% | 27.72 | 0.027835 | 28.00% | 1.02 | 100.00% | 48.522 | 0 |
| 32 | contaminated-local-probes | mbcd-no-independent-validation | 100/100 | 0.06889 | 5200 | 0.00% | 24.374 | 0.0295 | 24.00% | 1.04 | 100.00% | 43.935 | 0 |
| 32 | orthogonal-full-rank | asad | 100/100 | 1.2097 | 6240 | 0.00% | 90 | n/a | 0.00% | n/a | 27.00% | 244.7 | 0 |
| 32 | orthogonal-full-rank | ransac | 0/100 | n/a | 14592 | 0.00% | n/a | n/a | n/a | n/a | n/a | 248.72 | 100 |
| 32 | orthogonal-full-rank | star-probe | 100/100 | 2.0974 | 25168 | 0.00% | 50.006 | 1.0023e-11 | 0.00% | n/a | n/a | 215.55 | 0 |
| 32 | orthogonal-full-rank | adaptive | 100/100 | 2.0974 | 25168 | 0.00% | 62.184 | 1.0112e-11 | 100.00% | n/a | n/a | 285.9 | 0 |
| 32 | orthogonal-full-rank | bcad | 100/100 | 1.0545 | 6240 | 0.00% | 90 | n/a | 2.00% | n/a | n/a | 277.31 | 0 |
| 32 | orthogonal-full-rank | mbcd-full | 100/100 | 2.0895 | 5200 | 0.00% | 90 | n/a | 100.00% | 15.32 | 100.00% | 149.2 | 0 |
| 32 | orthogonal-full-rank | mbcd-single-rank | 100/100 | 2.0895 | 5200 | 0.00% | 90 | n/a | 100.00% | 15.32 | 100.00% | 149.14 | 0 |
| 32 | orthogonal-full-rank | mbcd-no-contradiction | 100/100 | 2.4814 | 5200 | 0.00% | 90 | n/a | 100.00% | 11.09 | 100.00% | 193.74 | 0 |
| 32 | orthogonal-full-rank | mbcd-no-independent-validation | 100/100 | 2.1583 | 5200 | 0.00% | 90 | n/a | 100.00% | 14.95 | 100.00% | 153.28 | 0 |

## Integrity and limitations

The independent verifier recomputes paired fixture hashes, evaluation-set hashes, seeds, budgets, actual-to-diagnostic oracle counts, completion rates, false-accepted rates, and the formal verdict from raw records. A method that throws remains unscored rather than receiving a synthetic MAE. Mean normal and offset error are calculated only for geometrically matched planes; misses and false accepts are reported separately. MBCD's reconstruction-complete rate is shown so a low MAE on incomplete outputs cannot be mistaken for reliable geometric recovery.
