# Active-Subspace Adaptive Discovery (ASAD): Experimental Report

**Run:** 2026-10-29T14-06-49-237Z  
**Recorded verdict:** partially-supported (the historical summary records 7 key cells)  
**Audited verdict under the stated criteria:** not-supported (0 of 8 key cells qualify)  
**Dimensions:** 6, 8, 16, 32  
**Scenarios:** isolated-low-rank, multi-crossing-low-rank, near-parallel-low-rank, small-amplitude-low-rank, contaminated-local-probes, orthogonal-full-rank, dense-crossing-full-rank  
**Seeds per dimension/scenario/method cell:** 100  
**Raw observations:** [asad-2026-10-29T14-06-49-237Z-raw.jsonl](../../scratch/results/asad/asad-2026-10-29T14-06-49-237Z-raw.jsonl)

## Research question and method

ASAD keeps local affine hypotheses built only from oracle values and finite-difference gradients. For each next center query, it scores a deterministic random candidate pool by the disagreement of those hypotheses' predicted values, with a novelty term. Gradient covariance estimates a changing active subspace; when rank is not identifiable from the available observations, ASAD switches to global exploration rather than treating a low observed rank as evidence of a low true rank. Candidate hyperplanes come from agreement between gradient jumps and two-sided affine intercepts. Within-side gradient variation lowers candidate confidence; ASAD refuses to label an uncertain or incomplete result as a complete reconstruction.

The solver receives no teacher weights, normals, offsets, scenario label, or test-point data. The benchmark uses deterministic teacher metadata only after solving to score structure and prediction. Evaluation points are generated from a separate seed and never passed to a solver.

Each method receives the same oracle-call cap for its dimension/scenario cell. The benchmark independently increments a wrapper around every solver oracle call; a returned diagnostic count is checked against that measurement. Evaluation-only teacher calls are outside the solver query budget. Runtime is measured around the solver call, not around test scoring.

## Results

| D | Scenario | Method | Scored/runs | MAE | Mean angle (deg) | Correct planes | False planes | Oracle calls | Runtime (ms) | Rank error | Rank uncertainty | Candidate uncertainty | Relative fit residual | Complete | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | contaminated-local-probes | adaptive | 100/100 | 1.8908e-11 | 0.00 | 1.00 | 0.00 | 10203 | 5.49 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 16 | contaminated-local-probes | asad | 100/100 | 0.000048227 | 0.00 | 1.00 | 0.00 | 891 | 8.34 | 100.0% | 0.000 | 0.000 | 0.0000 | 100.0% | 0.0% |
| 16 | contaminated-local-probes | ransac | 4/100 | 7.3879e-11 | 86.40 | 0.04 | 0.00 | 8502 | 3.85 | n/a | n/a | n/a | n/a | 4.0% | 96.0% |
| 16 | contaminated-local-probes | star-probe | 100/100 | 1.8908e-11 | 0.00 | 1.00 | 0.00 | 9874 | 5.37 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 16 | dense-crossing-full-rank | adaptive | 100/100 | 1.4091 | 35.83 | 9.63 | 1.67 | 12848 | 33.57 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 16 | dense-crossing-full-rank | asad | 100/100 | 0.92008 | 90.00 | 0.00 | 0.00 | 3168 | 75.79 | 19.0% | 0.490 | 1.000 | 0.4852 | 0.0% | 0.0% |
| 16 | dense-crossing-full-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 8448 | 29.65 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 16 | dense-crossing-full-rank | star-probe | 100/100 | 1.4091 | 2.53 | 15.55 | 0.00 | 12848 | 30.20 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 16 | isolated-low-rank | adaptive | 100/100 | 4.8585e-13 | 0.00 | 1.00 | 0.00 | 10202 | 5.19 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 16 | isolated-low-rank | asad | 100/100 | 0.000055132 | 0.00 | 1.00 | 0.00 | 891 | 8.19 | 0.0% | 0.000 | 0.000 | 0.0000 | 100.0% | 0.0% |
| 16 | isolated-low-rank | ransac | 6/100 | 9.2577e-10 | 84.60 | 0.06 | 0.00 | 8530 | 3.67 | n/a | n/a | n/a | n/a | 6.0% | 94.0% |
| 16 | isolated-low-rank | star-probe | 100/100 | 4.8585e-13 | 0.00 | 1.00 | 0.00 | 9874 | 5.10 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 16 | multi-crossing-low-rank | adaptive | 98/100 | 0.0058741 | 1.80 | 2.94 | 0.00 | 11293 | 8.94 | n/a | n/a | n/a | n/a | 97.0% | 2.0% |
| 16 | multi-crossing-low-rank | asad | 100/100 | 1.0724 | 88.50 | 0.05 | 0.01 | 3168 | 64.12 | 0.0% | 0.046 | 0.940 | 0.3054 | 0.0% | 0.0% |
| 16 | multi-crossing-low-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 8448 | 6.97 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 16 | multi-crossing-low-rank | star-probe | 98/100 | 1.3565e-12 | 1.80 | 2.94 | 0.00 | 10312 | 8.38 | n/a | n/a | n/a | n/a | 98.0% | 2.0% |
| 16 | near-parallel-low-rank | adaptive | 100/100 | 1.4793e-12 | 0.00 | 2.00 | 0.00 | 10717 | 6.98 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 16 | near-parallel-low-rank | asad | 100/100 | 0.51848 | 69.80 | 0.45 | 0.04 | 3124 | 60.24 | 100.0% | 0.000 | 0.540 | 0.1608 | 2.0% | 0.0% |
| 16 | near-parallel-low-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 8448 | 5.20 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 16 | near-parallel-low-rank | star-probe | 100/100 | 1.4793e-12 | 0.00 | 2.00 | 0.00 | 10059 | 6.66 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 16 | orthogonal-full-rank | adaptive | 100/100 | 1.4945 | 35.49 | 9.69 | 1.56 | 12848 | 33.16 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 16 | orthogonal-full-rank | asad | 100/100 | 0.90297 | 90.00 | 0.00 | 0.00 | 3168 | 75.61 | 22.0% | 0.475 | 1.000 | 0.4830 | 0.0% | 0.0% |
| 16 | orthogonal-full-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 8448 | 29.64 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 16 | orthogonal-full-rank | star-probe | 100/100 | 1.4945 | 3.21 | 15.43 | 0.00 | 12848 | 30.12 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 16 | small-amplitude-low-rank | adaptive | 3/100 | 4.2079e-12 | 87.30 | 0.03 | 0.00 | 9231 | 3.73 | n/a | n/a | n/a | n/a | 3.0% | 97.0% |
| 16 | small-amplitude-low-rank | asad | 100/100 | 2.9193e-7 | 0.00 | 1.00 | 0.00 | 891 | 8.12 | 0.0% | 0.000 | 0.000 | 0.0000 | 100.0% | 0.0% |
| 16 | small-amplitude-low-rank | ransac | 3/100 | 4.2079e-12 | 87.30 | 0.03 | 0.00 | 8489 | 3.60 | n/a | n/a | n/a | n/a | 3.0% | 97.0% |
| 16 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 9190 | 3.64 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | contaminated-local-probes | adaptive | 100/100 | 3.5500e-13 | 0.00 | 1.00 | 0.00 | 18011 | 19.11 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 32 | contaminated-local-probes | asad | 100/100 | 2.1715e-9 | 0.00 | 1.00 | 0.00 | 3315 | 41.59 | 100.0% | 0.000 | 0.000 | 0.0000 | 100.0% | 0.0% |
| 32 | contaminated-local-probes | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 10.99 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | contaminated-local-probes | star-probe | 100/100 | 3.5500e-13 | 0.00 | 1.00 | 0.00 | 17362 | 18.61 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 32 | dense-crossing-full-rank | adaptive | 100/100 | 2.0384 | 62.75 | 9.69 | 3.40 | 25168 | 278.87 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 32 | dense-crossing-full-rank | asad | 100/100 | 1.2263 | 90.00 | 0.00 | 0.00 | 6240 | 241.20 | 29.0% | 0.751 | 1.000 | 0.4354 | 0.0% | 0.0% |
| 32 | dense-crossing-full-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 236.97 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | dense-crossing-full-rank | star-probe | 100/100 | 2.0384 | 49.87 | 14.27 | 0.00 | 25168 | 207.31 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 32 | isolated-low-rank | adaptive | 100/100 | 4.1424e-13 | 0.00 | 1.00 | 0.00 | 18008 | 18.23 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 32 | isolated-low-rank | asad | 100/100 | 1.6181e-9 | 0.00 | 1.00 | 0.00 | 3315 | 40.77 | 0.0% | 0.000 | -0.000 | 0.0000 | 100.0% | 0.0% |
| 32 | isolated-low-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 10.25 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | isolated-low-rank | star-probe | 100/100 | 4.1424e-13 | 0.00 | 1.00 | 0.00 | 17362 | 17.80 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 32 | multi-crossing-low-rank | adaptive | 45/100 | 1.7680e-12 | 49.50 | 1.35 | 0.00 | 19479 | 26.02 | n/a | n/a | n/a | n/a | 45.0% | 55.0% |
| 32 | multi-crossing-low-rank | asad | 100/100 | 2.1696 | 88.80 | 0.04 | 0.02 | 6240 | 126.59 | 0.0% | 0.019 | 0.941 | 0.2440 | 0.0% | 0.0% |
| 32 | multi-crossing-low-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 22.57 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | multi-crossing-low-rank | star-probe | 45/100 | 1.7680e-12 | 49.50 | 1.35 | 0.00 | 17997 | 24.53 | n/a | n/a | n/a | n/a | 45.0% | 55.0% |
| 32 | near-parallel-low-rank | adaptive | 85/100 | 1.0141e-12 | 13.50 | 1.70 | 0.00 | 18747 | 22.89 | n/a | n/a | n/a | n/a | 85.0% | 15.0% |
| 32 | near-parallel-low-rank | asad | 100/100 | 0.55076 | 67.13 | 0.51 | 0.04 | 6109 | 111.96 | 100.0% | 0.000 | 0.511 | 0.1273 | 5.0% | 0.0% |
| 32 | near-parallel-low-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 14.05 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | near-parallel-low-rank | star-probe | 85/100 | 1.0141e-12 | 13.50 | 1.70 | 0.00 | 17561 | 21.96 | n/a | n/a | n/a | n/a | 85.0% | 15.0% |
| 32 | orthogonal-full-rank | adaptive | 100/100 | 2.0974 | 62.18 | 9.89 | 3.27 | 25168 | 280.30 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 32 | orthogonal-full-rank | asad | 100/100 | 1.2097 | 90.00 | 0.00 | 0.00 | 6240 | 245.54 | 27.0% | 0.743 | 1.000 | 0.4392 | 0.0% | 0.0% |
| 32 | orthogonal-full-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 245.63 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | orthogonal-full-rank | star-probe | 100/100 | 2.0974 | 50.01 | 14.22 | 0.00 | 25168 | 211.22 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 32 | small-amplitude-low-rank | adaptive | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 15902 | 10.96 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | small-amplitude-low-rank | asad | 100/100 | 3.1039e-10 | 0.00 | 1.00 | 0.00 | 3315 | 40.98 | 0.0% | 0.000 | -0.000 | 0.0000 | 100.0% | 0.0% |
| 32 | small-amplitude-low-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 14592 | 10.29 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 32 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 15902 | 10.93 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 6 | contaminated-local-probes | adaptive | 100/100 | 0.00011878 | 0.00 | 1.00 | 0.00 | 6904 | 1.97 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | contaminated-local-probes | asad | 100/100 | 0.0084010 | 1.47 | 0.99 | 0.00 | 201 | 2.20 | 100.0% | 0.023 | 0.015 | 0.0052 | 96.0% | 0.0% |
| 6 | contaminated-local-probes | ransac | 99/100 | 0.090569 | 77.40 | 0.14 | 0.85 | 6698 | 2.39 | n/a | n/a | n/a | n/a | 99.0% | 1.0% |
| 6 | contaminated-local-probes | star-probe | 100/100 | 0.00011878 | 0.00 | 1.00 | 0.00 | 6774 | 1.96 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | dense-crossing-full-rank | adaptive | 100/100 | 0.89579 | 0.30 | 5.98 | 0.00 | 7724 | 4.02 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 6 | dense-crossing-full-rank | asad | 100/100 | 0.53927 | 89.85 | 0.01 | 0.00 | 1248 | 34.91 | 6.0% | 0.253 | 0.990 | 0.4607 | 0.0% | 0.0% |
| 6 | dense-crossing-full-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 6144 | 3.60 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 6 | dense-crossing-full-rank | star-probe | 100/100 | 0.070525 | 0.30 | 5.98 | 0.02 | 7469 | 4.43 | n/a | n/a | n/a | n/a | 92.0% | 0.0% |
| 6 | isolated-low-rank | adaptive | 100/100 | 1.1267e-12 | 0.00 | 1.00 | 0.00 | 6860 | 1.92 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | isolated-low-rank | asad | 100/100 | 1.5190e-8 | 0.00 | 1.00 | 0.00 | 191 | 1.92 | 0.0% | 0.000 | -0.000 | 0.0000 | 100.0% | 0.0% |
| 6 | isolated-low-rank | ransac | 100/100 | 7.7566e-12 | 0.00 | 1.00 | 0.00 | 6704 | 2.20 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | isolated-low-rank | star-probe | 100/100 | 1.1267e-12 | 0.00 | 1.00 | 0.00 | 6730 | 1.90 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | multi-crossing-low-rank | adaptive | 100/100 | 0.026522 | 0.00 | 3.00 | 0.00 | 7375 | 2.85 | n/a | n/a | n/a | n/a | 96.0% | 0.0% |
| 6 | multi-crossing-low-rank | asad | 100/100 | 0.55230 | 84.32 | 0.19 | 0.00 | 1248 | 34.48 | 0.0% | 0.171 | 0.820 | 0.3042 | 0.0% | 0.0% |
| 6 | multi-crossing-low-rank | ransac | 7/100 | 2.5784e-11 | 83.70 | 0.21 | 0.00 | 6194 | 2.47 | n/a | n/a | n/a | n/a | 7.0% | 93.0% |
| 6 | multi-crossing-low-rank | star-probe | 100/100 | 1.8046e-12 | 0.00 | 3.00 | 0.00 | 6986 | 2.76 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | near-parallel-low-rank | adaptive | 100/100 | 3.3099e-12 | 0.00 | 2.00 | 0.00 | 7105 | 2.31 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | near-parallel-low-rank | asad | 100/100 | 0.30630 | 76.08 | 0.31 | 0.02 | 1227 | 33.81 | 100.0% | 0.000 | 0.696 | 0.2307 | 2.0% | 0.0% |
| 6 | near-parallel-low-rank | ransac | 75/100 | 1.0663e-11 | 22.50 | 1.50 | 0.00 | 6624 | 2.46 | n/a | n/a | n/a | n/a | 75.0% | 25.0% |
| 6 | near-parallel-low-rank | star-probe | 100/100 | 3.3099e-12 | 0.00 | 2.00 | 0.00 | 6845 | 2.26 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | orthogonal-full-rank | adaptive | 100/100 | 0.89249 | 0.15 | 5.99 | 0.01 | 7724 | 4.04 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 6 | orthogonal-full-rank | asad | 100/100 | 0.53431 | 89.70 | 0.02 | 0.00 | 1248 | 34.86 | 8.0% | 0.235 | 0.980 | 0.4508 | 0.0% | 0.0% |
| 6 | orthogonal-full-rank | ransac | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 6144 | 3.61 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 6 | orthogonal-full-rank | star-probe | 100/100 | 0.079588 | 0.90 | 5.94 | 0.06 | 7449 | 4.41 | n/a | n/a | n/a | n/a | 92.0% | 0.0% |
| 6 | small-amplitude-low-rank | adaptive | 100/100 | 0.0087404 | 0.00 | 1.00 | 0.00 | 7204 | 1.95 | n/a | n/a | n/a | n/a | 89.0% | 0.0% |
| 6 | small-amplitude-low-rank | asad | 100/100 | 5.1993e-10 | 0.00 | 1.00 | 0.00 | 182 | 1.49 | 0.0% | 0.000 | -0.000 | 0.0000 | 100.0% | 0.0% |
| 6 | small-amplitude-low-rank | ransac | 100/100 | 1.7614e-13 | 0.00 | 1.00 | 0.00 | 6704 | 1.90 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 6 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 6648 | 1.55 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |
| 8 | contaminated-local-probes | adaptive | 100/100 | 0.0000094700 | 0.00 | 1.00 | 0.00 | 7105 | 2.36 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | contaminated-local-probes | asad | 100/100 | 0.0028305 | 0.21 | 1.00 | 0.00 | 268 | 2.13 | 100.0% | 0.000 | 0.001 | 0.0011 | 100.0% | 0.0% |
| 8 | contaminated-local-probes | ransac | 100/100 | 0.083469 | 70.29 | 0.22 | 0.78 | 6864 | 3.64 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | contaminated-local-probes | star-probe | 100/100 | 0.0000094700 | 0.00 | 1.00 | 0.00 | 6935 | 2.33 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | dense-crossing-full-rank | adaptive | 100/100 | 1.0077 | 4.39 | 7.61 | 0.05 | 8224 | 6.46 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 8 | dense-crossing-full-rank | asad | 100/100 | 0.60455 | 89.66 | 0.03 | 0.00 | 1632 | 41.43 | 6.0% | 0.224 | 0.970 | 0.4577 | 0.0% | 0.0% |
| 8 | dense-crossing-full-rank | ransac | 30/100 | 0.37303 | 90.00 | 0.00 | 2.40 | 6528 | 7.50 | n/a | n/a | n/a | n/a | 30.0% | 70.0% |
| 8 | dense-crossing-full-rank | star-probe | 100/100 | 0.28530 | 0.56 | 7.95 | 0.05 | 8045 | 7.04 | n/a | n/a | n/a | n/a | 72.0% | 0.0% |
| 8 | isolated-low-rank | adaptive | 100/100 | 8.6732e-13 | 0.00 | 1.00 | 0.00 | 7068 | 2.11 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | isolated-low-rank | asad | 100/100 | 1.8026e-8 | 0.00 | 1.00 | 0.00 | 273 | 2.12 | 0.0% | 0.000 | -0.000 | 0.0000 | 100.0% | 0.0% |
| 8 | isolated-low-rank | ransac | 98/100 | 1.9287e-11 | 1.80 | 0.98 | 0.00 | 6850 | 2.28 | n/a | n/a | n/a | n/a | 98.0% | 2.0% |
| 8 | isolated-low-rank | star-probe | 100/100 | 8.6732e-13 | 0.00 | 1.00 | 0.00 | 6898 | 2.09 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | multi-crossing-low-rank | adaptive | 100/100 | 0.047683 | 0.00 | 3.00 | 0.00 | 7703 | 3.51 | n/a | n/a | n/a | n/a | 91.0% | 0.0% |
| 8 | multi-crossing-low-rank | asad | 100/100 | 0.73720 | 85.50 | 0.15 | 0.02 | 1632 | 40.46 | 0.0% | 0.153 | 0.830 | 0.3124 | 0.0% | 0.0% |
| 8 | multi-crossing-low-rank | ransac | 2/100 | 0.20528 | 89.70 | 0.01 | 0.05 | 6162 | 3.06 | n/a | n/a | n/a | n/a | 2.0% | 98.0% |
| 8 | multi-crossing-low-rank | star-probe | 100/100 | 1.6036e-12 | 0.00 | 3.00 | 0.00 | 7201 | 3.41 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | near-parallel-low-rank | adaptive | 100/100 | 2.7209e-12 | 0.00 | 2.00 | 0.00 | 7366 | 2.80 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | near-parallel-low-rank | asad | 100/100 | 0.19947 | 68.02 | 0.49 | 0.02 | 1622 | 39.07 | 100.0% | 0.000 | 0.512 | 0.1659 | 1.0% | 0.0% |
| 8 | near-parallel-low-rank | ransac | 19/100 | 1.1130e-10 | 72.90 | 0.38 | 0.00 | 6296 | 2.50 | n/a | n/a | n/a | n/a | 19.0% | 81.0% |
| 8 | near-parallel-low-rank | star-probe | 100/100 | 2.7209e-12 | 0.00 | 2.00 | 0.00 | 7026 | 2.73 | n/a | n/a | n/a | n/a | 100.0% | 0.0% |
| 8 | orthogonal-full-rank | adaptive | 100/100 | 1.0371 | 4.16 | 7.63 | 0.03 | 8224 | 6.39 | n/a | n/a | n/a | n/a | 0.0% | 0.0% |
| 8 | orthogonal-full-rank | asad | 100/100 | 0.63484 | 89.78 | 0.02 | 0.00 | 1632 | 41.38 | 14.0% | 0.295 | 0.980 | 0.4635 | 0.0% | 0.0% |
| 8 | orthogonal-full-rank | ransac | 23/100 | 0.37308 | 90.00 | 0.00 | 1.84 | 6438 | 7.26 | n/a | n/a | n/a | n/a | 23.0% | 77.0% |
| 8 | orthogonal-full-rank | star-probe | 100/100 | 0.31905 | 0.23 | 7.98 | 0.02 | 8059 | 7.06 | n/a | n/a | n/a | n/a | 70.0% | 0.0% |
| 8 | small-amplitude-low-rank | adaptive | 93/100 | 0.0013412 | 6.30 | 0.93 | 0.00 | 7315 | 2.34 | n/a | n/a | n/a | n/a | 92.0% | 7.0% |
| 8 | small-amplitude-low-rank | asad | 100/100 | 1.2998e-9 | 0.00 | 1.00 | 0.00 | 259 | 1.91 | 0.0% | 0.000 | -0.000 | 0.0000 | 100.0% | 0.0% |
| 8 | small-amplitude-low-rank | ransac | 93/100 | 7.7810e-13 | 6.30 | 0.93 | 0.00 | 6814 | 2.26 | n/a | n/a | n/a | n/a | 93.0% | 7.0% |
| 8 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 90.00 | 0.00 | 0.00 | 6646 | 1.76 | n/a | n/a | n/a | n/a | 0.0% | 100.0% |

## Acceptance criteria

- High-dimensional key cells are dimensions 16 and 32 for isolated, multi-crossing, near-parallel, and small-amplitude low-rank scenarios. Each has 100 seeds. A cell qualifies if ASAD uses at least 20% fewer calls than an existing mode whose MAE is within 5% relative difference (symmetric denominator; absolute floor 1e-8), or has at least 25% lower MAE at the common call cap.
- The historical summary reports 7 qualifying key cells. An independent recomputation from all 11,200 raw rows finds 0 of 8. In the 7 cells with at least one scored baseline, no baseline MAE is within the stated 5% comparison tolerance of ASAD; the remaining cell has no scored baseline. Therefore none has a valid query-savings comparison, and none meets the MAE-reduction criterion.
- The discrepancy is a scoring defect, not a raw-data discrepancy: the original verdict calculation used `Infinity` when there was no MAE-comparable baseline, so `1 - asadCalls / Infinity` incorrectly counted as 100% query reduction. The corrected calculation treats an empty comparison set as no query-saving result. Raw JSONL and historical summary JSON are retained unchanged.
- Audited key-cell results:

| D | Scenario | ASAD MAE | Best scored baseline MAE | MAE-comparable baseline modes | Query-savings condition | Qualifies |
|---:|---|---:|---:|---|---|---|
| 16 | isolated-low-rank | 5.5132e-5 | 4.8585e-13 | none | not evaluable | no |
| 16 | multi-crossing-low-rank | 1.0724 | 1.3565e-12 | none | not evaluable | no |
| 16 | near-parallel-low-rank | 0.51848 | 1.4793e-12 | none | not evaluable | no |
| 16 | small-amplitude-low-rank | 2.9193e-7 | 4.2079e-12 | none | not evaluable | no |
| 32 | isolated-low-rank | 1.6181e-9 | 4.1424e-13 | none | not evaluable | no |
| 32 | multi-crossing-low-rank | 2.1696 | 1.7680e-12 | none | not evaluable | no |
| 32 | near-parallel-low-rank | 0.55076 | 1.0141e-12 | none | not evaluable | no |
| 32 | small-amplitude-low-rank | 3.1039e-10 | no scored baseline | none | not evaluable | no |

- Audited key cells meeting either numerical criterion: 0 of 8.
- Non-key required cells where ASAD mean MAE is over 5% worse than the best scored baseline (baseline-relative, with the same absolute floor): 13 of 20 scored cells; 0 cells lack a comparable scored baseline or ASAD output.
- Actual oracle-budget overruns across all records: 0.
- All required seed cells have at least 100 seeds: yes.

## Conclusion

**not-supported under the stated key-cell success criteria**. The previously reported `partially-supported` result is not reproducible from the raw records under its own MAE comparability requirement. This audit does not alter the historical raw observations or summary JSON; it corrects the interpretation and identifies the scoring defect. The new BCAD campaign uses the corrected empty-comparison behavior.

## Evidence and limits

- Raw per-run records: [asad-2026-10-29T14-06-49-237Z-raw.jsonl](../../scratch/results/asad/asad-2026-10-29T14-06-49-237Z-raw.jsonl)
- Summary JSON: [summary-2026-10-29T14-06-49-237Z.json](../../scratch/results/asad/summary-2026-10-29T14-06-49-237Z.json)
- Historical verdict audit: [audit-asad-results.ts](../../scripts/audit-asad-results.ts)
- Oracle-call reconciliation is recorded per run as actual wrapper invocations versus solver diagnostics; records without diagnostics are explicitly marked unavailable.
- ReLU teacher and contamination generator are deterministic from the benchmark code and recorded seed. The test corpus is independent of solver acquisition.
- The experiment is synthetic and scalar-output only. It does not establish performance on deep networks, noisy/quantized real APIs, or high-dimensional language-model behavior.
- Full-rank cases are expensive and expose finite-sample and support limitations; missing predictions from solver exceptions are not replaced with invented MAE values and remain null in the raw records.
