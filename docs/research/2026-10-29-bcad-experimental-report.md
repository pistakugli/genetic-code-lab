# Boundary-Conditioned Active Discovery (BCAD): Focused Experiment

**Run:** 2026-10-29T14-46-37-077Z  
**Verdict:** not-supported  
**Dimensions:** 16, 32  
**Scenarios:** near-parallel-low-rank, multi-crossing-low-rank, small-amplitude-low-rank, contaminated-local-probes, orthogonal-full-rank  
**Methods/ablations:** asad, ransac, star-probe, adaptive, bcad-boundary-delayed, bcad-prediction-control, bcad-gradient-jump, bcad-full-dimensional, bcad-fixed-initial-rank, bcad-stable-adaptive-rank, bcad-single-hypothesis-acquisition  
**Seeds per dimension/scenario/method cell:** 100  
**Raw observations:** [bcad-2026-10-29T14-46-37-077Z-raw.jsonl](../../scratch/results/bcad/bcad-2026-10-29T14-46-37-077Z-raw.jsonl)

## Research question and implementation

BCAD maintains several candidate boundary planes inferred from oracle-only finite-difference gradient jumps and affine intercept differences. Near-parallel candidates with distinct offsets remain separate. Query acquisition can score the same candidate pool by (A) local affine prediction disagreement, (B) disagreement between competing boundary-conditioned gradient predictions plus probability of crossing a candidate kink, or (C) local gradient-jump detectability. Candidate boundary hypotheses do not imply that the active subspace or all facets have been identified.

One gradient observation costs exactly $2d+1$ actual oracle calls: one center value and two coordinate samples per input dimension. A single wrapper counts every call; full stencils are started only when the remaining budget can cover them. Model fitting performs no additional oracle calls. Solver input contains no teacher weights, normals, offsets, scenario labels, model hashes, or evaluation points.

## Fairness and evaluation

All methods share the deterministic teacher generated from the same dimension, scenario, and seed, the same oracle budget from the existing ASAD benchmark budget function, the same solver random seed, and the same independently generated 192-point test set. Teacher metadata and test points are used only by the benchmark after the solver returns. SHA-256 fingerprints of the teacher fixture and test points are recorded per run and independently reconciled by the verifier. Runtime measures only the solver call.

The fixed-rank arm freezes its gradient-covariance rank estimate after its initial sample window. The stable-adaptive arm changes rank only after the estimate and normalized spectrum remain stable across repeated updates. The delayed-reduction arm retains full-dimensional exploration until enough supported boundary candidates exist and their uncertainty is low. The full-dimensional arm ablates subspace restriction. The single-hypothesis acquisition arm tests the contribution of comparing alternatives. The BCAD prediction-control arm isolates acquisition scoring while retaining BCAD candidate generation; the unchanged ASAD arm is also included as the original algorithm baseline.

## Acceptance criteria and primary comparison

- The three primary problem classes are near-parallel facets, multiple crossings, and small-amplitude jumps at dimensions 16 and 32. A cell passes if primary BCAD has at least 25% lower MAE than the best scored existing mode, or uses at least 30% fewer calls than an existing mode within 5% MAE of the best existing mode while BCAD MAE is no more than 5% worse than that best MAE.
- False-accepted boundary rate must not exceed the lowest rate among the best-MAE baseline comparison group by more than one percentage point.
- At least two of the three primary problem classes must pass at both dimensions, with 100 seeds per cell and zero actual oracle-budget overruns.
- Best existing modes are selected from ASAD, RANSAC, Star-Probe, and Adaptive. An unscored method is not treated as a zero-MAE result. Empty MAE-comparable sets do not count as query savings.
- Relative MAE reduction is shown as n/a when the baseline is at or below the $1e-8$ absolute floor; percentage change from a near-zero error is not meaningful.

**Primary problem classes passing at both dimensions:** none.  
**Overall verdict:** not-supported.  
**Actual budget overruns:** 0.  
**Required seeds per cell reached:** yes.

| D | Scenario | Best baseline | BCAD MAE | Baseline MAE | Call reduction | MAE reduction | False-accept increase | Pass | Reason |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 16 | near-parallel-low-rank | star-probe | 0.026189 | 1.4793e-12 | 76.0% | n/a | 0.00 pp | no | neither_performance_threshold_met |
| 16 | multi-crossing-low-rank | star-probe | 0.27624 | 1.3565e-12 | 69.7% | n/a | 18.00 pp | no | false_accepted_plane_rate_increase_over_1_percentage_point |
| 16 | small-amplitude-low-rank | ransac | 0.00030953 | 4.2079e-12 | 86.4% | n/a | 0.00 pp | no | neither_performance_threshold_met |
| 32 | near-parallel-low-rank | star-probe | 0.030583 | 1.0141e-12 | 72.5% | n/a | 0.00 pp | no | neither_performance_threshold_met |
| 32 | multi-crossing-low-rank | star-probe | 0.32163 | 1.7680e-12 | 65.3% | n/a | 8.00 pp | no | false_accepted_plane_rate_increase_over_1_percentage_point |
| 32 | small-amplitude-low-rank | asad | 0.00034159 | 3.1039e-10 | -7.3% | n/a | 0.00 pp | no | neither_performance_threshold_met |

## Results by method and ablation

| D | Scenario | Method | Scored/runs | MAE | Oracle calls | Angle (deg) | Correct planes | Mean false planes | Runs with false plane | Rank error | Estimated rank | Query rank | Runtime (ms) | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | contaminated-local-probes | adaptive | 100/100 | 1.8908e-11 | 10203 | 0.00 | 1.00 | 0.000 | 0.0% | n/a | n/a | n/a | 3.76 | 0% |
| 16 | contaminated-local-probes | asad | 100/100 | 0.000048227 | 891 | 0.00 | 1.00 | 0.000 | 0.0% | 100.0% | 1.00 | n/a | 7.92 | 0% |
| 16 | contaminated-local-probes | bcad-boundary-delayed | 100/100 | 0.0091804 | 1162 | 5.41 | 0.94 | 0.000 | 0.0% | 100.0% | 1.02 | 2.07 | 14.81 | 0% |
| 16 | contaminated-local-probes | bcad-fixed-initial-rank | 100/100 | 0.0094706 | 1139 | 5.41 | 0.94 | 0.000 | 0.0% | 100.0% | 1.00 | 1.00 | 14.14 | 0% |
| 16 | contaminated-local-probes | bcad-full-dimensional | 100/100 | 0.0073872 | 1037 | 4.51 | 0.95 | 0.000 | 0.0% | 100.0% | 1.00 | 16.00 | 11.68 | 0% |
| 16 | contaminated-local-probes | bcad-gradient-jump | 100/100 | 0.011452 | 1206 | 7.23 | 0.92 | 0.010 | 1.0% | 100.0% | 1.01 | 2.21 | 15.62 | 0% |
| 16 | contaminated-local-probes | bcad-prediction-control | 100/100 | 0.11841 | 1215 | 7.21 | 0.92 | 0.000 | 0.0% | 100.0% | 1.00 | 2.20 | 13.22 | 0% |
| 16 | contaminated-local-probes | bcad-single-hypothesis-acquisition | 100/100 | 0.014799 | 1217 | 9.01 | 0.90 | 0.000 | 0.0% | 100.0% | 1.02 | 2.67 | 14.34 | 0% |
| 16 | contaminated-local-probes | bcad-stable-adaptive-rank | 100/100 | 0.012051 | 1183 | 7.24 | 0.92 | 0.000 | 0.0% | 100.0% | 1.02 | 1.16 | 14.84 | 0% |
| 16 | contaminated-local-probes | ransac | 4/100 | 7.3879e-11 | 8502 | 86.40 | 0.04 | 0.000 | 0.0% | n/a | n/a | n/a | 2.30 | 96% |
| 16 | contaminated-local-probes | star-probe | 100/100 | 1.8908e-11 | 9874 | 0.00 | 1.00 | 0.000 | 0.0% | n/a | n/a | n/a | 3.64 | 0% |
| 16 | multi-crossing-low-rank | adaptive | 98/100 | 0.0058741 | 11293 | 1.80 | 2.94 | 0.000 | 0.0% | n/a | n/a | n/a | 4.43 | 2% |
| 16 | multi-crossing-low-rank | asad | 100/100 | 1.0724 | 3168 | 88.50 | 0.05 | 0.010 | 1.0% | 0.0% | 3.00 | n/a | 61.84 | 0% |
| 16 | multi-crossing-low-rank | bcad-boundary-delayed | 100/100 | 0.27624 | 3128 | 81.36 | 0.29 | 0.190 | 18.0% | 0.0% | 3.00 | 15.74 | 62.27 | 0% |
| 16 | multi-crossing-low-rank | bcad-fixed-initial-rank | 100/100 | 0.27040 | 3130 | 78.91 | 0.37 | 0.210 | 18.0% | 0.0% | 3.00 | 2.86 | 61.68 | 0% |
| 16 | multi-crossing-low-rank | bcad-full-dimensional | 100/100 | 0.27720 | 3168 | 81.36 | 0.29 | 0.170 | 17.0% | 0.0% | 3.00 | 16.00 | 62.27 | 0% |
| 16 | multi-crossing-low-rank | bcad-gradient-jump | 100/100 | 0.27544 | 3146 | 80.18 | 0.33 | 0.250 | 20.0% | 0.0% | 3.00 | 15.61 | 61.73 | 0% |
| 16 | multi-crossing-low-rank | bcad-prediction-control | 100/100 | 0.33286 | 3060 | 73.50 | 0.55 | 0.480 | 37.0% | 0.0% | 3.00 | 13.79 | 54.17 | 0% |
| 16 | multi-crossing-low-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.27087 | 3168 | 79.80 | 0.34 | 0.250 | 22.0% | 0.0% | 3.00 | 15.48 | 51.57 | 0% |
| 16 | multi-crossing-low-rank | bcad-stable-adaptive-rank | 100/100 | 0.26914 | 3149 | 81.00 | 0.30 | 0.210 | 20.0% | 0.0% | 3.00 | 3.00 | 61.28 | 0% |
| 16 | multi-crossing-low-rank | ransac | 0/100 | n/a | 8448 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 3.58 | 100% |
| 16 | multi-crossing-low-rank | star-probe | 98/100 | 1.3565e-12 | 10312 | 1.80 | 2.94 | 0.000 | 0.0% | n/a | n/a | n/a | 4.21 | 2% |
| 16 | near-parallel-low-rank | adaptive | 100/100 | 1.4793e-12 | 10717 | 0.00 | 2.00 | 0.000 | 0.0% | n/a | n/a | n/a | 3.85 | 0% |
| 16 | near-parallel-low-rank | asad | 100/100 | 0.51848 | 3124 | 69.80 | 0.45 | 0.040 | 4.0% | 100.0% | 1.00 | n/a | 58.48 | 0% |
| 16 | near-parallel-low-rank | bcad-boundary-delayed | 100/100 | 0.026189 | 2419 | 21.60 | 1.52 | 0.000 | 0.0% | 100.0% | 1.00 | 8.05 | 36.86 | 0% |
| 16 | near-parallel-low-rank | bcad-fixed-initial-rank | 100/100 | 0.027587 | 2557 | 22.10 | 1.51 | 0.000 | 0.0% | 100.0% | 1.00 | 1.00 | 38.81 | 0% |
| 16 | near-parallel-low-rank | bcad-full-dimensional | 100/100 | 0.025870 | 2391 | 21.15 | 1.53 | 0.000 | 0.0% | 100.0% | 1.00 | 16.00 | 35.88 | 0% |
| 16 | near-parallel-low-rank | bcad-gradient-jump | 100/100 | 0.024087 | 2400 | 20.70 | 1.54 | 0.000 | 0.0% | 100.0% | 1.00 | 7.90 | 36.23 | 0% |
| 16 | near-parallel-low-rank | bcad-prediction-control | 100/100 | 0.038214 | 2509 | 24.30 | 1.46 | 0.000 | 0.0% | 100.0% | 1.00 | 8.95 | 36.57 | 0% |
| 16 | near-parallel-low-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.027886 | 2503 | 23.85 | 1.47 | 0.000 | 0.0% | 100.0% | 1.00 | 8.95 | 36.37 | 0% |
| 16 | near-parallel-low-rank | bcad-stable-adaptive-rank | 100/100 | 0.035786 | 2558 | 27.45 | 1.39 | 0.000 | 0.0% | 100.0% | 1.00 | 1.00 | 38.74 | 0% |
| 16 | near-parallel-low-rank | ransac | 0/100 | n/a | 8448 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 2.84 | 100% |
| 16 | near-parallel-low-rank | star-probe | 100/100 | 1.4793e-12 | 10059 | 0.00 | 2.00 | 0.000 | 0.0% | n/a | n/a | n/a | 3.82 | 0% |
| 16 | orthogonal-full-rank | adaptive | 100/100 | 1.4945 | 12848 | 35.49 | 9.69 | 1.560 | 99.0% | n/a | n/a | n/a | 11.25 | 0% |
| 16 | orthogonal-full-rank | asad | 100/100 | 0.90297 | 3168 | 90.00 | 0.00 | 0.000 | 0.0% | 22.0% | 15.73 | n/a | 68.55 | 0% |
| 16 | orthogonal-full-rank | bcad-boundary-delayed | 100/100 | 0.67812 | 3168 | 90.00 | 0.00 | 0.150 | 15.0% | 0.0% | 16.00 | 16.00 | 115.40 | 0% |
| 16 | orthogonal-full-rank | bcad-fixed-initial-rank | 100/100 | 0.67901 | 3168 | 90.00 | 0.00 | 0.210 | 20.0% | 0.0% | 16.00 | 5.54 | 114.77 | 0% |
| 16 | orthogonal-full-rank | bcad-full-dimensional | 100/100 | 0.67812 | 3168 | 90.00 | 0.00 | 0.150 | 15.0% | 0.0% | 16.00 | 16.00 | 115.17 | 0% |
| 16 | orthogonal-full-rank | bcad-gradient-jump | 100/100 | 0.67747 | 3168 | 90.00 | 0.00 | 0.210 | 19.0% | 0.0% | 16.00 | 16.00 | 115.63 | 0% |
| 16 | orthogonal-full-rank | bcad-prediction-control | 100/100 | 0.70595 | 3168 | 90.00 | 0.00 | 0.220 | 22.0% | 0.0% | 16.00 | 16.00 | 65.17 | 0% |
| 16 | orthogonal-full-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.67373 | 3168 | 90.00 | 0.00 | 0.100 | 10.0% | 0.0% | 16.00 | 16.00 | 61.30 | 0% |
| 16 | orthogonal-full-rank | bcad-stable-adaptive-rank | 100/100 | 0.67612 | 3168 | 90.00 | 0.00 | 0.130 | 13.0% | 0.0% | 16.00 | 16.00 | 115.23 | 0% |
| 16 | orthogonal-full-rank | ransac | 0/100 | n/a | 8448 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 15.30 | 100% |
| 16 | orthogonal-full-rank | star-probe | 100/100 | 1.4945 | 12848 | 3.21 | 15.43 | 0.000 | 0.0% | n/a | n/a | n/a | 8.19 | 0% |
| 16 | small-amplitude-low-rank | adaptive | 3/100 | 4.2079e-12 | 9231 | 87.30 | 0.03 | 0.000 | 0.0% | n/a | n/a | n/a | 1.97 | 97% |
| 16 | small-amplitude-low-rank | asad | 100/100 | 2.9193e-7 | 891 | 0.00 | 1.00 | 0.000 | 0.0% | 0.0% | 1.00 | n/a | 7.86 | 0% |
| 16 | small-amplitude-low-rank | bcad-boundary-delayed | 100/100 | 0.00030953 | 1153 | 6.30 | 0.93 | 0.000 | 0.0% | 0.0% | 1.00 | 2.05 | 12.96 | 0% |
| 16 | small-amplitude-low-rank | bcad-fixed-initial-rank | 100/100 | 0.00024976 | 1066 | 4.52 | 0.95 | 0.000 | 0.0% | 0.0% | 1.00 | 1.30 | 11.01 | 0% |
| 16 | small-amplitude-low-rank | bcad-full-dimensional | 100/100 | 0.00027671 | 1139 | 5.45 | 0.94 | 0.000 | 0.0% | 0.0% | 1.00 | 16.00 | 12.41 | 0% |
| 16 | small-amplitude-low-rank | bcad-gradient-jump | 100/100 | 0.00031476 | 1153 | 6.30 | 0.93 | 0.000 | 0.0% | 0.0% | 1.00 | 2.05 | 12.87 | 0% |
| 16 | small-amplitude-low-rank | bcad-prediction-control | 100/100 | 0.021713 | 1108 | 5.40 | 0.94 | 0.000 | 0.0% | 0.0% | 1.00 | 1.90 | 11.26 | 0% |
| 16 | small-amplitude-low-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.00030680 | 1142 | 6.30 | 0.93 | 0.000 | 0.0% | 0.0% | 1.00 | 2.05 | 12.60 | 0% |
| 16 | small-amplitude-low-rank | bcad-stable-adaptive-rank | 100/100 | 0.00040577 | 1174 | 8.14 | 0.91 | 0.000 | 0.0% | 0.0% | 1.00 | 1.00 | 13.21 | 0% |
| 16 | small-amplitude-low-rank | ransac | 3/100 | 4.2079e-12 | 8489 | 87.30 | 0.03 | 0.000 | 0.0% | n/a | n/a | n/a | 1.93 | 97% |
| 16 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 9190 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 1.88 | 100% |
| 32 | contaminated-local-probes | adaptive | 100/100 | 3.5500e-13 | 18011 | 0.00 | 1.00 | 0.000 | 0.0% | n/a | n/a | n/a | 12.61 | 0% |
| 32 | contaminated-local-probes | asad | 100/100 | 2.1715e-9 | 3315 | 0.00 | 1.00 | 0.000 | 0.0% | 100.0% | 1.00 | n/a | 37.37 | 0% |
| 32 | contaminated-local-probes | bcad-boundary-delayed | 100/100 | 0.0081937 | 3554 | 4.50 | 0.95 | 0.020 | 2.0% | 100.0% | 1.00 | 1.93 | 49.44 | 0% |
| 32 | contaminated-local-probes | bcad-fixed-initial-rank | 100/100 | 0.011640 | 3698 | 6.32 | 0.93 | 0.020 | 2.0% | 100.0% | 1.00 | 1.00 | 52.48 | 0% |
| 32 | contaminated-local-probes | bcad-full-dimensional | 100/100 | 0.0064961 | 3511 | 3.60 | 0.96 | 0.000 | 0.0% | 100.0% | 1.00 | 32.00 | 48.50 | 0% |
| 32 | contaminated-local-probes | bcad-gradient-jump | 100/100 | 0.0032386 | 3515 | 1.80 | 0.98 | 0.000 | 0.0% | 100.0% | 1.00 | 1.93 | 47.97 | 0% |
| 32 | contaminated-local-probes | bcad-prediction-control | 100/100 | 0.11896 | 3539 | 5.40 | 0.94 | 0.000 | 0.0% | 100.0% | 1.00 | 2.86 | 36.82 | 0% |
| 32 | contaminated-local-probes | bcad-single-hypothesis-acquisition | 100/100 | 0.0032996 | 3637 | 1.80 | 0.98 | 0.000 | 0.0% | 100.0% | 1.00 | 1.93 | 51.20 | 0% |
| 32 | contaminated-local-probes | bcad-stable-adaptive-rank | 100/100 | 0.0050045 | 3560 | 2.70 | 0.97 | 0.000 | 0.0% | 100.0% | 1.00 | 1.00 | 49.21 | 0% |
| 32 | contaminated-local-probes | ransac | 0/100 | n/a | 14592 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 5.34 | 100% |
| 32 | contaminated-local-probes | star-probe | 100/100 | 3.5500e-13 | 17362 | 0.00 | 1.00 | 0.000 | 0.0% | n/a | n/a | n/a | 12.21 | 0% |
| 32 | multi-crossing-low-rank | adaptive | 45/100 | 1.7680e-12 | 19479 | 49.50 | 1.35 | 0.000 | 0.0% | n/a | n/a | n/a | 11.13 | 55.00000000000001% |
| 32 | multi-crossing-low-rank | asad | 100/100 | 2.1696 | 6240 | 88.80 | 0.04 | 0.020 | 2.0% | 0.0% | 3.00 | n/a | 114.20 | 0% |
| 32 | multi-crossing-low-rank | bcad-boundary-delayed | 100/100 | 0.32163 | 6240 | 85.20 | 0.16 | 0.100 | 8.0% | 0.0% | 3.00 | 32.00 | 108.41 | 0% |
| 32 | multi-crossing-low-rank | bcad-fixed-initial-rank | 100/100 | 0.34651 | 6240 | 84.62 | 0.18 | 0.230 | 21.0% | 0.0% | 3.00 | 2.82 | 106.22 | 0% |
| 32 | multi-crossing-low-rank | bcad-full-dimensional | 100/100 | 0.31911 | 6213 | 84.60 | 0.18 | 0.110 | 9.0% | 0.0% | 3.00 | 32.00 | 107.20 | 0% |
| 32 | multi-crossing-low-rank | bcad-gradient-jump | 100/100 | 0.31479 | 6213 | 83.43 | 0.22 | 0.160 | 15.0% | 0.0% | 3.00 | 31.42 | 107.94 | 0% |
| 32 | multi-crossing-low-rank | bcad-prediction-control | 100/100 | 0.42810 | 6199 | 73.20 | 0.56 | 0.360 | 30.0% | 0.0% | 3.00 | 28.52 | 92.61 | 0% |
| 32 | multi-crossing-low-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.31548 | 6240 | 79.23 | 0.36 | 0.250 | 23.0% | 0.0% | 3.00 | 32.00 | 89.75 | 0% |
| 32 | multi-crossing-low-rank | bcad-stable-adaptive-rank | 100/100 | 0.34451 | 6240 | 84.62 | 0.18 | 0.170 | 16.0% | 0.0% | 3.00 | 3.00 | 106.37 | 0% |
| 32 | multi-crossing-low-rank | ransac | 0/100 | n/a | 14592 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 11.15 | 100% |
| 32 | multi-crossing-low-rank | star-probe | 45/100 | 1.7680e-12 | 17997 | 49.50 | 1.35 | 0.000 | 0.0% | n/a | n/a | n/a | 10.71 | 55.00000000000001% |
| 32 | near-parallel-low-rank | adaptive | 85/100 | 1.0141e-12 | 18747 | 13.50 | 1.70 | 0.000 | 0.0% | n/a | n/a | n/a | 12.46 | 15% |
| 32 | near-parallel-low-rank | asad | 100/100 | 0.55076 | 6109 | 67.13 | 0.51 | 0.040 | 4.0% | 100.0% | 1.00 | n/a | 101.35 | 0% |
| 32 | near-parallel-low-rank | bcad-boundary-delayed | 100/100 | 0.030583 | 4835 | 19.80 | 1.56 | 0.000 | 0.0% | 100.0% | 1.00 | 13.40 | 64.40 | 0% |
| 32 | near-parallel-low-rank | bcad-fixed-initial-rank | 100/100 | 0.050755 | 5077 | 22.96 | 1.49 | 0.000 | 0.0% | 100.0% | 1.00 | 1.00 | 66.55 | 0% |
| 32 | near-parallel-low-rank | bcad-full-dimensional | 100/100 | 0.029876 | 4777 | 19.35 | 1.57 | 0.000 | 0.0% | 100.0% | 1.00 | 32.00 | 63.10 | 0% |
| 32 | near-parallel-low-rank | bcad-gradient-jump | 100/100 | 0.027835 | 4677 | 17.10 | 1.62 | 0.000 | 0.0% | 100.0% | 1.00 | 11.85 | 61.43 | 0% |
| 32 | near-parallel-low-rank | bcad-prediction-control | 100/100 | 0.057223 | 5111 | 25.20 | 1.44 | 0.000 | 0.0% | 100.0% | 1.00 | 16.81 | 62.11 | 0% |
| 32 | near-parallel-low-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.046307 | 5301 | 27.90 | 1.38 | 0.000 | 0.0% | 100.0% | 1.00 | 18.98 | 71.65 | 0% |
| 32 | near-parallel-low-rank | bcad-stable-adaptive-rank | 100/100 | 0.041920 | 5082 | 22.95 | 1.49 | 0.010 | 1.0% | 100.0% | 1.00 | 1.00 | 67.51 | 0% |
| 32 | near-parallel-low-rank | ransac | 0/100 | n/a | 14592 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 5.54 | 100% |
| 32 | near-parallel-low-rank | star-probe | 85/100 | 1.0141e-12 | 17561 | 13.50 | 1.70 | 0.000 | 0.0% | n/a | n/a | n/a | 12.05 | 15% |
| 32 | orthogonal-full-rank | adaptive | 100/100 | 2.0974 | 25168 | 62.18 | 9.89 | 3.270 | 100.0% | n/a | n/a | n/a | 119.02 | 0% |
| 32 | orthogonal-full-rank | asad | 100/100 | 1.2097 | 6240 | 90.00 | 0.00 | 0.000 | 0.0% | 27.0% | 31.69 | n/a | 196.80 | 0% |
| 32 | orthogonal-full-rank | bcad-boundary-delayed | 100/100 | 1.0545 | 6240 | 90.00 | 0.00 | 0.020 | 2.0% | 0.0% | 32.00 | 32.00 | 230.67 | 0% |
| 32 | orthogonal-full-rank | bcad-fixed-initial-rank | 100/100 | 1.0864 | 6240 | 90.00 | 0.00 | 0.040 | 4.0% | 1.0% | 31.99 | 5.57 | 227.01 | 0% |
| 32 | orthogonal-full-rank | bcad-full-dimensional | 100/100 | 1.0545 | 6240 | 90.00 | 0.00 | 0.020 | 2.0% | 0.0% | 32.00 | 32.00 | 229.87 | 0% |
| 32 | orthogonal-full-rank | bcad-gradient-jump | 100/100 | 1.0622 | 6240 | 90.00 | 0.00 | 0.020 | 2.0% | 1.0% | 31.99 | 32.00 | 226.02 | 0% |
| 32 | orthogonal-full-rank | bcad-prediction-control | 100/100 | 1.0972 | 6240 | 90.00 | 0.00 | 0.030 | 3.0% | 0.0% | 32.00 | 32.00 | 132.62 | 0% |
| 32 | orthogonal-full-rank | bcad-single-hypothesis-acquisition | 100/100 | 1.0607 | 6240 | 90.00 | 0.00 | 0.000 | 0.0% | 0.0% | 32.00 | 32.00 | 127.90 | 0% |
| 32 | orthogonal-full-rank | bcad-stable-adaptive-rank | 100/100 | 1.0559 | 6240 | 90.00 | 0.00 | 0.020 | 2.0% | 0.0% | 32.00 | 32.00 | 229.43 | 0% |
| 32 | orthogonal-full-rank | ransac | 0/100 | n/a | 14592 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 150.48 | 100% |
| 32 | orthogonal-full-rank | star-probe | 100/100 | 2.0974 | 25168 | 50.01 | 14.22 | 0.000 | 0.0% | n/a | n/a | n/a | 48.96 | 0% |
| 32 | small-amplitude-low-rank | adaptive | 0/100 | n/a | 15902 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 4.91 | 100% |
| 32 | small-amplitude-low-rank | asad | 100/100 | 3.1039e-10 | 3315 | 0.00 | 1.00 | 0.000 | 0.0% | 0.0% | 1.00 | n/a | 37.26 | 0% |
| 32 | small-amplitude-low-rank | bcad-boundary-delayed | 100/100 | 0.00034159 | 3557 | 6.30 | 0.93 | 0.000 | 0.0% | 0.0% | 1.00 | 3.17 | 47.73 | 0% |
| 32 | small-amplitude-low-rank | bcad-fixed-initial-rank | 100/100 | 0.000099750 | 3438 | 1.80 | 0.98 | 0.000 | 0.0% | 0.0% | 1.00 | 1.31 | 45.03 | 0% |
| 32 | small-amplitude-low-rank | bcad-full-dimensional | 100/100 | 0.00023600 | 3485 | 4.50 | 0.95 | 0.000 | 0.0% | 0.0% | 1.00 | 32.00 | 46.86 | 0% |
| 32 | small-amplitude-low-rank | bcad-gradient-jump | 100/100 | 0.00023599 | 3485 | 4.50 | 0.95 | 0.000 | 0.0% | 0.0% | 1.00 | 2.55 | 46.03 | 0% |
| 32 | small-amplitude-low-rank | bcad-prediction-control | 100/100 | 0.059168 | 3474 | 2.70 | 0.97 | 0.000 | 0.0% | 0.0% | 1.00 | 1.93 | 36.87 | 0% |
| 32 | small-amplitude-low-rank | bcad-single-hypothesis-acquisition | 100/100 | 0.00023599 | 3489 | 4.50 | 0.95 | 0.000 | 0.0% | 0.0% | 1.00 | 2.55 | 45.94 | 0% |
| 32 | small-amplitude-low-rank | bcad-stable-adaptive-rank | 100/100 | 0.00021841 | 3432 | 3.60 | 0.96 | 0.000 | 0.0% | 0.0% | 1.00 | 1.00 | 44.98 | 0% |
| 32 | small-amplitude-low-rank | ransac | 0/100 | n/a | 14592 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 4.79 | 100% |
| 32 | small-amplitude-low-rank | star-probe | 0/100 | n/a | 15902 | 90.00 | 0.00 | 0.000 | 0.0% | n/a | n/a | n/a | 4.92 | 100% |

Interpret component comparisons causally only within the listed one-factor arms: boundary-uncertainty versus prediction-control and gradient-jump isolate acquisition scoring; delayed-reduction versus full-dimensional isolates dimension reduction; delayed-reduction versus fixed-initial/stable-adaptive isolates rank policy; delayed-reduction versus single-hypothesis acquisition tests competing-boundary information. These comparisons do not tune thresholds on the independent test points.

## Completion and exception counts

| Method | Complete | Incomplete | Threw |
|---|---:|---:|---:|
| asad | 407 | 593 | 0 |
| ransac | 7 | 0 | 993 |
| star-probe | 528 | 200 | 272 |
| adaptive | 530 | 201 | 269 |
| bcad-boundary-delayed | 489 | 511 | 0 |
| bcad-prediction-control | 481 | 519 | 0 |
| bcad-gradient-jump | 496 | 504 | 0 |
| bcad-full-dimensional | 496 | 504 | 0 |
| bcad-fixed-initial-rank | 486 | 514 | 0 |
| bcad-stable-adaptive-rank | 468 | 532 | 0 |
| bcad-single-hypothesis-acquisition | 463 | 537 | 0 |
| **Total** | **4851** | **4615** | **1534** |

## Conclusion

BCAD is **not-supported** under the predefined criteria: 0 of 6 primary dimension/scenario cells qualify, and 1,534 of 11,000 solver attempts throw (all thrown attempts are baseline methods). Passing model-based regression tests does not establish a benchmark advantage. When the criteria fail, use the per-cell MAE, boundary counts, false-accept rates, rank errors, query counts, incomplete runs, and exceptions above to distinguish acquisition failure, rank-estimation failure, and unresolved boundary identifiability. No failing runs are replaced by synthetic scores.

## Evidence and limits

- Raw per-run records: [bcad-2026-10-29T14-46-37-077Z-raw.jsonl](../../scratch/results/bcad/bcad-2026-10-29T14-46-37-077Z-raw.jsonl)
- The JSON summary next to the raw file contains per-cell aggregates, ablation arms, per-cell acceptance assessments, and the verdict inputs.
- This is a deterministic synthetic scalar-output experiment; it does not establish performance on noisy real APIs, deep neural networks, or language models.
- Training-fit residual is evaluated on acquired observations and is not an independent holdout; benchmark MAE uses a separate evaluation set.
