# Global Function Distillation with Variable Projection (GFD-VP) - Pilot

**Run:** 2026-10-29T20-20-20-882Z  
**Verdict:** not-supported  
**Seeds per cell:** 10  
**Dimensions:** 16, 32  
**Scenarios:** near-parallel-low-rank, multi-crossing-low-rank, small-amplitude-low-rank, contaminated-local-probes, orthogonal-full-rank  
**Methods:** asad, ransac, star-probe, adaptive, bcad, mbcd-full, gfd-profiled, gfd-single-start, gfd-joint-fit  
**Raw runs:** [gfd-2026-10-29T20-20-20-882Z-raw.jsonl](../../scratch/results/gfd/pilot/gfd-2026-10-29T20-20-20-882Z-raw.jsonl)

## Pilot scope

This is a ten-seed screening experiment, not the predeclared 100-seed confirmation. Its purpose is to reject an unpromising approach early. A “not-supported” pilot verdict is evidence against proceeding unchanged, not a formal 100-seed estimate.


## Hypothesis and algorithm

GFD-VP fits the complete scalar ReLU sum directly to uniformly sampled oracle values. It does not estimate local oracle gradients, active rank, line kinks, or candidate hyperplanes before fitting. At each optimization checkpoint, it profiles out the affine/readout layer by solving a regularized least-squares system for the current hidden ReLU features; only hidden normals and offsets are updated by gradients of the value-fit objective. Hidden normals are gauge-normalized after every update. Independent held-out oracle samples select among deterministic restarts. Solver inputs contain only the oracle, dimension, ranges, term count, seed, and budget.

The method is non-convex and depends on the known term count. A successful value fit does not imply geometrically identified units because ReLU decompositions can be non-identifiable or nearly cancelling; all units and matched geometric errors are still reported.

## Paired design and unchanged acceptance criteria

Every method receives the same deterministic teacher, model seed, oracle-call ceiling, and independently generated 192-point test set for each seed/dimension/scenario. Existing methods include ASAD, RANSAC, Star-Probe, Adaptive, BCAD, and MBCD. The three primary classes are near-parallel facets, multiple crossings, and small-amplitude jumps at 16D and 32D; contaminated probes and orthogonal full-rank teachers are controls. This run uses 10 seeds per cell; the predeclared full campaign uses 100. All method predictions are scored on every seed; exceptions remain unscored and are not dropped from run counts. GFD-VP fits are dispatched through one persistent CUDA worker; samples are staged in pinned host memory and transferred to CUDA tensors. Each run records the CUDA device, tensor device, peak allocated VRAM, and GPU fit time separately from end-to-end solver runtime.

The previously fixed cell criteria are unchanged: at least 25% lower MAE than the best scored existing method, or at least 30% fewer oracle calls than a baseline within 5% of the best baseline MAE while GFD MAE is within 5% of that best MAE; false-accepted-run rate may rise by at most one percentage point. At least two of three primary classes must qualify at both dimensions, with no budget overrun. The verdict is not inferred from tests.

## Primary comparison

| D | Scenario | Best existing method | GFD-VP MAE | Baseline MAE | Call reduction | MAE reduction | GFD complete rate | False-accepted increase | Qualifies | Reason |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 16 | near-parallel-low-rank | star-probe | 0.01031 | 1.2652e-12 | 59.31% | n/a | 0.00% | 50.00% | no | neither_predeclared_performance_threshold_met |
| 16 | multi-crossing-low-rank | star-probe | 0.00057938 | 1.3391e-12 | 60.38% | n/a | 0.00% | 0.00% | no | neither_predeclared_performance_threshold_met |
| 16 | small-amplitude-low-rank | asad | 8.0073e-9 | 1.7708e-10 | -359.71% | n/a | 100.00% | 0.00% | no | neither_predeclared_performance_threshold_met |
| 32 | near-parallel-low-rank | star-probe | 0.021048 | 1.0599e-12 | 76.54% | n/a | 0.00% | 70.00% | no | neither_predeclared_performance_threshold_met |
| 32 | multi-crossing-low-rank | star-probe | 0.00036205 | 1.7438e-12 | 77.07% | n/a | 0.00% | 0.00% | no | neither_predeclared_performance_threshold_met |
| 32 | small-amplitude-low-rank | asad | 9.2595e-9 | 5.2606e-10 | -23.56% | n/a | 100.00% | 0.00% | no | neither_predeclared_performance_threshold_met |

**Primary scenario classes qualifying at both dimensions:** 0/3 (none).  
**Actual oracle-budget overruns:** 0.  
**Run statuses:** 255 complete, 491 incomplete, 154 threw.

## All-seed method metrics

| D | Scenario | Method | Scored/runs | MAE | Oracle calls | Complete rate | Normal error (deg) | Offset error | False-accepted run rate | Runtime (ms) | CUDA VRAM (MiB) | GPU fit (ms) | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | near-parallel-low-rank | asad | 10/10 | 1.1747 | 2950.2 | 10.00% | 63.055 | 0.50956 | 0.00% | 59.816 | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | ransac | 0/10 | n/a | 8448 | 0.00% | n/a | n/a | n/a | 6.1663 | n/a | n/a | 10 |
| 16 | near-parallel-low-rank | star-probe | 10/10 | 1.2652e-12 | 10066 | 100.00% | 1.4575e-7 | 5.6809e-13 | 0.00% | 7.8016 | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | adaptive | 10/10 | 1.2652e-12 | 10726 | 100.00% | 1.4575e-7 | 5.6809e-13 | 0.00% | 6.6879 | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | bcad | 10/10 | 0.027151 | 2356.2 | 50.00% | 22.521 | 0.0023404 | 0.00% | 40.871 | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | mbcd-full | 10/10 | 0.14779 | 2640 | 0.00% | 43.043 | 0.12533 | 80.00% | 27.867 | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | gfd-profiled | 10/10 | 0.01031 | 4096 | 0.00% | 32.261 | 0.23217 | 50.00% | 232.67 | 20.64 | 208.17 | 0 |
| 16 | near-parallel-low-rank | gfd-single-start | 10/10 | 0.064219 | 4096 | 0.00% | 67.691 | 0.6257 | 90.00% | 154.71 | 17.424 | 130.63 | 0 |
| 16 | near-parallel-low-rank | gfd-joint-fit | 10/10 | 0.018197 | 4096 | 0.00% | 36.759 | 0.42567 | 40.00% | 143.38 | 19.229 | 119.65 | 0 |
| 16 | multi-crossing-low-rank | asad | 10/10 | 1.0062 | 3168 | 0.00% | 84 | 0.8 | 0.00% | 64.996 | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | ransac | 0/10 | n/a | 8448 | 0.00% | n/a | n/a | n/a | 7.0477 | n/a | n/a | 10 |
| 16 | multi-crossing-low-rank | star-probe | 10/10 | 1.3391e-12 | 10338 | 100.00% | 6.8706e-8 | 8.5558e-13 | 0.00% | 8.3709 | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | adaptive | 10/10 | 1.3391e-12 | 11321 | 100.00% | 6.8706e-8 | 8.5558e-13 | 0.00% | 9.7331 | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | bcad | 10/10 | 0.28771 | 3168 | 0.00% | 87 | 0.9 | 20.00% | 62.969 | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | mbcd-full | 10/10 | 0.1385 | 2640 | 0.00% | 36 | 7.3169e-14 | 80.00% | 38.37 | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | gfd-profiled | 10/10 | 0.00057938 | 4096 | 0.00% | 0.11063 | 0.00058568 | 0.00% | 156.91 | 20.843 | 132.82 | 0 |
| 16 | multi-crossing-low-rank | gfd-single-start | 10/10 | 0.056552 | 4096 | 0.00% | 22.408 | 0.10509 | 40.00% | 152.08 | 17.5 | 128.51 | 0 |
| 16 | multi-crossing-low-rank | gfd-joint-fit | 10/10 | 0.024063 | 4096 | 0.00% | 6.8997 | 0.0070215 | 20.00% | 142.95 | 19.532 | 119.3 | 0 |
| 16 | small-amplitude-low-rank | asad | 10/10 | 1.7708e-10 | 891 | 100.00% | 2.0612e-7 | 4.0982e-12 | 0.00% | 8.8115 | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | ransac | 0/10 | n/a | 8448 | 0.00% | n/a | n/a | n/a | 3.6002 | n/a | n/a | 10 |
| 16 | small-amplitude-low-rank | star-probe | 0/10 | n/a | 9094.8 | 0.00% | n/a | n/a | n/a | 3.6605 | n/a | n/a | 10 |
| 16 | small-amplitude-low-rank | adaptive | 0/10 | n/a | 9094.8 | 0.00% | n/a | n/a | n/a | 3.6722 | n/a | n/a | 10 |
| 16 | small-amplitude-low-rank | bcad | 10/10 | 0.00095641 | 1749 | 80.00% | 18 | 0.2 | 0.00% | 24.001 | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | mbcd-full | 10/10 | 0.0016368 | 2640 | 0.00% | 22.314 | 0.22692 | 20.00% | 20.231 | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | gfd-profiled | 10/10 | 8.0073e-9 | 4096 | 100.00% | 0.000074881 | 3.0146e-7 | 0.00% | 158.79 | 20.586 | 134.77 | 0 |
| 16 | small-amplitude-low-rank | gfd-single-start | 10/10 | 0.0013726 | 4096 | 60.00% | 36 | 0.4 | 40.00% | 154.34 | 17.35 | 130.38 | 0 |
| 16 | small-amplitude-low-rank | gfd-joint-fit | 10/10 | 0.0035461 | 4096 | 0.00% | 90 | 1 | 100.00% | 145.35 | 18.927 | 121.59 | 0 |
| 16 | contaminated-local-probes | asad | 10/10 | 0.000065847 | 891 | 100.00% | 0.004058 | 0.0000076072 | 0.00% | 8.9025 | n/a | n/a | 0 |
| 16 | contaminated-local-probes | ransac | 0/10 | n/a | 8448 | 0.00% | n/a | n/a | n/a | 4.0701 | n/a | n/a | 10 |
| 16 | contaminated-local-probes | star-probe | 10/10 | 2.9497e-12 | 9874 | 100.00% | 0 | 2.4759e-13 | 0.00% | 5.6366 | n/a | n/a | 0 |
| 16 | contaminated-local-probes | adaptive | 10/10 | 2.9497e-12 | 10204 | 100.00% | 0 | 2.4759e-13 | 0.00% | 5.7201 | n/a | n/a | 0 |
| 16 | contaminated-local-probes | bcad | 10/10 | 0.015863 | 1188 | 90.00% | 9.0067 | 0.10001 | 0.00% | 16.071 | n/a | n/a | 0 |
| 16 | contaminated-local-probes | mbcd-full | 10/10 | 0.069164 | 2640 | 0.00% | 37.304 | 0.41675 | 40.00% | 21.05 | n/a | n/a | 0 |
| 16 | contaminated-local-probes | gfd-profiled | 10/10 | 0.010248 | 4096 | 90.00% | 9 | 0.1 | 10.00% | 155.27 | 20.586 | 130.27 | 0 |
| 16 | contaminated-local-probes | gfd-single-start | 10/10 | 0.074272 | 4096 | 30.00% | 63 | 0.7 | 70.00% | 151.6 | 17.35 | 126.42 | 0 |
| 16 | contaminated-local-probes | gfd-joint-fit | 10/10 | 0.00017464 | 4096 | 0.00% | 0.025084 | 0.00027321 | 0.00% | 146.52 | 18.927 | 115.63 | 0 |
| 16 | orthogonal-full-rank | asad | 10/10 | 0.7381 | 3168 | 0.00% | 90 | 1 | 0.00% | 73.497 | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | ransac | 0/10 | n/a | 8448 | 0.00% | n/a | n/a | n/a | 29.059 | n/a | n/a | 10 |
| 16 | orthogonal-full-rank | star-probe | 10/10 | 1.3198 | 12848 | 0.00% | 5.625 | 3.9466e-12 | 0.00% | 29.227 | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | adaptive | 10/10 | 1.3198 | 12848 | 0.00% | 35.438 | 3.5497e-12 | 100.00% | 33.188 | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | bcad | 10/10 | 0.66499 | 3168 | 0.00% | 90 | 1 | 10.00% | 116.99 | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | mbcd-full | 10/10 | 0.7406 | 2640 | 0.00% | 90 | 1 | 100.00% | 58.01 | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | gfd-profiled | 10/10 | 0.20615 | 4096 | 0.00% | 78.678 | 0.31791 | 100.00% | 175.52 | 24.005 | 141.59 | 0 |
| 16 | orthogonal-full-rank | gfd-single-start | 10/10 | 0.21692 | 4096 | 0.00% | 81.225 | 0.41109 | 100.00% | 165.78 | 18.48 | 131.96 | 0 |
| 16 | orthogonal-full-rank | gfd-joint-fit | 10/10 | 0.23273 | 4096 | 0.00% | 73.589 | 0.024906 | 100.00% | 155.85 | 24.008 | 121.71 | 0 |
| 32 | near-parallel-low-rank | asad | 10/10 | 0.91873 | 6006 | 10.00% | 67.54 | 0.50878 | 10.00% | 102.89 | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | ransac | 0/10 | n/a | 14592 | 0.00% | n/a | n/a | n/a | 13.334 | n/a | n/a | 10 |
| 32 | near-parallel-low-rank | star-probe | 8/10 | 1.0599e-12 | 17457 | 80.00% | 2.668e-7 | 7.6079e-13 | 0.00% | 21.331 | n/a | n/a | 2 |
| 32 | near-parallel-low-rank | adaptive | 8/10 | 1.0599e-12 | 18627 | 80.00% | 2.668e-7 | 7.6079e-13 | 0.00% | 22.294 | n/a | n/a | 2 |
| 32 | near-parallel-low-rank | bcad | 10/10 | 0.034677 | 5213 | 50.00% | 27 | 0.1 | 0.00% | 73.197 | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | mbcd-full | 10/10 | 0.094271 | 5200 | 0.00% | 50.82 | 0.32735 | 80.00% | 44.408 | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | gfd-profiled | 10/10 | 0.021048 | 4096 | 0.00% | 50.781 | 0.43969 | 70.00% | 194.25 | 24.653 | 144.76 | 0 |
| 32 | near-parallel-low-rank | gfd-single-start | 10/10 | 0.023403 | 4096 | 0.00% | 67.896 | 0.54134 | 100.00% | 182.79 | 18.356 | 132.95 | 0 |
| 32 | near-parallel-low-rank | gfd-joint-fit | 10/10 | 0.022324 | 4096 | 0.00% | 49.772 | 0.51675 | 60.00% | 170.81 | 21.617 | 121.67 | 0 |
| 32 | multi-crossing-low-rank | asad | 10/10 | 1.9177 | 6240 | 0.00% | 90 | 1 | 0.00% | 119.86 | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | ransac | 0/10 | n/a | 14592 | 0.00% | n/a | n/a | n/a | 18.385 | n/a | n/a | 10 |
| 32 | multi-crossing-low-rank | star-probe | 5/10 | 1.7438e-12 | 17864 | 50.00% | 5.6918e-8 | 1.284e-12 | 0.00% | 24.392 | n/a | n/a | 5 |
| 32 | multi-crossing-low-rank | adaptive | 5/10 | 1.7438e-12 | 19294 | 50.00% | 5.6918e-8 | 1.284e-12 | 0.00% | 26.026 | n/a | n/a | 5 |
| 32 | multi-crossing-low-rank | bcad | 10/10 | 0.30228 | 6240 | 0.00% | 87 | 0.9 | 0.00% | 106.89 | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | mbcd-full | 10/10 | 0.10662 | 5200 | 0.00% | 27 | 2.2938e-13 | 60.00% | 69.885 | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | gfd-profiled | 10/10 | 0.00036205 | 4096 | 0.00% | 0.049582 | 0.00011335 | 0.00% | 192.03 | 24.759 | 141.87 | 0 |
| 32 | multi-crossing-low-rank | gfd-single-start | 10/10 | 0.11464 | 4096 | 0.00% | 46.543 | 0.20852 | 70.00% | 182.01 | 18.433 | 131.52 | 0 |
| 32 | multi-crossing-low-rank | gfd-joint-fit | 10/10 | 0.035714 | 4096 | 0.00% | 10.433 | 0.011567 | 30.00% | 171.39 | 21.821 | 120.44 | 0 |
| 32 | small-amplitude-low-rank | asad | 10/10 | 5.2606e-10 | 3315 | 100.00% | 2.0612e-7 | 3.0566e-12 | 0.00% | 39.269 | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | ransac | 0/10 | n/a | 14592 | 0.00% | n/a | n/a | n/a | 10.008 | n/a | n/a | 10 |
| 32 | small-amplitude-low-rank | star-probe | 0/10 | n/a | 15840 | 0.00% | n/a | n/a | n/a | 10.706 | n/a | n/a | 10 |
| 32 | small-amplitude-low-rank | adaptive | 0/10 | n/a | 15840 | 0.00% | n/a | n/a | n/a | 10.739 | n/a | n/a | 10 |
| 32 | small-amplitude-low-rank | bcad | 10/10 | 0.0005487 | 3607.5 | 90.00% | 9 | 0.1 | 0.00% | 53.559 | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | mbcd-full | 10/10 | 0.00058979 | 5200 | 0.00% | 9.961 | 0.11142 | 10.00% | 34.324 | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | gfd-profiled | 10/10 | 9.2595e-9 | 4096 | 100.00% | 0.00008837 | 1.7808e-7 | 0.00% | 190.86 | 24.547 | 142.66 | 0 |
| 32 | small-amplitude-low-rank | gfd-single-start | 10/10 | 0.0006844 | 4096 | 80.00% | 18 | 0.2 | 20.00% | 182.21 | 18.281 | 133.68 | 0 |
| 32 | small-amplitude-low-rank | gfd-joint-fit | 10/10 | 0.0034076 | 4096 | 0.00% | 90 | 1 | 100.00% | 172.3 | 21.413 | 123.41 | 0 |
| 32 | contaminated-local-probes | asad | 10/10 | 8.831e-10 | 3315 | 100.00% | 2.915e-7 | 2.6652e-11 | 0.00% | 39.285 | n/a | n/a | 0 |
| 32 | contaminated-local-probes | ransac | 0/10 | n/a | 14592 | 0.00% | n/a | n/a | n/a | 10.578 | n/a | n/a | 10 |
| 32 | contaminated-local-probes | star-probe | 10/10 | 3.4911e-13 | 17362 | 100.00% | 3.8113e-7 | 8.743e-13 | 0.00% | 18.722 | n/a | n/a | 0 |
| 32 | contaminated-local-probes | adaptive | 10/10 | 3.4911e-13 | 18012 | 100.00% | 3.8113e-7 | 8.743e-13 | 0.00% | 19.503 | n/a | n/a | 0 |
| 32 | contaminated-local-probes | bcad | 10/10 | 0.014927 | 3692 | 90.00% | 9 | 0.1 | 0.00% | 54.487 | n/a | n/a | 0 |
| 32 | contaminated-local-probes | mbcd-full | 10/10 | 0.074215 | 5200 | 0.00% | 29.021 | 0.31801 | 30.00% | 35.374 | n/a | n/a | 0 |
| 32 | contaminated-local-probes | gfd-profiled | 10/10 | 0.011062 | 4096 | 90.00% | 9.0001 | 0.1 | 10.00% | 196.48 | 24.547 | 142.28 | 0 |
| 32 | contaminated-local-probes | gfd-single-start | 10/10 | 0.021862 | 4096 | 70.00% | 18.001 | 0.2 | 20.00% | 179.71 | 18.281 | 131.47 | 0 |
| 32 | contaminated-local-probes | gfd-joint-fit | 10/10 | 0.00013294 | 4096 | 0.00% | 0.025066 | 0.000052657 | 0.00% | 170.08 | 21.413 | 121.51 | 0 |
| 32 | orthogonal-full-rank | asad | 10/10 | 1.3069 | 6240 | 0.00% | 90 | 1 | 0.00% | 240.19 | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | ransac | 0/10 | n/a | 14592 | 0.00% | n/a | n/a | n/a | 244.29 | n/a | n/a | 10 |
| 32 | orthogonal-full-rank | star-probe | 10/10 | 1.996 | 25168 | 0.00% | 51.188 | 9.8111e-12 | 0.00% | 209.29 | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | adaptive | 10/10 | 1.996 | 25168 | 0.00% | 63.281 | 1.0099e-11 | 100.00% | 279.72 | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | bcad | 10/10 | 1.1058 | 6240 | 0.00% | 90 | 1 | 10.00% | 271.53 | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | mbcd-full | 10/10 | 2.2333 | 5200 | 0.00% | 90 | 1 | 100.00% | 144.78 | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | gfd-profiled | 10/10 | 0.50887 | 4096 | 0.00% | 90 | 1 | 100.00% | 244.77 | 31.036 | 163.55 | 0 |
| 32 | orthogonal-full-rank | gfd-single-start | 10/10 | 0.52378 | 4096 | 0.00% | 90 | 1 | 100.00% | 220.6 | 20.636 | 138.59 | 0 |
| 32 | orthogonal-full-rank | gfd-joint-fit | 10/10 | 0.54042 | 4096 | 0.00% | 90 | 1 | 100.00% | 204.23 | 31.041 | 122.62 | 0 |

## Ablations and limitations

GFD-single-start removes restart selection; GFD-joint-fit removes variable projection and trains all readout parameters jointly. Compare their matched MAE, geometry, completion, oracle use, and runtime against GFD-VP; ablations are explanatory and cannot replace the predeclared primary verdict. Synthetic scalar-output results do not establish effectiveness on deep networks or real noisy APIs. The verifier independently checks paired teacher/test fingerprints, run completeness, actual oracle counts, diagnostics, budget ceilings, and verdict calculations from raw records.

## Pilot interpretation and next experiment

The pilot does not meet the predeclared success criteria in any primary cell. The global fit is exact enough to complete the small-amplitude cases, but its MAE and oracle use do not beat ASAD there. It does not complete the near-parallel or multi-crossing reconstructions; multi-crossing normal estimates can be close while the function error and completion criterion still fail. The contaminated and full-rank controls also do not show reliable completion. This rejects the current GFD-VP configuration as a candidate for an unchanged 100-seed campaign; it is not evidence that global value fitting can never work.

The next distinct hypothesis is **global spectral-moment inversion**: use the distributional Hessian structure of a ReLU sum, whose Fourier transform encodes plane orientations and offsets, instead of optimizing latent hidden units against sampled values. This is unverified and should first be tested on a small, fixed synthetic pilot with the same oracle-only access and exact query accounting. It must demonstrate geometric identification on near-parallel and full-rank controls before any larger campaign.

