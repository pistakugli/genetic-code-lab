# NN Randomized Noisy 3D Black-Box ReLU Decompile Benchmark

- Teachers: randomized 3-input/3-hidden non-vertical ReLU MLPs
- Teacher profile: non-degenerate randomized, separated beta templates
- GCL access: repeated noisy oracle calls, `f(x,y,z)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Seeds: 101, 2026, 4242, 7777, 31415
- Noise std: 0.005
- Quantization step: 0.002
- Query samples per x,y,z: 21
- Z slices: 7
- Y slices per z-slice: 11
- Samples per horizontal line: 301
- Test rows per seed: 4095

| Seed | Pass | Terms | MAE | R2 | Max geom error |
|---:|---|---:|---:|---:|---:|
| 101 | yes | 3 | 1.084563e-3 | 0.999995004694 | 2.214624e-3 |
| 2026 | yes | 3 | 1.670918e-4 | 0.999999409872 | 3.141031e-4 |
| 4242 | yes | 3 | 1.217133e-3 | 0.999996118528 | 2.155225e-3 |
| 7777 | yes | 3 | 1.239950e-3 | 0.999996416595 | 4.160978e-3 |
| 31415 | yes | 3 | 1.890366e-3 | 0.999945552801 | 5.575704e-3 |

## Gate

- Pass: yes
- Passed seeds: 5/5
- Max MAE: 1.890366e-3
- Min R2: 0.999945552801
- Max delta-slope error: 5.575704e-3
- Max alphaY error: 2.427880e-3
- Max alphaZ error: 1.269603e-3
- Max beta error: 1.301518e-3
