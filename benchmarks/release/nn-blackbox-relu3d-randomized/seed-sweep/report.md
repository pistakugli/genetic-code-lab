# NN Randomized 3D Black-Box ReLU Decompile Benchmark

- Teachers: randomized 3-input/4-hidden non-vertical ReLU MLPs
- GCL access: oracle calls only, `f(x,y,z)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Seeds: 101, 2026, 4242, 7777, 31415
- Z slices: 7
- Y slices per z-slice: 21
- Samples per 2D slice line: 801
- Test rows per seed: 2717

| Seed | Pass | Terms | MAE | R2 | Max geom error |
|---:|---|---:|---:|---:|---:|
| 101 | yes | 4 | 6.617846e-16 | 1.000000000000 | 2.775558e-15 |
| 2026 | yes | 4 | 3.933787e-16 | 1.000000000000 | 2.997602e-15 |
| 4242 | yes | 4 | 5.027065e-16 | 1.000000000000 | 2.262079e-15 |
| 7777 | yes | 4 | 6.071066e-16 | 1.000000000000 | 3.108624e-15 |
| 31415 | yes | 4 | 6.641245e-16 | 1.000000000000 | 1.554312e-15 |

## Gate

- Pass: yes
- Passed seeds: 5/5
- Max MAE: 6.641245e-16
- Min R2: 1.000000000000
- Max delta-slope error: 1.887379e-15
- Max alphaY error: 2.262079e-15
- Max alphaZ error: 2.664535e-15
- Max beta error: 3.108624e-15
