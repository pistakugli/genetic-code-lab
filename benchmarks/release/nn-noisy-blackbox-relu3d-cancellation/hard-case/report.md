# NN Noisy 3D Black-Box ReLU Cancellation Hard-Case Benchmark

- Teachers: 3-input/3-hidden ReLU MLPs with close opposite-sign kink pairs
- GCL access: repeated noisy oracle calls, `f(x,y,z)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Fix under test: slice-only stitching plus global parametric ReLU surface refinement

## Claim Scope

- Proves: output-equivalent transparent symbolic 3D ReLU surface recovery on noisy near-cancellation cases.
- Does not claim: unique recovery of the original hidden-unit parameterization for cancelling near-collision pairs.

| Teacher | Pass | Slice MAE | Refined MAE | Refined R2 | Refined/Slice MAE |
|---|---|---:|---:|---:|---:|
| close-opposite-kink-pair-a | yes | 8.172974e-2 | 2.460570e-3 | 0.999961870967 | 3.010617e-2 |
| close-opposite-kink-pair-b | yes | 9.238225e-2 | 1.873035e-3 | 0.999855666706 | 2.027483e-2 |

## Gate

- Pass: yes
- Passed teachers: 2/2
- Max refined MAE: 2.460570e-3
- Min refined R2: 0.999855666706
- Worst refined/slice MAE ratio: 3.010617e-2
