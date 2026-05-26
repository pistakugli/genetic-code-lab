# NN Noisy 3D Black-Box ReLU Adversarial Sweep

- Teachers: 3-input/3-hidden ReLU MLPs with adversarial shallow-3D geometry
- Profiles: near-parallel, beta-collision, cancellation, oblique-wide, low-amplitude-mixed
- GCL access: repeated noisy oracle calls, `f(x,y,z)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Query budget is counted from actual oracle calls inside slice stitching and global refinement.

## Claim Scope

- Proves: output-equivalent transparent symbolic 3D ReLU surface recovery across this adversarial shallow-3D sweep.
- Does not claim: general-ND recovery, deep ReLU composition recovery, or superiority over external piecewise-linear extraction methods.

| Teacher | Profile | Pass | Slice MAE | Refined MAE | Refined R2 | Oracle calls |
|---|---|---|---:|---:|---:|---:|
| near-parallel-separated-offsets | near-parallel | yes | 1.663336e-4 | 2.039524e-4 | 0.999999930132 | 399742 |
| same-offset-crossing-hyperplanes | beta-collision | yes | 5.562449e-3 | 2.623726e-4 | 0.999998395088 | 399742 |
| close-opposite-kink-pair-a | cancellation | yes | 8.106758e-2 | 2.762764e-3 | 0.999950989613 | 399742 |
| close-opposite-kink-pair-b | cancellation | yes | 9.237453e-2 | 2.348975e-3 | 0.999866416184 | 399742 |
| oblique-wide-orientation-spread | oblique-wide | yes | 4.103899e-3 | 3.606806e-4 | 0.999999685104 | 399742 |
| low-amplitude-opposite-mixed | low-amplitude-mixed | yes | 3.721686e-2 | 6.098730e-4 | 0.999994909131 | 399742 |

## Gate

- Pass: yes
- Passed teachers: 6/6
- Max refined MAE: 2.762764e-3
- Min refined R2: 0.999866416184
- Worst refined/slice MAE ratio: 1.226164e+0
- Total oracle calls: 2398452
- Max oracle calls per teacher: 399742
