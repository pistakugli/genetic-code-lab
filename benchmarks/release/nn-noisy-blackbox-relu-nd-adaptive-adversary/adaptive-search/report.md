# Adaptive Noisy 4D/5D Adversarial ND ReLU Line-Probe Sweep

Generated: 2026-06-02T09:06:02.238Z

Pass: true

## Summary

- Teachers: 4/4
- Candidate pool size: 24
- Selected hard teachers: 4
- Profiles: mixed, mixed, near-parallel, near-parallel
- Dimensions: 4, 5
- Max MAE: 0.0016959558489471093
- Min R2: 0.9998286294723496
- Max plane error: 0.008612701190351582
- Total raw oracle calls: 3708980

## Results

| Teacher | Profile | Dim | Difficulty | Reasons | Pass | MAE | R2 | Linear MAE | Max plane error | Raw oracle calls | Error |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 4d-adaptive-mixed-270148 | mixed | 4 | 50.139 | near-parallel normals, offset collision, weak expected line kink, opposite-sign cancellation risk | yes | 4.413244e-4 | 0.999998442954 | 1.744173e-1 | 3.474133e-3 | 696300 |  |
| 5d-adaptive-mixed-270518 | mixed | 5 | 42.572 | near-parallel normals, offset collision, weak expected line kink, opposite-sign cancellation risk | yes | 7.245601e-4 | 0.999994349318 | 2.035488e-1 | 3.404301e-3 | 1275230 |  |
| 4d-adaptive-near-parallel-270000 | near-parallel | 4 | 40.569 | near-parallel normals, opposite-sign cancellation risk | yes | 1.364063e-3 | 0.999984178791 | 1.020909e-1 | 8.189794e-3 | 616770 |  |
| 5d-adaptive-near-parallel-270740 | near-parallel | 5 | 40.481 | near-parallel normals, weak expected line kink, opposite-sign cancellation risk | yes | 1.695956e-3 | 0.999828629472 | 4.675764e-2 | 8.612701e-3 | 1120680 |  |
