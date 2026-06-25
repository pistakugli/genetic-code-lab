# NN Live Blind SwiGLU Finite-Precision Ladder

Status: **PASS**

This report reads existing real-model oracle-only Mistral SwiGLU decompiler artifacts and centralizes finite-precision geometry degradation. It does not retune the solver.

| case | type | q | pass | R2 | MAE | worst | subspace cosine | raw calls | delta vs clean selected |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| clean selected | selected | 0 | true | 0.995341399534 | 0.001122070889 | 0.012468272718 | 0.99999661638 | 85649 | baseline |
| q=1e-8 selected | selected | 1.000000e-8 | true | 0.995450402376 | 0.001110461082 | 0.012667784693 | 0.99997244253 | 85649 | R2 0.000109002842, MAE -1.160981e-5, worst 0.000199511975, subspace -2.417385e-5 |
| q=1e-8 randomized | randomized | 1.000000e-8 | true | 0.995759462805 | 0.001008800502 | 0.009621486909 | 0.999982859729 | 171298 | not comparable to selected baseline |
| q=1e-7 selected | selected | 1.000000e-7 | true | 0.996334316414 | 0.00101318115 | 0.008584273469 | 0.999993062181 | 118425 | R2 0.000992916879, MAE -0.000108889739, worst -0.003883999248, subspace -3.554199e-6 |
| q=1e-7 randomized | randomized | 1.000000e-7 | true | 0.995860081805 | 0.001003628799 | 0.010330999089 | 0.99998577104 | 236850 | not comparable to selected baseline |
| q=1e-6 selected | selected | 1.000000e-6 | true | 0.995241421111 | 0.00116365928 | 0.009295537729 | 0.999353644734 | 220850 | R2 -9.997842e-5, MAE 4.158839e-5, worst -0.003172734989, subspace -0.000642971646 |
| q=1e-6 randomized | randomized | 1.000000e-6 | true | 0.995867637063 | 0.000985653719 | 0.014497109405 | 0.999440768464 | 581508 | not comparable to selected baseline |

## Summary

- rows: 7
- min R2: 0.995241421111
- max MAE: 0.00116365928
- max worst error: 0.014497109405
- min subspace principal cosine: 0.999353644734
- max total raw oracle calls: 581508
- access invariant: true

## Boundaries

- This is an artifact-backed finite-precision geometry audit, not a new solver tuning result.
- 4096D is the Mistral hidden/input dimension observed through scalar oracle calls.
- 8-unit is the recovered selected/randomized SwiGLU block size; this does not prove full-FFN or all-channel extraction.
- The solver access invariant remains oracle-only for the source reports; weights/hooks are post-hoc witnesses only.
- Randomized rows are deterministic two-case sweeps and are not directly delta-comparable to the selected clean baseline.

