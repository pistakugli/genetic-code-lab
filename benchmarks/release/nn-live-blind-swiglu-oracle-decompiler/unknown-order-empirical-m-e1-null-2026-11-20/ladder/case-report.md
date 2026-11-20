# Unknown-order SwiGLU recovery ladder report

- synthetic teacher: False
- pool: 15998 points, dimension 4096, splits {'train': 9598, 'selection': 3200, 'validation': 3200}
- ladder: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24] (early-stop threshold r_sel <= 0.05)

| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |
|---|---|---|---|---|---|---|
| 4 | 0.07940 | 0.07522 | 0.12257 | False | False | 1145204 |
| 5 | 0.06960 | 0.06615 | 0.11763 | False | False | 1147552 |
| 6 | 0.07372 | 0.06973 | 0.12297 | False | False | 1150790 |
| 8 | 0.06021 | 0.05853 | 0.11740 | False | False | 1157616 |
| 10 | 0.01291 | 0.01307 | 0.02952 | True | False | 1166702 |
| 12 | 0.02116 | 0.02181 | 0.05995 | True | False | 1176564 |
| 14 | 0.01896 | 0.01855 | 0.06182 | True | False | 1185784 |

- inferred order: 10
- verdict: inferred-order
- upper-region non-identifiable: True

## Null permutation control

- candidates: [4, 5, 6, 8, 10]
- verdict: no-sufficient-order
