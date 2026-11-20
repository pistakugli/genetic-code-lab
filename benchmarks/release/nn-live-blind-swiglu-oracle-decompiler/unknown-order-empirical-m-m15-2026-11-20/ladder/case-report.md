# Unknown-order SwiGLU recovery ladder report

- synthetic teacher: False
- pool: 15998 points, dimension 4096, splits {'train': 9598, 'selection': 3200, 'validation': 3200}
- ladder: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24] (early-stop threshold r_sel <= 0.05)

| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |
|---|---|---|---|---|---|---|
| 4 | 0.33642 | 0.32223 | 0.37814 | False | False | 1145294 |
| 5 | 0.29403 | 0.28620 | 0.41230 | False | False | 1147420 |
| 6 | 0.24939 | 0.24573 | 0.67421 | False | False | 1150634 |
| 8 | 0.13299 | 0.13243 | 0.69486 | False | False | 1158466 |
| 10 | 0.07375 | 0.07381 | 0.17533 | False | False | 1166786 |
| 12 | 0.09814 | 0.09880 | 0.25316 | False | False | 1177214 |
| 14 | 0.06130 | 0.06344 | 0.16402 | False | False | 1183196 |
| 16 | 0.06520 | 0.06797 | 0.19544 | False | False | 1189626 |
| 18 | 0.04721 | 0.04723 | 0.14192 | True | False | 1205902 |

- inferred order: 18
- verdict: inferred-order
- upper-region non-identifiable: False
