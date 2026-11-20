# Unknown-order SwiGLU recovery ladder report

- synthetic teacher: False
- pool: 21524 points, dimension 3072, splits {'train': 12914, 'selection': 4305, 'validation': 4305}
- ladder: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24] (early-stop threshold r_sel <= 0.05)

| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |
|---|---|---|---|---|---|---|
| 4 | 0.11955 | 0.09499 | 0.29991 | False | False | 882253 |
| 5 | 0.07428 | 0.05885 | 0.15096 | False | False | 885339 |
| 6 | 0.05951 | 0.04760 | 0.12537 | False | False | 888725 |
| 8 | 0.06568 | 0.05299 | 0.27186 | False | False | 895847 |
| 10 | 0.08936 | 0.06759 | 0.46598 | False | False | 905873 |
| 12 | 0.03556 | 0.02738 | 0.21094 | True | False | 914127 |
| 14 | 0.02917 | 0.02241 | 0.28690 | True | False | 926935 |
| 16 | 0.02498 | 0.01850 | 0.30189 | True | False | 936977 |

- inferred order: 12
- verdict: inferred-order
- upper-region non-identifiable: True
