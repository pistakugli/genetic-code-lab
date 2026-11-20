# Unknown-order SwiGLU recovery ladder report

- synthetic teacher: False
- pool: 21524 points, dimension 3072, splits {'train': 12914, 'selection': 4305, 'validation': 4305}
- ladder: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24] (early-stop threshold r_sel <= 0.05)

| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |
|---|---|---|---|---|---|---|
| 4 | 0.40169 | 0.41517 | 0.40616 | False | False | 882307 |
| 5 | 0.31754 | 0.32652 | 0.45018 | False | False | 885361 |
| 6 | 0.13987 | 0.14665 | 0.21107 | False | False | 888621 |
| 8 | 0.11687 | 0.11723 | 0.33626 | False | False | 896377 |
| 10 | 0.08398 | 0.08375 | 0.36006 | False | False | 904823 |
| 12 | 0.07073 | 0.06964 | 0.20835 | False | False | 915211 |
| 14 | 0.05280 | 0.05280 | 0.22238 | False | False | 925543 |
| 16 | 0.04330 | 0.04247 | 0.18838 | True | False | 936565 |
| 18 | 0.02826 | 0.02815 | 0.17209 | True | False | 941619 |

- inferred order: 16
- verdict: inferred-order
- upper-region non-identifiable: True
