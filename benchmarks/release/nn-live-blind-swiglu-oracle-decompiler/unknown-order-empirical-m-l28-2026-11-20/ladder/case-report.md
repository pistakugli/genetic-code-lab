# Unknown-order SwiGLU recovery ladder report

- synthetic teacher: False
- pool: 15998 points, dimension 4096, splits {'train': 9598, 'selection': 3200, 'validation': 3200}
- ladder: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24] (early-stop threshold r_sel <= 0.05)

| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |
|---|---|---|---|---|---|---|
| 4 | 0.52392 | 0.52386 | 3.48777 | False | False | 1145330 |
| 5 | 0.43343 | 0.44627 | 3.72569 | False | False | 1147850 |
| 6 | 0.44080 | 0.43700 | 4.54235 | False | False | 1151216 |
| 8 | 0.23985 | 0.25302 | 1.26337 | False | False | 1158432 |
| 10 | 0.12190 | 0.11908 | 0.11929 | False | False | 1166534 |
| 12 | 0.10403 | 0.09947 | 0.46801 | False | False | 1176964 |
| 14 | 0.19292 | 0.19780 | 2.92415 | False | False | 1183370 |
| 16 | 0.19196 | 0.19306 | 2.06530 | False | False | 1186802 |
| 18 | 0.16354 | 0.17073 | 0.21931 | False | False | 1204618 |

- inferred order: none
- verdict: no-sufficient-order
- upper-region non-identifiable: False
