# Unknown-order SwiGLU recovery ladder report

- synthetic teacher: False
- pool: 21524 points, dimension 3072, splits {'train': 12914, 'selection': 4305, 'validation': 4305}
- ladder: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24] (early-stop threshold r_sel <= 0.05)

| candidate | r_sel | r_eval | r_ood | sufficient | fit failed | oracle calls |
|---|---|---|---|---|---|---|
| 4 | 0.07001 | 0.06530 | 0.12256 | False | False | 882563 |
| 5 | 0.09636 | 0.09390 | 0.27670 | False | False | 884975 |
| 6 | 0.07128 | 0.07018 | 0.28704 | False | False | 888169 |
| 8 | 0.04344 | 0.04331 | 0.21659 | True | False | 896615 |
| 10 | 0.03399 | 0.03423 | 0.18131 | True | False | 904025 |
| 12 | 0.02245 | 0.02251 | 0.17055 | True | False | 914411 |

- inferred order: 8
- verdict: inferred-order
- upper-region non-identifiable: True
