# Deep Black-Box ReLU ND Region-Sampler Benchmark

Generated: 2026-06-06T05:53:17.405Z

## Claim Scope

- Proves: benchmark-scoped output-equivalent affine-region recovery in 3D/4D using random local-gradient probes and query-free kNN region assignment
- Does not claim: unique hidden-layer recovery, exact layer attribution, topology-perfect region merging, noisy ND deep recovery, or arbitrary production API extraction

## Summary

- Pass: true
- Teachers passed: 4/4
- Dimensions: 3, 4
- Min recovered regions: 40
- Min deep-only region estimate: 35
- Max MAE: 0.011095250643373489
- Min R2: 0.998534063321796
- Min linear MAE improvement ratio: 42.81846369086524
- Total raw oracle calls: 315800

## Results

### deep-3d-11

- Pass: true
- Dimension: 3
- Recovered regions: 40
- Deep-only region estimate: 35
- Retained samples: 6500
- Dropped samples: 0
- Max region fit RMSE: 1.1763100133136143e-9
- GCL MAE: 0.002430047386382316
- GCL R2: 0.999809260995288
- Linear MAE: 0.3942185084817401
- Linear improvement ratio: 162.2266753689215
- Raw oracle calls: 47700

### deep-3d-29

- Pass: true
- Dimension: 3
- Recovered regions: 42
- Deep-only region estimate: 37
- Retained samples: 6482
- Dropped samples: 18
- Max region fit RMSE: 2.8692085419217e-10
- GCL MAE: 0.0038965794152928937
- GCL R2: 0.9997521846260714
- Linear MAE: 0.500048333402195
- Linear improvement ratio: 128.33007623036164
- Raw oracle calls: 47700

### deep-4d-11

- Pass: true
- Dimension: 4
- Recovered regions: 47
- Deep-only region estimate: 42
- Retained samples: 11978
- Dropped samples: 22
- Max region fit RMSE: 3.975276128948011e-10
- GCL MAE: 0.011095250643373489
- GCL R2: 0.998534063321796
- Linear MAE: 0.5614313392808276
- Linear improvement ratio: 50.60105060502946
- Raw oracle calls: 110200

### deep-4d-29

- Pass: true
- Dimension: 4
- Recovered regions: 46
- Deep-only region estimate: 41
- Retained samples: 11985
- Dropped samples: 15
- Max region fit RMSE: 1.2211343227402162e-8
- GCL MAE: 0.00794338030550133
- GCL R2: 0.9995218599549637
- Linear MAE: 0.34012334119384274
- Linear improvement ratio: 42.81846369086524
- Raw oracle calls: 110200

