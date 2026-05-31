# Active-Subspace FFN Decompiler Benchmark

Generated: 2026-05-31T12:06:51.150Z

## Claim Scope

- Proves if pass: GCL can discover a low-dimensional active subspace in controlled high-dimensional GELU/SwiGLU-style FFN channels and build a faithful output-equivalent surrogate with explicit query budget
- Does not claim: real LLM weight recovery, arbitrary 4096D production model extraction, exact GELU/SwiGLU parameter recovery, tokenizer/model integration, or broad SOTA status

## Summary

- Pass: true
- Teachers: 4/4
- Dimensions: 256, 1024
- Kinds: gelu-single-channel, swiglu-two-direction
- Max MAE: 0.00022774732575327203
- Min R2: 0.9999942261682238
- Min active energy ratio: 0.9999999999999994
- Max gradient residual ratio: 2.7349292596535903e-8
- Min true-direction projection score: 0.9999999999999991
- GCL MAE wins: 4/4
- Max GCL/best-baseline MAE ratio: 0.0006138582300879202
- Max raw oracle calls: 45360

## Results

### gelu-256d-3101

- Kind: gelu-single-channel
- Dimension: 256
- Active rank: 1
- Pass: true
- GCL MAE: 7.650774461605817e-7
- GCL R2: 0.9999999998338722
- Active energy ratio: 1.0000000000000007
- Min projection score: 1.0000000000000002
- Raw oracle calls: 8344
- Best baseline: mean-output
- Best baseline MAE: 0.7122936554181115
- GCL MAE ratio vs best baseline: 0.0000010741039743102674

Ranking:
- gcl-active-subspace: MAE 7.650774461605817e-7, R2 0.9999999998338722, queries 8344
- mean-output: MAE 0.7122936554181115, R2 -0.00005146987388249258, queries 2200
- coordinate-head-local-quadratic-rank1: MAE 0.7688741585775277, R2 -0.4061780386806979, queries 2200
- random-projection-local-quadratic-rank1: MAE 0.7842906486262999, R2 -0.48000884768088525, queries 2200

### swiglu-256d-3201

- Kind: swiglu-two-direction
- Dimension: 256
- Active rank: 2
- Pass: true
- GCL MAE: 0.00022774732575327203
- GCL R2: 0.9999942261682238
- Active energy ratio: 1.0000000000000002
- Min projection score: 0.9999999999999994
- Raw oracle calls: 12816
- Best baseline: mean-output
- Best baseline MAE: 0.37100964781502205
- GCL MAE ratio vs best baseline: 0.0006138582300879202

Ranking:
- gcl-active-subspace: MAE 0.00022774732575327203, R2 0.9999942261682238, queries 12816
- mean-output: MAE 0.37100964781502205, R2 -0.00011417090318688317, queries 3600
- random-projection-local-quadratic-rank2: MAE 0.4695973933456651, R2 -0.2628683133351777, queries 3600
- coordinate-head-local-quadratic-rank2: MAE 0.47050761701087135, R2 -0.23927001981833307, queries 3600

### gelu-1024d-3301

- Kind: gelu-single-channel
- Dimension: 1024
- Active rank: 1
- Pass: true
- GCL MAE: 0.000014829314606933888
- GCL R2: 0.9999997377788716
- Active energy ratio: 1.0000000000000016
- Min projection score: 0.9999999999999991
- Raw oracle calls: 22880
- Best baseline: mean-output
- Best baseline MAE: 0.7187611237846011
- GCL MAE ratio vs best baseline: 0.00002063177057886891

Ranking:
- gcl-active-subspace: MAE 0.000014829314606933888, R2 0.9999997377788716, queries 22880
- mean-output: MAE 0.7187611237846011, R2 -0.000023936636280197376, queries 2400
- random-projection-local-quadratic-rank1: MAE 0.8045945598427616, R2 -0.4606156821151972, queries 2400
- coordinate-head-local-quadratic-rank1: MAE 0.8078573972816085, R2 -0.5043991861223449, queries 2400

### swiglu-1024d-3401

- Kind: swiglu-two-direction
- Dimension: 1024
- Active rank: 2
- Pass: true
- GCL MAE: 0.00021232318128152567
- GCL R2: 0.9999962229228001
- Active energy ratio: 0.9999999999999994
- Min projection score: 0.9999999999999999
- Raw oracle calls: 45360
- Best baseline: mean-output
- Best baseline MAE: 0.36673254915926795
- GCL MAE ratio vs best baseline: 0.0005789591945636547

Ranking:
- gcl-active-subspace: MAE 0.00021232318128152567, R2 0.9999962229228001, queries 45360
- mean-output: MAE 0.36673254915926795, R2 -0.00003296902823191061, queries 4400
- random-projection-local-quadratic-rank2: MAE 0.46792646828003454, R2 -0.31429479109398994, queries 4400
- coordinate-head-local-quadratic-rank2: MAE 0.47457066968885925, R2 -0.33925232266868854, queries 4400

## Conclusion

Active-subspace FFN foundation passed on controlled 256D/1024D GELU and SwiGLU teachers; next step is 4096D stress plus package-backed external baselines.
