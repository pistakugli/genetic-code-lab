# Randomized Stratified SwiGLU Cold-Start Court

- status: `fail`
- master seed: `260729`
- cases: `1/6`
- families: `llama, mistral`
- layer bands: `early, late, middle`
- oracle modes: `down-proj-output-channel, selected-unit-sum`
- metric coverage: `6/6` cases produced a gated report
- min R2 over gated cases: `0.992711805955406`
- min R2 including ungated cases: `0.992711805955406`
- min paired atom cosine: `-0.06442893806971481`
- max oracle calls per case: `1455778/1500000`

| case | family | band | dim | layer | oracle mode | R2 | worst error | pairs | calls | cause | pass |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| mistral-4096d-early-layer9 | mistral | early | 4096 | 9 | down-proj-output-channel | 0.9939420260874505 | 0.00005758570751607735 | 11/16 | 1438260 | report-failed-court-checks | false |
| mistral-4096d-middle-layer11 | mistral | middle | 4096 | 11 | selected-unit-sum | 0.9999999496492865 | 0.0000320648452153266 | 16/16 | 1455778 | pass | true |
| mistral-4096d-late-layer28 | mistral | late | 4096 | 28 | down-proj-output-channel | 0.992711805955406 | 0.00005844894483986487 | 11/16 | 1435962 | report-failed-court-checks | false |
| llama-3072d-early-layer1 | llama | early | 3072 | 1 | down-proj-output-channel | 0.9988458593119249 | 0.0028897546313992415 | 12/16 | 1193210 | report-failed-court-checks | false |
| llama-3072d-middle-layer10 | llama | middle | 3072 | 10 | selected-unit-sum | 0.9999991252803332 | 0.005560692837129122 | 16/16 | 1202986 | report-failed-court-checks | false |
| llama-3072d-late-layer23 | llama | late | 3072 | 23 | down-proj-output-channel | 0.9980053437372589 | 0.0068207934774785345 | 12/16 | 1206120 | report-failed-court-checks | false |

## Failures

- mistral-4096d-early-layer9: report is not pass
- mistral-4096d-early-layer9: R2=0.9939420260874505 < 0.999
- mistral-4096d-early-layer9: subspace cosine=0.9853595369461892 < 0.999
- mistral-4096d-early-layer9: paired atom minimum=-0.06442893806971481 < 0.99
- mistral-4096d-early-layer9: paired atom count is not 16/16: 11
- mistral-4096d-late-layer28: report is not pass
- mistral-4096d-late-layer28: R2=0.992711805955406 < 0.999
- mistral-4096d-late-layer28: subspace cosine=0.9790955523785366 < 0.999
- mistral-4096d-late-layer28: paired atom minimum=0.06556068217791547 < 0.99
- mistral-4096d-late-layer28: paired atom count is not 16/16: 11
- llama-3072d-early-layer1: report is not pass
- llama-3072d-early-layer1: R2=0.9988458593119249 < 0.999
- llama-3072d-early-layer1: subspace cosine=0.9982453221496379 < 0.999
- llama-3072d-early-layer1: paired atom minimum=0.011313265986362696 < 0.99
- llama-3072d-early-layer1: paired atom count is not 16/16: 12
- llama-3072d-middle-layer10: report is not pass
- llama-3072d-middle-layer10: worstError=0.005560692837129122 > 0.005
- llama-3072d-late-layer23: report is not pass
- llama-3072d-late-layer23: R2=0.9980053437372589 < 0.999
- llama-3072d-late-layer23: worstError=0.0068207934774785345 > 0.005
- llama-3072d-late-layer23: paired atom minimum=0.10087696558935158 < 0.99
- llama-3072d-late-layer23: paired atom count is not 16/16: 12
