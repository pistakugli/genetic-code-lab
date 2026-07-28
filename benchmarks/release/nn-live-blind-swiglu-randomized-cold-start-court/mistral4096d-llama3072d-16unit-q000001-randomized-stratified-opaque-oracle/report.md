# Randomized Stratified SwiGLU Cold-Start Court

- status: `fail`
- master seed: `260729`
- cases: `1/6`
- families: `llama, mistral`
- layer bands: `early, late, middle`
- oracle modes: `down-proj-output-channel, selected-unit-sum`
- min R2: `0.9999999496492865`
- min paired atom cosine: `0.9999996141369001`
- max oracle calls per case: `1455778/1500000`

| case | family | band | dim | layer | oracle mode | R2 | worst error | pairs | calls | cause | pass |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| mistral-4096d-early-layer9 | mistral | early | 4096 | 9 | down-proj-output-channel | 0.9939420260874505 (observed) | 0.00005758570751607735 (observed) | n/a | 1438260 (observed) | recovered-but-low-r2 | false |
| mistral-4096d-middle-layer11 | mistral | middle | 4096 | 11 | selected-unit-sum | 0.9999999496492865 | 0.0000320648452153266 | 16/16 | 1455778 | pass | true |
| mistral-4096d-late-layer28 | mistral | late | 4096 | 28 | down-proj-output-channel | 0.992711805955406 (observed) | 0.00005844894483986487 (observed) | n/a | 1435962 (observed) | recovered-but-low-r2 | false |
| llama-3072d-early-layer1 | llama | early | 3072 | 1 | down-proj-output-channel | 0.9988458593119249 (observed) | 0.0028897546313992415 (observed) | n/a | 1193210 (observed) | recovered-but-low-r2 | false |
| llama-3072d-middle-layer10 | llama | middle | 3072 | 10 | selected-unit-sum | 0.9999991252803332 (observed) | 0.005560692837129122 (observed) | n/a | 1202986 (observed) | recovered-but-worst-error-over-bar | false |
| llama-3072d-late-layer23 | llama | late | 3072 | 23 | down-proj-output-channel | 0.9980053437372589 (observed) | 0.0068207934774785345 (observed) | n/a | 1206120 (observed) | recovered-but-inaccurate | false |

## Failures

- mistral-4096d-early-layer9: orchestrator produced no report (exit 1); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.9939420260874505; observedWorstError=0.00005758570751607735
- mistral-4096d-late-layer28: orchestrator produced no report (exit 1); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.992711805955406; observedWorstError=0.00005844894483986487
- llama-3072d-early-layer1: orchestrator produced no report (exit 1); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.9988458593119249; observedWorstError=0.0028897546313992415
- llama-3072d-middle-layer10: orchestrator produced no report (exit 1); cause=recovered-but-worst-error-over-bar; lastPhase=validation-complete; observedR2=0.9999991252803332; observedWorstError=0.005560692837129122
- llama-3072d-late-layer23: orchestrator produced no report (exit 1); cause=recovered-but-inaccurate; lastPhase=validation-complete; observedR2=0.9980053437372589; observedWorstError=0.0068207934774785345
