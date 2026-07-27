# Randomized Stratified SwiGLU Cold-Start Court

- status: `fail`
- master seed: `260729`
- cases: `1/6`
- families: `llama, mistral`
- layer bands: `early, late, middle`
- oracle modes: `down-proj-output-channel, selected-unit-sum`
- min R2: `0.9999999194986897`
- min paired atom cosine: `0.9999820862066797`
- max oracle calls per case: `1282868/1500000`

| case | family | band | dim | layer | oracle mode | R2 | worst error | pairs | calls | cause | pass |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| mistral-4096d-early-layer9 | mistral | early | 4096 | 9 | down-proj-output-channel | n/a | n/a | n/a | 1113984 (observed) | structure-not-recovered | false |
| mistral-4096d-middle-layer11 | mistral | middle | 4096 | 11 | selected-unit-sum | 0.9999999194986897 | 0.00009755367550526306 | 16/16 | 1282868 | pass | true |
| mistral-4096d-late-layer28 | mistral | late | 4096 | 28 | down-proj-output-channel | 0.978313682106669 (observed) | 0.0001494694114639203 (observed) | n/a | 1282454 (observed) | recovered-but-low-r2 | false |
| llama-3072d-early-layer1 | llama | early | 3072 | 1 | down-proj-output-channel | 0.9899565033743015 (observed) | 0.010985830786309943 (observed) | n/a | 1011970 (observed) | recovered-but-inaccurate | false |
| llama-3072d-middle-layer10 | llama | middle | 3072 | 10 | selected-unit-sum | 0.9999991253174392 (observed) | 0.005582131817034153 (observed) | n/a | 1019004 (observed) | recovered-but-worst-error-over-bar | false |
| llama-3072d-late-layer23 | llama | late | 3072 | 23 | down-proj-output-channel | 0.9948086207127055 (observed) | 0.01800143558433407 (observed) | n/a | 1016246 (observed) | recovered-but-inaccurate | false |

## Failures

- mistral-4096d-early-layer9: orchestrator produced no report (exit null); cause=structure-not-recovered; lastPhase=asymptotic-cold-start; runtimeError=asymptotic cold start failed: structure-court-rejected
- mistral-4096d-late-layer28: orchestrator produced no report (exit null); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.978313682106669; observedWorstError=0.0001494694114639203
- llama-3072d-early-layer1: orchestrator produced no report (exit null); cause=recovered-but-inaccurate; lastPhase=validation-complete; observedR2=0.9899565033743015; observedWorstError=0.010985830786309943
- llama-3072d-middle-layer10: orchestrator produced no report (exit null); cause=recovered-but-worst-error-over-bar; lastPhase=validation-complete; observedR2=0.9999991253174392; observedWorstError=0.005582131817034153
- llama-3072d-late-layer23: orchestrator produced no report (exit null); cause=recovered-but-inaccurate; lastPhase=validation-complete; observedR2=0.9948086207127055; observedWorstError=0.01800143558433407
