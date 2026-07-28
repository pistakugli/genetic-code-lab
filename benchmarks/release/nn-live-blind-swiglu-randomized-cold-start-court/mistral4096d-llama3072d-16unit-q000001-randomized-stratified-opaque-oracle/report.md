# Randomized Stratified SwiGLU Cold-Start Court

- status: `fail`
- master seed: `260729`
- cases: `1/6`
- families: `llama, mistral`
- layer bands: `early, late, middle`
- oracle modes: `down-proj-output-channel, selected-unit-sum`
- min R2: `0.9999999496492864`
- min paired atom cosine: `0.9999996141368153`
- max oracle calls per case: `1451682/1500000`

| case | family | band | dim | layer | oracle mode | R2 | worst error | pairs | calls | cause | pass |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| mistral-4096d-early-layer9 | mistral | early | 4096 | 9 | down-proj-output-channel | 0.9924715090375857 (observed) | 0.000060398946622265715 (observed) | n/a | 1434164 (observed) | recovered-but-low-r2 | false |
| mistral-4096d-middle-layer11 | mistral | middle | 4096 | 11 | selected-unit-sum | 0.9999999496492864 | 0.0000320648457047476 | 16/16 | 1451682 | pass | true |
| mistral-4096d-late-layer28 | mistral | late | 4096 | 28 | down-proj-output-channel | 0.992711818353843 (observed) | 0.00005846314179156416 (observed) | n/a | 1431866 (observed) | recovered-but-low-r2 | false |
| llama-3072d-early-layer1 | llama | early | 3072 | 1 | down-proj-output-channel | 0.9985555114427959 (observed) | 0.002785015657273862 (observed) | n/a | 1189114 (observed) | recovered-but-low-r2 | false |
| llama-3072d-middle-layer10 | llama | middle | 3072 | 10 | selected-unit-sum | 0.9999991255648174 (observed) | 0.0055576882292353025 (observed) | n/a | 1198890 (observed) | recovered-but-worst-error-over-bar | false |
| llama-3072d-late-layer23 | llama | late | 3072 | 23 | down-proj-output-channel | 0.9951764733429751 (observed) | 0.013968965675283326 (observed) | n/a | 1202024 (observed) | recovered-but-inaccurate | false |

## Failures

- mistral-4096d-early-layer9: orchestrator produced no report (exit 1); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.9924715090375857; observedWorstError=0.000060398946622265715
- mistral-4096d-late-layer28: orchestrator produced no report (exit 1); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.992711818353843; observedWorstError=0.00005846314179156416
- llama-3072d-early-layer1: orchestrator produced no report (exit 1); cause=recovered-but-low-r2; lastPhase=validation-complete; observedR2=0.9985555114427959; observedWorstError=0.002785015657273862
- llama-3072d-middle-layer10: orchestrator produced no report (exit 1); cause=recovered-but-worst-error-over-bar; lastPhase=validation-complete; observedR2=0.9999991255648174; observedWorstError=0.0055576882292353025
- llama-3072d-late-layer23: orchestrator produced no report (exit 1); cause=recovered-but-inaccurate; lastPhase=validation-complete; observedR2=0.9951764733429751; observedWorstError=0.013968965675283326
