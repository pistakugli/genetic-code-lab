# Noisy Mixed SwiGLU FFN Structure Benchmark

Generated: 2026-06-03T07:34:06.225Z

## Claim Scope

- Proves if pass: GCL can recover explicit mixed SwiGLU value/gate pair structure under controlled stochastic noise and quantization, including balanced, near-parallel value, and weak-but-resolvable pair profiles
- Does not claim: full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production transformer block decompilation, semantic original-neuron identity from output-only access, exact hidden weights under non-identifiability, XGBoost/RF/MLP structure-decompiler superiority, or broad SOTA status

## Summary

- Pass: true
- Cases: 3
- Passed cases: 3
- Failed cases: none
- Profiles: balanced-noisy, near-parallel-values-noisy, weak-resolvable-noisy
- Effective pair counts: 2, 2, 3
- Max MAE: 0.003938356995627657
- Min R2: 0.999874503247406
- Max worst error: 0.03733335221578926
- Min value-direction projection: 0.999310139471901
- Min gate-direction projection: 0.9998244527769446
- Min mean pair projection: 0.9998467737576421
- Max distribution MAE: 0.007389576137780751
- Min distribution R2: 0.999874503247406
- Max distribution worst error: 0.09266107798318846
- Max raw oracle calls: 345960

## Cases

| Case | Profile | Pass | MAE | R2 | Pairs | Min value projection | Min gate projection | Raw calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced-two-pair-noisy-24d | balanced-noisy | true | 0.0037579706586037564 | 0.9999406274141684 | 2 | 0.9999378067025928 | 0.9999768341611212 | 164824 |
| near-parallel-values-two-pair-noisy-18d | near-parallel-values-noisy | true | 0.003938356995627657 | 0.999874503247406 | 2 | 0.9999782353877339 | 0.9999452377790431 | 247936 |
| weak-resolvable-three-pair-noisy-16d | weak-resolvable-noisy | true | 0.003426551755394013 | 0.9999282551187072 | 3 | 0.999310139471901 | 0.9998244527769446 | 345960 |

## Distribution Holdout

| Case | Distribution | Kind | MAE | R2 | Worst error |
| --- | --- | --- | --- | --- | --- |
| balanced-two-pair-noisy-24d | iid-uniform | iid | 0.0037579706586037564 | 0.9999406274141684 | 0.03733335221578926 |
| balanced-two-pair-noisy-24d | wide-ood | ood | 0.006460436110556813 | 0.9999414750648691 | 0.07685782123507856 |
| balanced-two-pair-noisy-24d | shifted-ood | ood | 0.0029023186114491544 | 0.9998938079476508 | 0.030454253694227162 |
| near-parallel-values-two-pair-noisy-18d | iid-uniform | iid | 0.003938356995627657 | 0.999874503247406 | 0.03516047415928325 |
| near-parallel-values-two-pair-noisy-18d | wide-ood | ood | 0.007389576137780751 | 0.9998770436374168 | 0.09266107798318846 |
| near-parallel-values-two-pair-noisy-18d | shifted-ood | ood | 0.003942890839124986 | 0.9999132752153321 | 0.05641600724185558 |
| weak-resolvable-three-pair-noisy-16d | iid-uniform | iid | 0.003426551755394013 | 0.9999282551187072 | 0.024054658545496155 |
| weak-resolvable-three-pair-noisy-16d | wide-ood | ood | 0.0069086193985408795 | 0.999893918836575 | 0.07212389769463412 |
| weak-resolvable-three-pair-noisy-16d | shifted-ood | ood | 0.003910524928887284 | 0.9999444986026761 | 0.03559930484365581 |

## Gate Details

### balanced-two-pair-noisy-24d

- Profile: balanced-noisy
- Noise std: 0.002
- Quantization: 0.00075
- Query samples: 11
- Summary: `{"pass":true,"mae":0.0037579706586037564,"rmse":0.005696901877487932,"r2":0.9999406274141684,"worstError":0.03733335221578926,"effectivePairCount":2,"rawPairCount":3,"prunedWeakPairCount":0,"duplicatePairCount":0,"minValueDirectionProjection":0.9999378067025928,"minGateDirectionProjection":0.9999768341611212,"meanPairProjection":0.9999734797748244,"recoveredNearParallelGatePairs":0,"recoveredNearParallelValuePairs":0,"strongOnlyMae":0.154815450854355,"maxDistributionMae":0.006460436110556813,"minDistributionR2":0.9998938079476508,"maxDistributionWorstError":0.07685782123507856,"rawOracleCalls":164824}`
- recoversRankAdaptiveStructure: true
- doesNotUseTruePairCount: true
- effectivePairCountMatches: true
- duplicatePairsAbsent: true
- valueDirectionRecoveryAboveThreshold: true
- gateDirectionRecoveryAboveThreshold: true
- meanPairProjectionAboveThreshold: true
- iidR2AboveThreshold: true
- iidMaeBelowThreshold: true
- iidWorstErrorBelowThreshold: true
- allDistributionsR2AboveThreshold: true
- allDistributionsMaeBelowThreshold: true
- allDistributionsWorstErrorBelowThreshold: true
- nearParallelValueDiagnosticPresent: true
- weakResolvableSignalNotPruned: true
- rawOracleCallsBelowBudget: true

### near-parallel-values-two-pair-noisy-18d

- Profile: near-parallel-values-noisy
- Noise std: 0.0025
- Quantization: 0.001
- Query samples: 13
- Summary: `{"pass":true,"mae":0.003938356995627657,"rmse":0.005909827167030297,"r2":0.999874503247406,"worstError":0.03516047415928325,"effectivePairCount":2,"rawPairCount":3,"prunedWeakPairCount":0,"duplicatePairCount":0,"minValueDirectionProjection":0.9999782353877339,"minGateDirectionProjection":0.9999452377790431,"meanPairProjection":0.9999672977292233,"recoveredNearParallelGatePairs":0,"recoveredNearParallelValuePairs":1,"strongOnlyMae":0.13619641586957124,"maxDistributionMae":0.007389576137780751,"minDistributionR2":0.999874503247406,"maxDistributionWorstError":0.09266107798318846,"rawOracleCalls":247936}`
- recoversRankAdaptiveStructure: true
- doesNotUseTruePairCount: true
- effectivePairCountMatches: true
- duplicatePairsAbsent: true
- valueDirectionRecoveryAboveThreshold: true
- gateDirectionRecoveryAboveThreshold: true
- meanPairProjectionAboveThreshold: true
- iidR2AboveThreshold: true
- iidMaeBelowThreshold: true
- iidWorstErrorBelowThreshold: true
- allDistributionsR2AboveThreshold: true
- allDistributionsMaeBelowThreshold: true
- allDistributionsWorstErrorBelowThreshold: true
- nearParallelValueDiagnosticPresent: true
- weakResolvableSignalNotPruned: true
- rawOracleCallsBelowBudget: true

### weak-resolvable-three-pair-noisy-16d

- Profile: weak-resolvable-noisy
- Noise std: 0.0025
- Quantization: 0.001
- Query samples: 15
- Summary: `{"pass":true,"mae":0.003426551755394013,"rmse":0.004650614759550887,"r2":0.9999282551187072,"worstError":0.024054658545496155,"effectivePairCount":3,"rawPairCount":4,"prunedWeakPairCount":0,"duplicatePairCount":0,"minValueDirectionProjection":0.999310139471901,"minGateDirectionProjection":0.9998244527769446,"meanPairProjection":0.9998467737576421,"recoveredNearParallelGatePairs":0,"recoveredNearParallelValuePairs":0,"strongOnlyMae":0.042708523960866676,"maxDistributionMae":0.0069086193985408795,"minDistributionR2":0.999893918836575,"maxDistributionWorstError":0.07212389769463412,"rawOracleCalls":345960}`
- recoversRankAdaptiveStructure: true
- doesNotUseTruePairCount: true
- effectivePairCountMatches: true
- duplicatePairsAbsent: true
- valueDirectionRecoveryAboveThreshold: true
- gateDirectionRecoveryAboveThreshold: true
- meanPairProjectionAboveThreshold: true
- iidR2AboveThreshold: true
- iidMaeBelowThreshold: true
- iidWorstErrorBelowThreshold: true
- allDistributionsR2AboveThreshold: true
- allDistributionsMaeBelowThreshold: true
- allDistributionsWorstErrorBelowThreshold: true
- nearParallelValueDiagnosticPresent: true
- weakResolvableSignalNotPruned: true
- rawOracleCallsBelowBudget: true

## Conclusion

The noisy mixed SwiGLU structure gate passed. GCL recovered explicit value/gate pairs and effective pair counts from repeated noisy/quantized oracle calls across balanced, near-parallel-value, and weak-resolvable profiles.
