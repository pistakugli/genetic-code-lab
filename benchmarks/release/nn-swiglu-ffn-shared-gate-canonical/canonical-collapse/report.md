# Shared-Gate Cancellation SwiGLU Canonical Benchmark

Generated: 2026-06-06T07:41:57.071Z

## Claim Scope

- Proves if pass: GCL canonicalizes shared-gate/cancellation-heavy SwiGLU degeneracy into the minimal functionally equivalent value/gate pair instead of promoting duplicate source channels
- Does not claim: unique original source-neuron identity from output-only access, exact hidden weights under non-identifiability, full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D transformer block decompilation, or broad SOTA status

## Summary

- Pass: true
- Cases: 2
- Passed cases: 2
- Failed cases: none
- Profiles: shared-gate-clean-collapse, shared-gate-noisy-cancellation
- Source pair counts: 3, 3
- Effective canonical pair counts: 1, 1
- Min canonical value-direction projection: 0.9992111582432534
- Min shared-gate direction projection: 0.9999470719484649
- Min cancellation ratio: 0.05876108139509095
- Max MAE: 0.00241722715125567
- Min R2: 0.9995956095908356
- Max worst error: 0.02882824355941427
- Max distribution MAE: 0.0038654875223167406
- Min distribution R2: 0.9992352419312791
- Max distribution worst error: 0.07707410825476324
- Max raw oracle calls: 276672

## Cases

| Case | Profile | Pass | Source pairs | Effective pairs | Cancellation | MAE | R2 | Value projection | Gate projection | Raw calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-gate-three-source-clean-collapse-14d | shared-gate-clean-collapse | true | 3 | 1 | 0.05876108139509095 | 0.00030495801883697086 | 0.9999997349461373 | 0.9999999984429478 | 0.999999999965965 | 16788 |
| shared-gate-noisy-cancellation-heavy-collapse-16d | shared-gate-noisy-cancellation | true | 3 | 1 | 0.9032502910798934 | 0.00241722715125567 | 0.9995956095908356 | 0.9992111582432534 | 0.9999470719484649 | 276672 |

## Distribution Holdout

| Case | Distribution | Kind | MAE | R2 | Worst error |
| --- | --- | --- | --- | --- | --- |
| shared-gate-three-source-clean-collapse-14d | iid-uniform | iid | 0.00030495801883697086 | 0.9999997349461373 | 0.0033468825935651925 |
| shared-gate-three-source-clean-collapse-14d | wide-ood | ood | 0.0006120496833982786 | 0.9999996546652662 | 0.007513954262032385 |
| shared-gate-three-source-clean-collapse-14d | shifted-ood | ood | 0.0004446939376086271 | 0.999999010580124 | 0.0038621176469595397 |
| shared-gate-noisy-cancellation-heavy-collapse-16d | iid-uniform | iid | 0.00241722715125567 | 0.9995956095908356 | 0.02882824355941427 |
| shared-gate-noisy-cancellation-heavy-collapse-16d | wide-ood | ood | 0.0038654875223167406 | 0.9992352419312791 | 0.07707410825476324 |
| shared-gate-noisy-cancellation-heavy-collapse-16d | shifted-ood | ood | 0.002449652754535476 | 0.9996001575267439 | 0.021109563498013406 |

## Gate Details

### shared-gate-three-source-clean-collapse-14d

- Summary: `{"pass":true,"trueSourcePairCount":3,"canonicalPairCount":1,"rawPairBudgetGivenToGcl":4,"effectivePairCount":1,"rawPairCount":4,"prunedWeakPairCount":0,"duplicatePairCount":0,"canonicalValueDirectionProjection":0.9999999984429478,"sharedGateDirectionProjection":0.999999999965965,"cancellationRatio":0.05876108139509095,"mae":0.00030495801883697086,"r2":0.9999997349461373,"worstError":0.0033468825935651925,"maxDistributionMae":0.0006120496833982786,"minDistributionR2":0.999999010580124,"maxDistributionWorstError":0.007513954262032385,"rawOracleCalls":16788}`
- recoversRankAdaptiveStructure: true
- sourcePairCountExceedsCanonicalCount: true
- canonicalEffectivePairCountIsOne: true
- doesNotUseTrueSourcePairCount: true
- canonicalValueDirectionRecovered: true
- sharedGateDirectionRecovered: true
- cancellationPressurePresent: true
- iidR2AboveThreshold: true
- iidMaeBelowThreshold: true
- allDistributionR2AboveThreshold: true
- allDistributionMaeBelowThreshold: true
- allDistributionWorstErrorBelowThreshold: true
- duplicateRawPairsNotPromoted: true
- rawOracleCallsBelowBudget: true

### shared-gate-noisy-cancellation-heavy-collapse-16d

- Summary: `{"pass":true,"trueSourcePairCount":3,"canonicalPairCount":1,"rawPairBudgetGivenToGcl":4,"effectivePairCount":1,"rawPairCount":4,"prunedWeakPairCount":0,"duplicatePairCount":0,"canonicalValueDirectionProjection":0.9992111582432534,"sharedGateDirectionProjection":0.9999470719484649,"cancellationRatio":0.9032502910798934,"mae":0.00241722715125567,"r2":0.9995956095908356,"worstError":0.02882824355941427,"maxDistributionMae":0.0038654875223167406,"minDistributionR2":0.9992352419312791,"maxDistributionWorstError":0.07707410825476324,"rawOracleCalls":276672}`
- recoversRankAdaptiveStructure: true
- sourcePairCountExceedsCanonicalCount: true
- canonicalEffectivePairCountIsOne: true
- doesNotUseTrueSourcePairCount: true
- canonicalValueDirectionRecovered: true
- sharedGateDirectionRecovered: true
- cancellationPressurePresent: true
- iidR2AboveThreshold: true
- iidMaeBelowThreshold: true
- allDistributionR2AboveThreshold: true
- allDistributionMaeBelowThreshold: true
- allDistributionWorstErrorBelowThreshold: true
- duplicateRawPairsNotPromoted: true
- rawOracleCallsBelowBudget: true

## Conclusion

The shared-gate/cancellation-heavy SwiGLU hard-case passed. GCL collapsed multiple source channels into the minimal canonical value/gate structure and did not promote duplicate non-identifiable source channels.
