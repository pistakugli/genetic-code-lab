# GELU FFN Formalization Negative-Case Benchmark

Generated: 2026-06-02T07:21:34.216Z

## Claim Scope

- Proves if pass: GCL has executable guardrails for duplicate/cancellation canonicalization, weak-under-noise non-identifiability, weak-but-resolvable channel recovery, activation-family mismatch rejection, and minimum equivalent structure under overcomplete budgets
- Does not claim: semantic identity of original hidden neurons, noisy full mixed-block recovery, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM decompilation, or broad SOTA status

## Summary

- Pass: true
- Cases: 5
- Passed cases: 5
- Failed cases: none

## Cases

| Case | Family | Expected behavior | Pass |
| --- | --- | --- | --- |
| manual-duplicate-cancellation-weak-canonicalizer | exact duplicate split + equal/opposite cancellation + weak atom | collapse duplicate split, remove cancelling pair, prune weak atom, and keep only observable canonical support | true |
| weak-channel-below-noise | weak hidden channel under stochastic output noise and quantization | keep output fidelity but refuse to claim all original hidden channels are identifiable | true |
| weak-channel-above-noise-must-recover | weak but resolvable hidden channel above stochastic output noise and quantization | recover the third channel when held-out oracle evidence makes it identifiable; do not always collapse to the smallest model | true |
| mis-specified-activation-family | sinusoidal teacher passed to GELU structure finder | do not promote a GELU structure claim when held-out residuals show activation-family mismatch | true |
| overcomplete-budget-minimum-equivalent | two-channel GELU teacher with five-channel solver budget | select the smallest held-out-valid structure instead of promoting the overcomplete budget | true |

## Gate Details

### manual-duplicate-cancellation-weak-canonicalizer

- Family: exact duplicate split + equal/opposite cancellation + weak atom
- Expected correct behavior: collapse duplicate split, remove cancelling pair, prune weak atom, and keep only observable canonical support
- Pass: true
- Summary: `{"canonicalAtomCount":1,"mergedDuplicateCount":2,"removedCancellationCount":2,"removedWeakAtomCount":1}`
- duplicateSplitCollapsed: true
- cancellationRemoved: true
- weakAtomRemoved: true
- onlyObservableAtomRemains: true

### weak-channel-below-noise

- Family: weak hidden channel under stochastic output noise and quantization
- Expected correct behavior: keep output fidelity but refuse to claim all original hidden channels are identifiable
- Pass: true
- Summary: `{"effectiveChannelCount":2,"canonicalAtomCount":2,"selectedCandidateKind":"trained-4-strength-threshold","bestCandidateMetrics":{"mae":0.010372755537322597,"rmse":0.01337573005325225,"r2":0.9992167195156504},"selectedValidationMetrics":{"mae":0.024751667583372407,"rmse":0.02844370737365886,"r2":0.9964579497387951},"prunedWeakAtomCount":2,"signatureRemovedWeakAtomCount":0,"mae":0.023809979188975757,"r2":0.9968910245119376,"maxWeakContribution":0.005819782566052295,"rawOracleCalls":43064}`
- outputStillFaithful: true
- weakSignalBelowNoiseFloor: true
- hiddenWeakAtomNotPromotedAsSemanticRecovery: true
- weakOrPrunedEvidenceRecorded: true
- rawOracleCallsBelow80000: true

### weak-channel-above-noise-must-recover

- Family: weak but resolvable hidden channel above stochastic output noise and quantization
- Expected correct behavior: recover the third channel when held-out oracle evidence makes it identifiable; do not always collapse to the smallest model
- Pass: true
- Summary: `{"effectiveChannelCount":3,"canonicalAtomCount":3,"selectedCandidateKind":"trained-3-strength-threshold","selectedCandidateRank":5,"bestCandidateMetrics":{"mae":0.006772337571127612,"rmse":0.00880801277207202,"r2":0.9996273783515787},"selectedValidationMetrics":{"mae":0.011372461639168507,"rmse":0.013805598666581412,"r2":0.9990845751456228},"prunedWeakAtomCount":0,"selectedAtomStrengths":[0.857706044816364,0.5220840237278643,0.06893052988147018],"mae":0.010968259412436983,"r2":0.9992502726292716,"strongOnlyMae":0.03121065181034743,"strongOnlyR2":0.9904076987558283,"maxWeakContribution":0.21016262002998667,"meanWeakContribution":0.03121065181034743,"noiseStd":0.003,"quantization":0.001,"rawOracleCalls":104524}`
- outputFaithful: true
- weakSignalAboveNoiseFloor: true
- strongOnlyUnderfitsHeldoutOracle: true
- exactlyResolvableThreeAtomsRecovered: true
- rawOracleCallsBelow120000: true

### mis-specified-activation-family

- Family: sinusoidal teacher passed to GELU structure finder
- Expected correct behavior: do not promote a GELU structure claim when held-out residuals show activation-family mismatch
- Pass: true
- Summary: `{"effectiveChannelCount":3,"canonicalAtomCount":3,"mae":0.11213943566373409,"r2":0.8777833727574628,"worstError":2.3545163455618057,"wouldPromoteStructure":false,"rawOracleCalls":3800}`
- mismatchNotPromotedAsGeluStructure: true
- residualFailureIsMeasurable: true
- attemptedModelRecordedAtoms: true
- rawOracleCallsBelow45000: true

### overcomplete-budget-minimum-equivalent

- Family: two-channel GELU teacher with five-channel solver budget
- Expected correct behavior: select the smallest held-out-valid structure instead of promoting the overcomplete budget
- Pass: true
- Summary: `{"rawChannelCount":5,"effectiveChannelCount":2,"canonicalAtomCount":2,"selectedCandidateKind":"trained-3-strength-threshold","mae":0.00610512989140079,"r2":0.999703082360723,"rawOracleCalls":7130}`
- selectedMinimumEquivalentCount: true
- overcompleteBudgetNotUsedAsClaim: true
- heldoutR2Above0999: true
- heldoutMaeBelow002: true
- rawOracleCallsBelow70000: true

## Conclusion

The negative-case gate passed. The formalization is now backed by executable checks that prevent unsupported structure promotion in these benchmarked failure modes.
