# Rank-Adaptive Mixed GELU FFN Structure Benchmark

Generated: 2026-06-02T07:17:30.827Z

## Claim Scope

- Proves if pass: GCL can recover explicit mixed GELU atom structure when given a maximum channel budget but not the true channel count
- Does not claim: full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, or broad SOTA status

## Summary

- Pass: true
- True channel count: 3
- Max channel budget given to GCL: 4
- Effective recovered channel count: 3
- Pruned weak atoms: 0
- Duplicate pairs: 0
- MAE: 0.0014707898276633694
- R2: 0.9999899685243602
- Min direction projection: 0.9998572617229938
- Max weight-norm error: 0.03168688372064543
- Max bias error: 0.009782376019626393
- Raw oracle calls: 14458

## Gate

- recoversRankAdaptiveStructure: true
- doesNotUseTrueChannelCount: true
- effectiveChannelCountMatches: true
- usesInternalCandidateSelection: true
- duplicatePairsAbsent: true
- heldoutR2Above0999: true
- heldoutMaeBelow002: true
- minDirectionProjectionAbove0995: true
- structuralParameterErrorsBounded: true
- beatsMeanBy90Percent: true

## Conclusion

The rank-adaptive mixed GELU structure gate passed. GCL recovered explicit atoms and the effective channel count without being told the true count.
