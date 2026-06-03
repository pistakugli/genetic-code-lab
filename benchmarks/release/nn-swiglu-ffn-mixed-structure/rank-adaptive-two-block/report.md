# Rank-Adaptive Mixed SwiGLU FFN Structure Benchmark

Generated: 2026-06-03T09:08:50.870Z

## Claim Scope

- Proves if pass: GCL can recover explicit SwiGLU value/gate pair structure when given a maximum pair budget but not the true pair count
- Does not claim: noisy SwiGLU recovery, full GPT-2/LLM FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production transformer block decompilation, semantic original-neuron identity from output-only access, or broad SOTA status

## Structure Summary

- Pass: true
- True pair count: 2
- Max pair budget given to GCL: 3
- Effective recovered pair count: 2
- Pruned weak pairs: 0
- Duplicate pairs: 0
- Min value-direction projection: 0.9999996446143484
- Min gate-direction projection: 0.9999996288279506
- Mean pair projection: 0.9999997837062355
- Max gate-norm error: 0.008049915513471673
- Max gate-bias error: 0.005227321558675863
- Raw oracle calls: 15456

## Prediction Sanity

- MAE: 0.0008717937894303614
- R2: 0.9999961336476628
- Mean baseline MAE: 0.387757525811948
- GCL/mean MAE ratio: 0.0022482962454561306

## Gate

- recoversRankAdaptiveStructure: true
- doesNotUseTruePairCount: true
- effectivePairCountMatches: true
- usesInternalCandidateSelection: true
- duplicatePairsAbsent: true
- heldoutR2Above0995: true
- heldoutMaeBelow0035: true
- minValueDirectionProjectionAbove099: true
- minGateDirectionProjectionAbove099: true
- meanPairProjectionAbove0993: true
- gateParameterErrorsBounded: true
- predictionSanityBeatsMeanBy90Percent: true
- rawOracleCallsBelow150k: true

## Conclusion

The mixed SwiGLU structure gate passed. GCL recovered explicit value/gate pairs and the effective pair count without being told the true count.
