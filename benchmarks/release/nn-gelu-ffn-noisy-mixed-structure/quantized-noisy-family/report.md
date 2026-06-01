# Noisy Mixed GELU FFN Structure Benchmark

Generated: 2026-06-01T16:23:44.938Z

## Claim Scope

- Proves if pass: GCL can recover explicit mixed GELU atom structure under controlled stochastic noise and quantization, including balanced, near-parallel, and weak-but-resolvable channel profiles, while beating dependency-free surrogate baselines on held-out noiseless truth
- Does not claim: full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM block decompilation, noisy SwiGLU mixed-block recovery, semantic original-neuron identity from output-only access, or broad SOTA status

## Summary

- Pass: true
- Cases: 3
- Passed cases: 3
- Failed cases: none
- Max MAE: 0.003728882150765997
- Min R2: 0.9997459345781228
- Max worst error: 0.04908248358529477
- Min direction projection: 0.9987338205891503
- Effective channel counts: 3, 3, 3
- Canonical atom counts: 3, 3, 3
- Max GCL/best-baseline MAE ratio: 0.04297672596679889
- Max raw oracle calls: 155012

## Cases

| Case | Profile | Pass | MAE | R2 | Canonical atoms | Best baseline | GCL/baseline MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced-three-channel-noisy-24d | balanced-noisy | true | 0.0032677474320668303 | 0.999944613493118 | 3 | random-gelu-ridge | 0.014699144721447517 |
| near-parallel-three-channel-noisy-16d | near-parallel-noisy | true | 0.003600342312263999 | 0.9997459345781228 | 3 | random-gelu-ridge | 0.04297672596679889 |
| weak-resolvable-three-channel-noisy-12d | weak-resolvable-noisy | true | 0.003728882150765997 | 0.9999181646486015 | 3 | random-gelu-ridge | 0.030734465623733977 |

## Gate Details

### balanced-three-channel-noisy-24d

- Profile: balanced-noisy
- Noise std: 0.0025
- Quantization: 0.001
- Query samples: 15
- Summary: `{"pass":true,"mae":0.0032677474320668303,"rmse":0.004301418014557782,"r2":0.999944613493118,"worstError":0.018258777737773624,"effectiveChannelCount":3,"canonicalAtomCount":3,"minDirectionProjection":0.9997152450697433,"duplicatePairCount":0,"nearParallelPairCount":0,"prunedWeakAtomCount":0,"bestBaselineId":"random-gelu-ridge","bestBaselineMae":0.22230867808919938,"gclVsBestBaselineMaeRatio":0.014699144721447517,"strongOnlyMae":0.22900940666322703,"rawOracleCalls":137970}`
- outputR2AboveThreshold: true
- outputMaeBelowThreshold: true
- worstErrorBelowThreshold: true
- effectiveChannelCountMatches: true
- canonicalAtomCountMatches: true
- directionRecoveryAboveThreshold: true
- noDuplicateCollapse: true
- weakResolvableSignalNotPruned: true
- beatsBestBaselineByMae: true
- rawOracleCallsBelowBudget: true

### near-parallel-three-channel-noisy-16d

- Profile: near-parallel-noisy
- Noise std: 0.002
- Quantization: 0.001
- Query samples: 11
- Summary: `{"pass":true,"mae":0.003600342312263999,"rmse":0.004711240353647738,"r2":0.9997459345781228,"worstError":0.019976668856817058,"effectiveChannelCount":3,"canonicalAtomCount":3,"minDirectionProjection":0.9995791895618263,"duplicatePairCount":0,"nearParallelPairCount":1,"prunedWeakAtomCount":0,"bestBaselineId":"random-gelu-ridge","bestBaselineMae":0.08377423433896282,"gclVsBestBaselineMaeRatio":0.04297672596679889,"strongOnlyMae":0.2478805455002617,"rawOracleCalls":116468}`
- outputR2AboveThreshold: true
- outputMaeBelowThreshold: true
- worstErrorBelowThreshold: true
- effectiveChannelCountMatches: true
- canonicalAtomCountMatches: true
- directionRecoveryAboveThreshold: true
- noDuplicateCollapse: true
- weakResolvableSignalNotPruned: true
- beatsBestBaselineByMae: true
- rawOracleCallsBelowBudget: true

### weak-resolvable-three-channel-noisy-12d

- Profile: weak-resolvable-noisy
- Noise std: 0.003
- Quantization: 0.001
- Query samples: 13
- Summary: `{"pass":true,"mae":0.003728882150765997,"rmse":0.004872093510514659,"r2":0.9999181646486015,"worstError":0.04908248358529477,"effectiveChannelCount":3,"canonicalAtomCount":3,"minDirectionProjection":0.9987338205891503,"duplicatePairCount":0,"nearParallelPairCount":0,"prunedWeakAtomCount":0,"bestBaselineId":"random-gelu-ridge","bestBaselineMae":0.1213257518909473,"gclVsBestBaselineMaeRatio":0.030734465623733977,"strongOnlyMae":0.03284863477358395,"rawOracleCalls":155012}`
- outputR2AboveThreshold: true
- outputMaeBelowThreshold: true
- worstErrorBelowThreshold: true
- effectiveChannelCountMatches: true
- canonicalAtomCountMatches: true
- directionRecoveryAboveThreshold: true
- noDuplicateCollapse: true
- weakResolvableSignalNotPruned: true
- beatsBestBaselineByMae: true
- rawOracleCallsBelowBudget: true

## Conclusion

The noisy mixed GELU structure gate passed. GCL recovered explicit atom structure under stochastic noise/quantization and beat the dependency-free baseline court on held-out noiseless truth.
