# Noisy Mixed GELU FFN Structure Benchmark

Generated: 2026-06-05T16:03:43.327Z

## Claim Scope

- Proves if pass: GCL can recover explicit mixed GELU atom structure under controlled stochastic noise and quantization, including balanced, near-parallel, and weak-but-resolvable channel profiles; the recovered structure is also prediction-sanity checked against dependency-free, scikit-learn, XGBoost, and sklearn-MLP fit baselines across IID and OOD held-out noiseless truth
- Does not claim: full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D/16384D production LLM block decompilation, noisy SwiGLU mixed-block recovery, PyTorch deep baseline superiority, XGBoost/MLP/RF as structure-decompiler competitors, semantic original-neuron identity from output-only access, or broad SOTA status

## Prediction Baseline Scope

- Role: prediction-sanity only
- Purpose: checks that the recovered explicit GELU structure is not merely worse than strong black-box output fitting on held-out truth
- Not a claim: XGBoost, random forests, and sklearn MLP are not structure decompilers and are not treated as direct rivals for recovered channel identity, directions, or canonical atoms

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
- Max GCL/best-prediction-baseline MAE ratio: 0.08120754460430137
- Distribution ids: iid-uniform, wide-ood, shifted-ood
- Max distribution MAE: 0.005876071181397475
- Min distribution R2: 0.9995038453250289
- Max distribution worst error: 0.09735167309682635
- Max distribution GCL/best-prediction-baseline MAE ratio: 0.08567143575266337
- Max raw oracle calls: 155012

## Cases

| Case | Profile | Pass | MAE | R2 | Canonical atoms | Best prediction baseline | GCL/prediction-baseline MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced-three-channel-noisy-24d | balanced-noisy | true | 0.0032677474320668303 | 0.999944613493118 | 3 | sklearn-poly2-ridge | 0.05434423511145036 |
| near-parallel-three-channel-noisy-16d | near-parallel-noisy | true | 0.003600342312263999 | 0.9997459345781228 | 3 | sklearn-poly2-ridge | 0.06806622054146819 |
| weak-resolvable-three-channel-noisy-12d | weak-resolvable-noisy | true | 0.003728882150765997 | 0.9999181646486015 | 3 | sklearn-poly2-ridge | 0.08120754460430137 |

## Distribution Prediction-Sanity Court

| Case | Distribution | Kind | GCL MAE | GCL R2 | Best prediction baseline | Best prediction baseline MAE | GCL/prediction-baseline MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced-three-channel-noisy-24d | iid-uniform | iid | 0.0032677474320668303 | 0.999944613493118 | sklearn-poly2-ridge | 0.06013052581134432 | 0.05434423511145036 |
| balanced-three-channel-noisy-24d | wide-ood | ood | 0.0051884080522364056 | 0.9999316450434367 | sklearn-poly2-ridge | 0.12867872335664055 | 0.04032063667477048 |
| balanced-three-channel-noisy-24d | shifted-ood | ood | 0.0029133992282563735 | 0.9999370364507343 | sklearn-poly2-ridge | 0.05654462772172779 | 0.051523890874196586 |
| near-parallel-three-channel-noisy-16d | iid-uniform | iid | 0.003600342312263999 | 0.9997459345781228 | sklearn-poly2-ridge | 0.052894699949889996 | 0.06806622054146819 |
| near-parallel-three-channel-noisy-16d | wide-ood | ood | 0.005876071181397475 | 0.9995038453250289 | sklearn-poly2-ridge | 0.0955727477413446 | 0.061482706318126475 |
| near-parallel-three-channel-noisy-16d | shifted-ood | ood | 0.0036154298861450496 | 0.9997588639172782 | sklearn-poly2-ridge | 0.04845809868318247 | 0.0746094045039327 |
| weak-resolvable-three-channel-noisy-12d | iid-uniform | iid | 0.003728882150765997 | 0.9999181646486015 | sklearn-poly2-ridge | 0.04591792756369693 | 0.08120754460430137 |
| weak-resolvable-three-channel-noisy-12d | wide-ood | ood | 0.005471693586587925 | 0.999836382970466 | sklearn-mlp-regressor | 0.07593876567789458 | 0.07205402323494321 |
| weak-resolvable-three-channel-noisy-12d | shifted-ood | ood | 0.003676269568637131 | 0.9999223282952052 | sklearn-poly2-ridge | 0.04291126367078356 | 0.08567143575266337 |

## Prediction Baseline Families

- mean-output: constant (local)
- linear-ridge: linear ridge (local)
- quadratic-ridge: linear + squared coordinates ridge (local)
- random-gelu-ridge: random GELU feature ridge surrogate (local)
- sklearn-ridge: sklearn linear ridge (scikit-learn)
- sklearn-poly2-ridge: sklearn polynomial ridge (scikit-learn)
- sklearn-random-forest: sklearn random forest (scikit-learn)
- sklearn-extra-trees: sklearn extra trees (scikit-learn)
- sklearn-hist-gradient-boosting: sklearn histogram gradient boosting (scikit-learn)
- xgboost-hist: XGBoost histogram gradient trees (XGBoost)
- sklearn-mlp-regressor: sklearn neural network MLP regressor (scikit-learn)
- sklearn-pls-active-subspace-poly-ridge: sklearn active-subspace surrogate (scikit-learn)

## Gate Details

### balanced-three-channel-noisy-24d

- Profile: balanced-noisy
- Noise std: 0.0025
- Quantization: 0.001
- Query samples: 15
- Summary: `{"pass":true,"mae":0.0032677474320668303,"rmse":0.004301418014557782,"r2":0.999944613493118,"worstError":0.018258777737773624,"effectiveChannelCount":3,"canonicalAtomCount":3,"minDirectionProjection":0.9997152450697433,"duplicatePairCount":0,"nearParallelPairCount":0,"prunedWeakAtomCount":0,"bestPredictionBaselineId":"sklearn-poly2-ridge","bestPredictionBaselineMae":0.06013052581134432,"gclVsBestPredictionBaselineMaeRatio":0.05434423511145036,"maxDistributionMae":0.0051884080522364056,"minDistributionR2":0.9999316450434367,"maxDistributionWorstError":0.036268970516537546,"maxDistributionGclVsBestPredictionBaselineMaeRatio":0.05434423511145036,"worstDistributionId":"wide-ood","strongOnlyMae":0.22900940666322703,"rawOracleCalls":137970}`
- outputR2AboveThreshold: true
- outputMaeBelowThreshold: true
- worstErrorBelowThreshold: true
- oodR2AboveThreshold: true
- oodMaeBelowThreshold: true
- oodWorstErrorBelowThreshold: true
- effectiveChannelCountMatches: true
- canonicalAtomCountMatches: true
- directionRecoveryAboveThreshold: true
- noDuplicateCollapse: true
- weakResolvableSignalNotPruned: true
- beatsBestPredictionBaselineByMae: true
- beatsBestPredictionBaselineOnEveryDistribution: true
- packageBackedPredictionBaselinesCovered: true
- xgboostPredictionBaselineCovered: true
- nnPredictionBaselineCovered: true
- activeSubspacePredictionBaselineCovered: true
- nonlinearPackagePredictionBaselinesCovered: true
- rawOracleCallsBelowBudget: true

### near-parallel-three-channel-noisy-16d

- Profile: near-parallel-noisy
- Noise std: 0.002
- Quantization: 0.001
- Query samples: 11
- Summary: `{"pass":true,"mae":0.003600342312263999,"rmse":0.004711240353647738,"r2":0.9997459345781228,"worstError":0.019976668856817058,"effectiveChannelCount":3,"canonicalAtomCount":3,"minDirectionProjection":0.9995791895618263,"duplicatePairCount":0,"nearParallelPairCount":1,"prunedWeakAtomCount":0,"bestPredictionBaselineId":"sklearn-poly2-ridge","bestPredictionBaselineMae":0.052894699949889996,"gclVsBestPredictionBaselineMaeRatio":0.06806622054146819,"maxDistributionMae":0.005876071181397475,"minDistributionR2":0.9995038453250289,"maxDistributionWorstError":0.044916008556176834,"maxDistributionGclVsBestPredictionBaselineMaeRatio":0.0746094045039327,"worstDistributionId":"wide-ood","strongOnlyMae":0.2478805455002617,"rawOracleCalls":116468}`
- outputR2AboveThreshold: true
- outputMaeBelowThreshold: true
- worstErrorBelowThreshold: true
- oodR2AboveThreshold: true
- oodMaeBelowThreshold: true
- oodWorstErrorBelowThreshold: true
- effectiveChannelCountMatches: true
- canonicalAtomCountMatches: true
- directionRecoveryAboveThreshold: true
- noDuplicateCollapse: true
- weakResolvableSignalNotPruned: true
- beatsBestPredictionBaselineByMae: true
- beatsBestPredictionBaselineOnEveryDistribution: true
- packageBackedPredictionBaselinesCovered: true
- xgboostPredictionBaselineCovered: true
- nnPredictionBaselineCovered: true
- activeSubspacePredictionBaselineCovered: true
- nonlinearPackagePredictionBaselinesCovered: true
- rawOracleCallsBelowBudget: true

### weak-resolvable-three-channel-noisy-12d

- Profile: weak-resolvable-noisy
- Noise std: 0.003
- Quantization: 0.001
- Query samples: 13
- Summary: `{"pass":true,"mae":0.003728882150765997,"rmse":0.004872093510514659,"r2":0.9999181646486015,"worstError":0.04908248358529477,"effectiveChannelCount":3,"canonicalAtomCount":3,"minDirectionProjection":0.9987338205891503,"duplicatePairCount":0,"nearParallelPairCount":0,"prunedWeakAtomCount":0,"bestPredictionBaselineId":"sklearn-poly2-ridge","bestPredictionBaselineMae":0.04591792756369693,"gclVsBestPredictionBaselineMaeRatio":0.08120754460430137,"maxDistributionMae":0.005471693586587925,"minDistributionR2":0.999836382970466,"maxDistributionWorstError":0.09735167309682635,"maxDistributionGclVsBestPredictionBaselineMaeRatio":0.08567143575266337,"worstDistributionId":"wide-ood","strongOnlyMae":0.03284863477358395,"rawOracleCalls":155012}`
- outputR2AboveThreshold: true
- outputMaeBelowThreshold: true
- worstErrorBelowThreshold: true
- oodR2AboveThreshold: true
- oodMaeBelowThreshold: true
- oodWorstErrorBelowThreshold: true
- effectiveChannelCountMatches: true
- canonicalAtomCountMatches: true
- directionRecoveryAboveThreshold: true
- noDuplicateCollapse: true
- weakResolvableSignalNotPruned: true
- beatsBestPredictionBaselineByMae: true
- beatsBestPredictionBaselineOnEveryDistribution: true
- packageBackedPredictionBaselinesCovered: true
- xgboostPredictionBaselineCovered: true
- nnPredictionBaselineCovered: true
- activeSubspacePredictionBaselineCovered: true
- nonlinearPackagePredictionBaselinesCovered: true
- rawOracleCallsBelowBudget: true

## Conclusion

The noisy mixed GELU structure gate passed. GCL recovered explicit atom structure under stochastic noise/quantization, and that recovered structure also beat the dependency-free, scikit-learn, XGBoost, and sklearn-MLP prediction-sanity baselines across IID and OOD held-out noiseless truth. These prediction baselines are not structure-decompiler competitors.
