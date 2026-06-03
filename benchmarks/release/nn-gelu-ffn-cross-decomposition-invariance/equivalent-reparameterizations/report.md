# GELU FFN Cross-Decomposition Invariance Benchmark

Generated: 2026-06-03T07:28:19.947Z

## Claim Scope

- Proves if pass: GCL returns the same canonical recovered GELU structure for two distinct internal decompositions of the same clean function, under different solver seeds
- Does not claim: semantic identity of original hidden neurons, noisy invariance, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D LLM block decompilation, or broad SOTA status

## Teacher Audit

- Dimension: 16
- Canonical hidden atom count: 3
- Minimal hidden atom count: 3
- Overcomplete hidden atom count: 6
- Overcomplete hidden redundant atom count: 3
- Oracle delta on validation: 4.440892098500626e-16

## Summary

- Pass: true
- Effective channel counts: 3, 3
- Canonical atom counts: 3, 3
- Signature distance: 0.052726404715341284
- Max atom distance: 0.07144167430917188
- Max MAE: 0.0019376881321900733
- Min R2: 0.9999767700614473
- Min direction projection: 0.9998814649324326
- Model agreement MAE: 0.0021717078319734404
- Max raw oracle calls: 25710

## Per Representation

| Representation | Hidden atoms | Effective | Canonical | MAE | R2 | Min direction projection | Oracle calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minimal | 3 | 3 | 3 | 0.0019376881321900733 | 0.9999767700614473 | 0.9998814649324326 | 25710 |
| overcomplete-equivalent | 6 | 3 | 3 | 0.0007323187445132345 | 0.9999975825241255 | 0.9999956316903414 | 25710 |

## Gate

- oracleFunctionsEquivalent: true
- bothEffectiveCountsMatch: true
- bothCanonicalCountsMatch: true
- bothHeldoutR2Above09998: true
- bothHeldoutMaeBelow0008: true
- bothDirectionRecoveryStable: true
- canonicalSignatureInvariant: true
- modelPredictionsAgree: true
- rawOracleCallsBelow50000Each: true

## Conclusion

The cross-decomposition invariance gate passed. GCL recovered the same canonical GELU structure from two equivalent hidden decompositions of the same oracle function.
