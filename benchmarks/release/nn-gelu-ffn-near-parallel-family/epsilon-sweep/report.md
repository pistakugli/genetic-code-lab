# Near-Parallel Mixed GELU FFN Family Benchmark

Generated: 2026-06-05T15:58:27.922Z

## Claim Scope

- Proves if pass: GCL can keep mixed GELU atom count, direction recovery, and near-parallel diagnostics stable as two channels become nearly parallel in a controlled clean family
- Does not claim: noisy mixed-block recovery, full GPT-2 FFN extraction, arbitrary 3072-channel recovery, arbitrary 4096D production LLM block decompilation, SwiGLU mixed-block recovery, or broad SOTA status

## Summary

- Pass: true
- Epsilons: 0.1, 0.01, 0.001, 0.0001
- Effective channel counts: 3, 3, 3, 3
- Duplicate pair counts: 0, 0, 0, 0
- Near-parallel pair counts: 0, 1, 1, 1
- Max MAE: 0.002510633422272873
- Min R2: 0.999883779624714
- Min direction projection: 0.9993621598125698
- Max raw oracle calls: 18248

## Epsilon Rows

- eps=0.1: pass=true, effective=3, nearPairs=0, duplicates=0, MAE=0.0022200191372670737, R2=0.9999024875553998, minProjection=0.9993621598125698
- eps=0.01: pass=true, effective=3, nearPairs=1, duplicates=0, MAE=0.002510633422272873, R2=0.999883779624714, minProjection=0.9998153653667718
- eps=0.001: pass=true, effective=3, nearPairs=1, duplicates=0, MAE=0.001982121469452344, R2=0.9999177758313744, minProjection=0.9998106521813376
- eps=0.0001: pass=true, effective=3, nearPairs=1, duplicates=0, MAE=0.00198939196832835, R2=0.9999280021555399, minProjection=0.9999023005713468

## Conclusion

The near-parallel family gate passed. GCL preserved explicit mixed GELU structure recovery while detecting near-parallel channel geometry.
