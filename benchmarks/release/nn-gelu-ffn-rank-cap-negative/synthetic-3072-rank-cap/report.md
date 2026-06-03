# GELU FFN Rank-Cap Negative Guard

Generated: 2026-06-03T11:08:04.933Z

## Purpose

This gate strengthens GCL by preventing a false full-slice structure claim when the rank budget is too small.

## Summary

- Pass: true
- Claim decision: insufficient-rank
- Dimension: 768
- Source channels: 3072
- Observable groups: 48
- Rank cap: 12
- Effective recovered atoms: 11
- R2: 0.8744173553254918
- MAE: 1.4944170756394612
- Worst error: 7.696315351510922
- Top-rank source coverage: 0.3290814157449527
- Raw oracle calls: 79172

## Decision Reasons

- rank cap 12 is below required observable group count 48
- top-rank source coverage 0.329081 is below 0.70

## Conclusion

The negative guard passed: GCL did not promote an unsupported full-slice structure claim under an insufficient rank cap.
