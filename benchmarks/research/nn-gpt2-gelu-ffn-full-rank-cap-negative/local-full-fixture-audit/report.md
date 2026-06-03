# Real GPT-2 Full-FFN Rank-Cap Audit

Generated: 2026-06-03T10:48:04.761Z

## Summary

- Pass: true
- Claim decision: insufficient-rank
- Fixture: /home/kugli/Downloads/Genetic-Code-Lab/external-pytorch-test/gpt2_full_ffn_v2.json
- Dimension: 768
- Source channels: 3072
- Rank cap: 20
- Top-rank source coverage: 0.034907769543268366
- R2 top-rank vs full: -0.0014440317377399925
- MAE top-rank vs full: 4.982383447755312
- Worst error: 16.26971003902171

## Decision Reasons

- rank cap 20 is below full GPT-2 source channel count 3072
- top-rank contribution coverage is 0.034908
- held-out capped-vs-full R2 is -0.001444

## Boundary

This is a local research audit over the full GPT-2 fixture. It is not a release-gated full-FFN extraction claim.
