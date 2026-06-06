# Real GPT-2 GELU FFN Source Extractor Benchmark

Generated: 2026-06-06T07:58:08.408Z

## Claim Scope

- Proves if pass: GCL can build and validate an exact explicit GELU source graph for a real GPT-2 FFN output slice from an instrumented source manifest, then map every atom to exact GPT-2 channel ids/layers/indices with strict provenance.
- Does not claim: output-only original-neuron identity, blind full-FFN decompilation, arbitrary 3072-channel GPT-2 extraction, tokenizer/model execution, or broad SOTA status.

## Summary

- Pass: true
- Top-K: 20
- Dimension: 768
- Source channels: 1614, 2452, 2053, 486, 2172, 1981, 137, 257, 1554, 2211, 638, 1984, 2733, 1784, 2547, 1311, 1476, 2836, 763, 1169
- Max MAE: 4.561527660225847e-17
- Min R2: 1
- Max worst error: 1.7763568394002505e-15
- Provenance confidence min: 0.9999999999999998
- Raw validation oracle calls: 3600

## Distributions

- iid-gpt2-activation-scale: R2=1, MAE=4.561527660225847e-17, worst=1.7763568394002505e-15
- wide-ood: R2=1, MAE=3.833738881908744e-17, worst=1.7763568394002505e-15
- shifted-ood: R2=1, MAE=4.007789470665083e-17, worst=8.881784197001252e-16

## Conclusion

The real GPT-2 source extractor gate passed. GCL produced an exact explicit GELU source graph from a real GPT-2 FFN source manifest and verified every source channel with strict provenance.
