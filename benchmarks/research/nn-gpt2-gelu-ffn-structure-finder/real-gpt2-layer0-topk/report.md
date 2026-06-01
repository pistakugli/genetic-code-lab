# GPT-2 GELU FFN Structure Finder Research Benchmark

Generated: 2026-05-31T17:54:35.246Z

## Scope

- Proves if pass: GCL can recover explicit GELU atom directions and high-fidelity outputs for a real GPT-2 single neuron and top-1/top-3 truncated output-channel sums on this fixture
- Does not claim: full GPT-2 FFN extraction, tokenizer integration, arbitrary layer extraction, arbitrary channel-count recovery, SwiGLU recovery, or release-gated production status

## Summary

- Pass: true
- Cases: 4/4
- Min R2: 0.9950578044301109
- Max MAE: 0.04830918330347916
- Min direction projection: 0.999868525265573

## Results

### real-gpt2-single-gelu-768d

- Pass: true
- R2: 0.9999539985740798
- MAE: 0.00455760482340957
- Raw oracle calls: 18660
- Projection: 0.9999999822798988

### real-gpt2-top1-gelu-ffn-768d

- Pass: true
- R2: 0.9973186489709484
- MAE: 0.02532853953376606
- Raw oracle calls: 22232
- Projection: 0.9999999767034381

### real-gpt2-top2-gelu-ffn-768d

- Pass: true
- R2: 0.9967983607537667
- MAE: 0.03459568839721633
- Raw oracle calls: 26104
- Projection: 0.999915848324277

### real-gpt2-top3-gelu-ffn-768d

- Pass: true
- R2: 0.9950578044301109
- MAE: 0.04830918330347916
- Raw oracle calls: 38756
- Projection: 0.999868525265573
