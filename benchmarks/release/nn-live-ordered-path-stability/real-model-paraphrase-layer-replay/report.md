# Live Ordered Path Stability Suite

This suite replays discovered real-model circuits across paraphrase and neighboring-layer controls. It does not fit a surrogate.

- Pass: `true`
- Passed: `4/4`

| model | status | cases | paraphrase pass | layer specificity | failed subjects | report |
|---|---:|---:|---:|---:|---|---|
| GPT-2 | pass | 5/5 | 1 | 1 | [] | benchmarks/release/nn-live-ordered-path-stability/real-model-paraphrase-layer-replay/gpt2/report.json |
| Pythia/GPT-NeoX | pass | 5/5 | 0.9333333333333333 | 1 | [] | benchmarks/release/nn-live-ordered-path-stability/real-model-paraphrase-layer-replay/pythia/report.json |
| Llama SwiGLU | pass | 4/4 | 1 | 1 | [] | benchmarks/release/nn-live-ordered-path-stability/real-model-paraphrase-layer-replay/llama/report.json |
| Mistral SwiGLU | pass | 4/4 | 1 | 1 | [] | benchmarks/release/nn-live-ordered-path-stability/real-model-paraphrase-layer-replay/mistral/report.json |
