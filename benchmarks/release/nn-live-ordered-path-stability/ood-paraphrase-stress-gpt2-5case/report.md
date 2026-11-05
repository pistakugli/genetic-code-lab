# Live Ordered Path Stability Suite

This suite replays discovered real-model circuits across paraphrase and neighboring-layer controls. It does not fit a surrogate.

- Pass: `false`
- Passed: `0/1`

- Paraphrase profile: `ood`

| model | status | cases | paraphrase pass | layer specificity | failed subjects | report |
|---|---:|---:|---:|---:|---|---|
| GPT-2 | fail | 2/5 | 0.43333333333333335 | 1 | Germany,Portugal,Greece | benchmarks/release/nn-live-ordered-path-stability/ood-paraphrase-stress-gpt2-5case/gpt2/report.json |
