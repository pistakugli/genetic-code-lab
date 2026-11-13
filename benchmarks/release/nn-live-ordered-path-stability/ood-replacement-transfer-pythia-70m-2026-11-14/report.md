# Live Ordered Path OOD Replacement-Transfer Suite (Pythia-70m, CPU)

Replays the recorded Pythia-70m ordered residual path across six held-out OOD paraphrase families with layer-shift rescue, replacement-path search, and the why-report layer attached to every recorded break.

- Pass: `False`
- Passed: `0/1`

- Paraphrase profile: `ood`
- Device: `cpu` (live model, no surrogate)

| model | status | cases | paraphrase pass | layer specificity | failed subjects | report |
|---|---:|---:|---:|---:|---|---|
| Pythia/GPT-NeoX | fail | 0/1 | 0.16666666666666666 | 1.0 | France | pythia/report.json |

## Why the path broke

See `pythia/report.md` for the per-break why-report: break classification, layer-shift rescue outcome, replacement-path comparison, and the claim boundary.
