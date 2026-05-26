# Legacy Artifact Audit

Last local audit: 2026-05-22.

This document classifies old report/log artifacts that predate the current
release artifact layout. These files are preserved for forensics, but they stay
outside the current release lane unless a current command regenerates them under
`benchmarks/release/<lane>/<variant>/` with a manifest.

## Scope

Audited paths:

| Path | Files observed | Verdict |
|---|---:|---|
| `REPORTS/` | 1 | Legacy forensic report; outside the current release lane. |
| `audit/` | 106 | Mixed old experiments, logs, datasets, broken JSON, and forensic traces. |
| `.gemini/antigravity` | 0 in this repo | No files found at this path during this audit. |
| root `output_*.log` | 11 | Legacy async-agent/co-evolution logs; outside the current release lane. |

## Classification

| Class | Files | Status | Notes |
|---|---|---|---|
| Current release evidence | None in this scope | Needs fresh rerun | Current evidence lives under `benchmarks/release/` with manifests. |
| Legacy forensic signal | `REPORTS/F3_AUTONOMOUS_VICTORY.md`, `audit/invariant_check.txt`, `audit/master_output.txt`, selected reauth/audit logs | Useful but not canonical | These can explain what a prior coder was trying to prove, but they need a fresh gate before becoming claims. |
| Santa Fe proof trail needing repair | `audit/fixed_run.json`, `audit/clean_best_tree.json`, `audit/final_proof.json`, `audit/verification_output_final*.txt`, `audit/FINAL_REPLAY*.txt`, `audit/koza_proof_output.txt` | Repair in progress | Some files report fitness `0.848904`, but replay verification reports only `50/89` food and fitness `0.5618`; other verifier files show `3/89`, `37/89`, or explicit reporting mismatch. `final_proof.json` is invalid JSON. |
| Failed/OOM diagnostics | `audit/diag-final.log`, `audit/diag-industrial.log`, `audit/ultimate_reauth_v2.log`, `audit/santafe_beyond_sota.json` | Repair diagnostic | These contain heap/OOM or hard runtime failures. They are useful repair inputs, not proof. |
| SOTA compliance diagnostics | `audit/diag-run.log`, `audit/diag-seeded.log`, `audit/diag-final-sota.log`, `audit/diag-rigorous-regimes.log` | Mixed/blocked | Several symbolic tasks pass, but logs include hard stops or final `SOME TESTS FAILED` status. |
| Quant/profit experiments | `audit/genetic_quant_*.log`, `audit/final_*profit*.log`, `audit/*alpha*.log`, `audit/*market_neutral*.log`, `audit/vision_fixed_alpha.log` | Needs repair/rerun | These are old trading experiments. Some have negative returns, no trades, NaN Sharpe, or benchmark-relative weakness. They need a corrected current protocol before financial claims. |
| Extrapolation old audit | `audit/extrapolation-results.txt`, `audit/extrapolation_data/*` | Mixed/diagnostic | Three functions show zero MSE, but F4 reports train MSE `0.00405306` and test MSE `22.47284231`, so this is not a clean extrapolation proof. |
| Root async-agent logs | `output_final*.log`, `output_payoff.log`, `output_revelation.log`, `output_smoking_gun.log`, `output_ultimate.log`, `output_phase*.log` | Diagnostic only | These show async bridge/co-evolution behavior. Many final fitness values are around `0.03` to `0.04` and agent utility is usually `0%`; one run throws `TypeError`. |

## Specific Findings

### `REPORTS/F3_AUTONOMOUS_VICTORY.md`

This report claims a verified autonomous F3-rational victory with 1.0 fitness
and a matching rational postfix signature. It may describe a real historical
fix, but it is not connected to a current release command or manifest. Treat it
as a historical forensic note until a current benchmark reproduces it.

### Santa Fe audit trail

The Santa Fe files are internally inconsistent:

| Artifact | Observed result | Boundary |
|---|---|---|
| `audit/fixed_run.json` | valid JSON, fitness `0.848904`, `test_food_pct=1`, rotated food `0.292135` | Needs independent replay; not enough by itself. |
| `audit/clean_best_tree.json` | same headline values as `fixed_run.json` | Needs independent replay. |
| `audit/final_proof.json` | invalid JSON | Cannot be canonical evidence. |
| `audit/verification_output_final_best.txt` | verified `37/89`, fitness `0.4157`, says mismatch continues | Does not support the headline claim. |
| `audit/verification_output_final_final.txt` | verified `50/89`, fitness `0.5618`, says mismatch continues | Does not support the headline claim. |
| `audit/FINAL_REPLAY.txt` | final replay `50/89`, fitness `0.5618` | Best available independent replay in this legacy set. |
| `audit/koza_proof_output.txt` | verified `11/89`, explicit failed note | Needs repair/rerun before any Santa Fe claim. |

Repair action: the canonical Santa Fe solver now uses the same 32x32 trail as
`trails/santafe.txt` and supports the legacy operation aliases used by these
old files (`if-food-ahead`, `progn2`, `progn3`, `left`, `right`). A current
rerun is still needed before promoting any old Santa Fe headline number.

### Industrial/SOTA diagnostics

Several `diag-*` logs show partial success on simple symbolic tasks, but they
are not clean gates:

- `audit/diag-run.log`: final status says some tests failed; Damped Oscillator
  misses the threshold.
- `audit/diag-seeded.log`: all visible rows pass, but final status still says
  some tests failed, so the artifact is inconsistent.
- `audit/diag-final-sota.log`: visible rows pass but final status still says
  some tests failed.
- `audit/diag-final.log` and `audit/diag-industrial.log`: contain heap/OOM
  failures.
- `audit/diag-rigorous-regimes.log`: many hard stops; useful stress log, not a
  pass artifact.

Verdict: useful for historical debugging; outside the current release lane until rerun with a manifest.

### Quant/profit logs

These files need a corrected current protocol before claims:

- `audit/genetic_quant_alpha.log`: strategy return `-57.58%` vs buy-and-hold
  `-3.93%`.
- `audit/genetic_quant_alpha_v2.log`: strategy return is positive but far below
  buy-and-hold in the same log.
- `audit/genetic_quant_alpha_v3.log`: `0` OOS transactions and `NaN` Sharpe.
- `audit/genetic_quant_sharpe.log`: strategy return `-7.86%` vs buy-and-hold
  `-3.93%`.
- `audit/vision_fixed_alpha.log`: strategy return `-43.60%` vs buy-and-hold
  `-3.93%`.
- `audit/final_real_profits.log`: claims market-beating OOS behavior, but the
  strategy return is still negative at `-1.96%` and the sample has only 10 OOS
  transactions.

Repair action: keep these as inputs for a future finance benchmark cleanup.
They should be rerun with explicit train/test splits, transaction-cost
accounting, and baseline comparison before any product claim.

### Root `output_*.log`

The root output logs are old async-agent/co-evolution traces. They are useful
  for debugging the async bridge history, but they need a current rerun before
  being used for model-capability claims:

- `output_final.log`: final fitness `0.036079`, agent utility `0%`.
- `output_final_proof.log`: final fitness `0.040460`, agent utility `0%`.
- `output_payoff.log`: final fitness `0.038113`, agent utility `0%`.
- `output_revelation.log`: final fitness `0.033460`, agent utility `0.60%`.
- `output_smoking_gun.log`: final fitness `0.033541`, agent utility `0%`.
- `output_ultimate.log`: final fitness `0.038213`, agent utility `0%`.
- `output_phase33.log`: throws `TypeError: Cannot read properties of undefined`.

Verdict: diagnostic input for future repair.

## Policy

1. Do not delete these files automatically; they may still explain historical
   work and bugs.
2. Do not cite them as current benchmark evidence until a current verifier reruns them.
3. Treat their titles as old labels, not current claim labels.
4. Any legacy result worth keeping must be re-run through a current script that
   writes:
   - `benchmarks/release/<lane>/<variant>/report.json`
   - `benchmarks/release/<lane>/<variant>/report.md`
   - `benchmarks/release/<lane>/<variant>/manifest.json`
5. Any promoted legacy result must include a command, date, pass/fail summary,
   failure boundary, and independent validation if the old trail contained a
   mismatch.

## Current Verdict

The legacy report/audit/output layer is now a repair queue. It does not
invalidate the current green release lanes, and it should be promoted only by
fixing/rerunning the relevant verifier with a current manifest.
