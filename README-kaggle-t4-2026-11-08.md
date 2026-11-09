# Kaggle T4 evidence day 2: quadratic-identity witness + lane F post-mortem (2026-11-08)

Continues `README-kaggle-t4-2026-11-07.md` (lanes A–E, kernels v3/v5). Three
things happened on 2026-11-08 in the headless lane kernels:

1. **Kernel v11 landed a new evidence class**: the oracle-only
   **SwiGLU quadratic-identity structural witness** passed on the real
   quantized LLaMA-3.2-3B layer-1 block — the first GCL evidence that
   certifies *structure* (bias-free SiLU gating) from `f(x)` queries alone,
   rather than reconstructing parameters.
2. **Kernel v13 extended the witness to all three model families**: after a
   witness environment regression in kernel v12 (the oracle subprocess did not
   inherit `HF_HOME`, so both new witnesses died `LocalEntryNotFoundError`
   before a single oracle call), kernel v13 reran them with end-to-end env
   propagation — **Mistral-7B (4096D) and Qwen2.5-7B (3584D) both PASS**, all
   identity residuals at ~1e-6 relative against a 0.02 tolerance.
3. **Lane F v1 (128-unit block, 8x lane E) failed**, and the failure is the
   informative kind: the fit interpolated the training set to machine
   precision while validation R2 was −39.6 — an identifiability failure by
   configuration, not a solver defect. Kernel v12 re-launched lane F at
   48 units with lane-E-matched data density; that session was killed by a
   version push over the running kernel and is being re-run (see the incident
   note below).

## Quadratic-identity witness (kernel v11, PASS)

`tools/llm_reader/swiglu_quadratic_witness.py` — for a bias-free SwiGLU
block, `silu(s) − silu(−s) = s` exactly, so the even part
`E(x) = (f(x) + f(−x))/2` is exactly a vector-valued quadratic map for **any**
weights, including 4-bit-quantized providers. The witness certifies this from
`f(x)`-only queries (homogeneity t=2/t=3, three-point polarization,
parallelogram, origin norm); a biased or non-SwiGLU-gated block fails at O(1)
relative error.

| check | median rel. residual | p90 | max | gate |
| --- | --- | --- | --- | --- |
| homogeneity-t2 | 7.329e-07 | 7.562e-07 | 7.930e-07 | pass |
| homogeneity-t3 | 1.242e-06 | 1.297e-06 | 1.383e-06 | pass |
| polarization | 1.404e-06 | 1.461e-06 | 1.580e-06 | pass |
| parallelogram | 1.171e-06 | 1.213e-06 | 1.278e-06 | pass |
| originNormRelative | 0.000e+00 | – | – | pass |

- Model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` layer 1,
  full-mlp-vector mode (all 8192 units), 256 checks, scale 0.35,
  Park–Miller seed 260621, tolerance 0.02.
- **6,145 oracle calls; weights never read.** 11.5 s on the T4.
- Evidence: `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/quadratic-witness-llama-3072d-layer1-kaggle-t4-2026-11-08/`
  (report.json, report.md, manifest.json; produced at repo commit `a9e96eb`).

## Lane F v1 post-mortem (kernel v11, FAIL — recorded, not landed)

Configuration: LLaMA-3.2-3B layer 1, 128 LCG-drawn units (seed 260621),
128 atoms, activeRank 256, train 32,000, validation 8,192, restarts 256,
case timeout 5 h.

Measured (kernel progress log, solver-relative time):

| event | t | value |
| --- | --- | --- |
| batched fit entered | 17 s | atomCount 128, rows 32,000 |
| initial fit complete | 13,429 s | **fitTrainMae 4.06e-14** (machine zero) |
| validation complete | 13,430 s | **R2 −39.61, MAE 3.75, worst error 20.5** |
| CEGIS cycle 1 | 17,019 s | R2 −40.06, 0 atoms accepted, 231,969 oracle calls |
| killed | 18,000 s | case timeout |

Perfect train fit with catastrophic validation is the signature of an
**underdetermined problem**: the fit has 128 × (2·256 + 1) = 65,664 free
parameters against 32,000 rows — 0.49 rows per parameter. Per atom it had 250
rows, below even the 3–4m spurious-solution zone and far below the ~6m fiber
bound recorded in `docs/research/2026-11-08-fullblock-joint-swiglu-recovery.md`.
Lane E, for contrast, ran 16,000 rows against 1,040 parameters — 15.4 rows
per parameter. The lane F v1 configuration scaled the block 8x but the data
only 2x, contradicting the session's own lower-bound analysis. The failure
was predictable from the recorded theory; it is recorded here so it is not
repeated.

## Lane F v2 (kernel v12, launched 2026-11-08)

Same recorded lane E solver environment, scaled with **constant
identifiability margins** instead of constant budget:

| | lane E (pass) | lane F v1 (fail) | lane F v2 (v12) |
| --- | --- | --- | --- |
| units = atoms | 16 | 128 | **48** (3x lane E) |
| activeRank | 32 | 256 | 96 |
| free parameters | 1,040 | 65,664 | 9,264 (8.9x lane E) |
| train rows | 16,000 | 32,000 | **92,640** |
| rows / parameter | 15.4 | 0.49 | **10.0** |
| rows / atom | 1,000 | 250 | **1,930** (1.9x lane E) |
| vs 6m fiber bound | 10.4x | 0.33x | **6.7x** |
| direction count | 384 | 1,024 | 1,152 (12/rank-dim, lane E parity) |
| fit restarts | 512 | 256 | 512 (lane E parity) |
| structured starts | on | off | **on** (lane E parity) |
| oracle budget | 300,000 | 2,400,000 | 2,400,000 |
| case timeout | 1 h | 5 h | 8 h |

- Units: the first 48 distinct Park–Miller draws (seed 260621) over the 8192
  intermediate units — a verified subset of the 128 units drawn for lane F v1
  (first 113, last 8143), so v2 recovers a sub-block of the very block v1
  attempted, not a freshly cherry-picked one.
- Train rows = 10 × parameters exactly (92,640 = 10 × 9,264), chosen over the
  15.4x lane E ratio to keep the projected fit (~2.5 h) far inside the 8 h
  case timeout with room for the adaptive post-fit stages.
- Kernel v12 additionally runs the quadratic-identity witness on the Mistral
  7B (4096D) and Qwen2.5-7B (3584D) layer-1 blocks — extending the new
  evidence class to all three model families in one session.
- The kernel source lives in the repository at
  `scripts/gpu/kaggle/swiglu-lane/kernel.py` (pushed to
  `kuglipista/gcl-swiglu-lane` as version 12), so it survives sandbox
  recycling; the v11 source had to be recovered from Kaggle.

## Quadratic-identity witnesses, Mistral + Qwen (kernel v13, PASS)

Kernel v12 ran both new witnesses cache-blind: the kernel downloaded the
providers into `HF_HOME=/tmp/hf-cache` and passed that env to the downloads
and the lane F solver, but **not** to the witness tool — whose oracle
subprocess inherits the parent env — so the oracle looked in the default
empty cache and died `LocalEntityNotFoundError` under `--local-files-only`,
before a single oracle call (6.1–6.6 s each). Kernel v13 (commit `964dbed`,
headless kernel `kuglipista/gcl-swiglu-witness`, 180.5 s total, exit 0) is a
witness-only rerun with the env propagated end-to-end
(kernel → witness → oracle) plus a hard pre-flight gate that refuses to run
a cache-blind witness.

Mistral-7B (`unsloth/mistral-7b-instruct-v0.3-bnb-4bit`) layer 1,
full-mlp-vector mode (all 14,336 units), 256 checks, scale 0.35,
seed 260621, tolerance 0.02 — **6,145 oracle calls, 13.5 s on the T4**:

| check | median rel. residual | p90 | max | gate |
| --- | --- | --- | --- | --- |
| homogeneity-t2 | 8.574e-07 | 8.976e-07 | 1.528e-06 | pass |
| homogeneity-t3 | 1.484e-06 | 1.582e-06 | 2.028e-06 | pass |
| polarization | 1.694e-06 | 1.828e-06 | 2.598e-06 | pass |
| parallelogram | 1.388e-06 | 1.483e-06 | 2.002e-06 | pass |
| originNormRelative | 0.000e+00 | – | – | pass |

Qwen2.5-7B (`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`) layer 1,
full-mlp-vector mode (all 18,944 units), same protocol — **6,145 oracle
calls, 13.0 s on the T4**:

| check | median rel. residual | p90 | max | gate |
| --- | --- | --- | --- | --- |
| homogeneity-t2 | 7.339e-07 | 7.724e-07 | 8.193e-07 | pass |
| homogeneity-t3 | 1.307e-06 | 1.387e-06 | 1.490e-06 | pass |
| polarization | 1.463e-06 | 1.550e-06 | 1.629e-06 | pass |
| parallelogram | 1.045e-06 | 1.109e-06 | 1.161e-06 | pass |
| originNormRelative | 0.000e+00 | – | – | pass |

All identity residuals sit four orders of magnitude below the 0.02 gate,
matching the LLaMA result — the bias-free SiLU-gated structure certificate
now covers **all three model families** (LLaMA 3072D, Mistral 4096D, Qwen
3584D). Both runs declare the full oracle-only audit
(`solverAccess=oracle-only`, weights/hooks/source-ids false,
`postHocWitnessOnly=true`).

- Evidence:
  `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/quadratic-witness-mistral-4096d-layer1-kaggle-t4-2026-11-08/`
  and
  `…/quadratic-witness-qwen-3584d-layer1-kaggle-t4-2026-11-08/`
  (report.json, report.md, manifest.json; produced at repo commit `964dbed`).
- Run summary (`allLanesPassed: true`) and the full kernel log are archived
  in the run output; the kernel source is the repository's
  `scripts/gpu/kaggle/swiglu-lane/kernel.py` at `964dbed`, pushed under the
  fresh slug `kuglipista/gcl-swiglu-witness` (kernel version 1).

## Lane F v2 session loss (v12/v13 incident, 2026-11-08)

The lane F v2 case in kernel v12 was unaffected by the witness env bug (the
lane solver received the env) and was progressing normally (48/48 atoms fit,
gradient scoring phase, ~491 s solver-relative at last observation). It was
killed when kernel v13 was pushed **over the running session** on the same
slug — Kaggle aborts a running version on a new push — and the
`gcl-swiglu-lane` slug subsequently became inaccessible to its owner via the
API (403 on kernels get/status/output, hidden from `kernels list --mine`),
so the partial run could not be harvested. Lessons applied going forward:
never push over a running kernel; run each long case under a dedicated fresh
slug; harvest the output before any further push. Lane F v2 is re-launched
unchanged (same units, argv, and recorded lane E environment) under its own
slug.

## Lane F v2 result (kernel v14, FAIL — recorded failure frontier)

Re-run under the dedicated slug `kuglipista/gcl-swiglu-lane-f` (repo commit
`521969a`, clean status): the case ran its full course — 25,758.4 s
(7.15 h) of solver time, 259,841 raw oracle calls, and **failed** every
recovery gate:

| gate | required | measured |
| --- | --- | --- |
| validation R2 | ≥ 0.995 | **0.0104** |
| MAE | ≤ 0.015 | **0.358** |
| worst error | ≤ 0.12 | **1.996** |
| subspace min principal cosine | ≥ 0.999 | **0.00126** |
| paired atoms (joint cosine ≥ 0.99) | 48 | **0** |

Unlike lane F v1 (which interpolated the training set to machine zero and
failed only on validation — spurious-solution signature), v2's fit did not
even reach train interpolation quality; validation R2 ≈ 0.01 means the
recovered 48-atom model explains ~1% of block variance. Solver phases from
the kernel log: basis recovery succeeded immediately (activeRank 96,
energyRatio 0.9999999999999992 — the linear-algebra stage is not the
bottleneck); the joint nonlinear fit then spent 23,002 s before validation;
two CEGIS cycles accepted **0 atoms** each (`replaced: false`); the
weak-gate analysis flagged 24 of 48 atoms with weak gate norms and refit
attempts on the 8-atom weak block did not move any gate. The access audit
is clean (`solverAccess=oracle-only`, weights/hooks/source-ids false,
truth locked until recovery frozen), so this is a genuine solver-frontier
result, not an access violation: **the recorded greedy-atom joint recovery
does not scale from 16 to 48 units, even at 10 rows/parameter with lane-E
probe geometry and restart counts.** The identifiability-matched data
density was necessary but not sufficient; the bottleneck is the non-convex
joint fit itself (48 interacting atoms on a rugged landscape), not data.

Evidence:
`benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-lcg48-48atom-forward-coordinate-oracle-only-kaggle-t4-2026-11-08/`
(report.json, report.md, manifest.json; status `fail`).

## Environment (kernel v11)

- GPU: Tesla T4 (14.56 GB, sm_75), CUDA build 12.8; torch 2.10.0+cu128,
  Python 3.12.3 venv (uv-pinned `requirements/gcl-python-gpu.txt`)
- Run window: 2026-11-08 03:08:20–08:10:43 UTC, 18,148.3 s total; exit 1
  (`RuntimeError: lane F or the quadratic-identity witness failed
  verification` — the witness passed, lane F timed out).
- Repo commit at run: `a9e96eb` (clean status).

## Claim boundary

The witness certifies the target *structure* (bias-free SiLU-gated even
part) of one layer of each model family, from `f(x)`-only access; it does
not recover parameters — that certificate now covers LLaMA-3.2-3B,
Mistral-7B-v0.3 and Qwen2.5-7B layer-1 blocks (kernels v11 + v13). Lane F
v2, if it passes, extends the recorded selected-unit-block recovery
evidence to 48 units — it is still not full-block recovery (8192 units),
which remains an open problem per the research note. The v1 failure is
evidence about configuration scaling, not about the solver's correctness at
recorded scales.
