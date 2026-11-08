# Kaggle T4 evidence day 2: quadratic-identity witness + lane F post-mortem (2026-11-08)

Continues `README-kaggle-t4-2026-11-07.md` (lanes A–E, kernels v3/v5). Two
things happened on 2026-11-08 in headless kernel `kuglipista/gcl-swiglu-lane`:

1. **Kernel v11 landed a new evidence class**: the oracle-only
   **SwiGLU quadratic-identity structural witness** passed on the real
   quantized LLaMA-3.2-3B layer-1 block — the first GCL evidence that
   certifies *structure* (bias-free SiLU gating) from `f(x)` queries alone,
   rather than reconstructing parameters.
2. **Lane F v1 (128-unit block, 8x lane E) failed**, and the failure is the
   informative kind: the fit interpolated the training set to machine
   precision while validation R2 was −39.6 — an identifiability failure by
   configuration, not a solver defect. Kernel v12 re-launches lane F at
   48 units with lane-E-matched data density.

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
not recover parameters. Lane F v2, if it passes, extends the recorded
selected-unit-block recovery evidence to 48 units — it is still not
full-block recovery (8192 units), which remains an open problem per the
research note. The v1 failure is evidence about configuration scaling, not
about the solver's correctness at recorded scales.
